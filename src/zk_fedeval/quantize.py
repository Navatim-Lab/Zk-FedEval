"""Post-training int8 quantization of the exported CNN for the integer SP1 guest.

Reads the `model.onnx` and `eval_data.npz` that `ClientApp.evaluate` exports, folds
the input normalization into conv1, quantizes weights per output channel to int8 and
activations to u8, and writes `model_int8.bin` for `zk-fedeval-prover/program-int8`.

It also runs the float model and the integer model side by side and reports their
top-1 agreement, and writes the integer model's predictions so the Rust native twin
can be checked for bit-exactness.

Usage:
    uv run python -m zk_fedeval.quantize [outputs/eval/partition-0]
"""

import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"ZKFEI8\x00\x01"
SHIFT = 31
INT8_MODEL_NAME = "model_int8.bin"
INT8_PREDICTIONS_NAME = "int8_predictions.npy"
LAYER_ORDER = ("conv1", "conv2", "fc1", "fc2", "fc3")


def load_float_layers(onnx_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read (weight, bias) for the two Conv and three Gemm nodes, in graph order."""
    model = onnx.load(str(onnx_path))
    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            layers.append((inits[node.input[1]], inits[node.input[2]]))
        elif node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            weight = inits[node.input[1]]
            layers.append((weight if trans_b else weight.T, inits[node.input[2]]))
    if len(layers) != len(LAYER_ORDER):
        raise ValueError(f"expected 2 Conv + 3 Gemm nodes, found {len(layers)}")
    return {name: (w.astype(np.float64), b.astype(np.float64)) for name, (w, b) in zip(LAYER_ORDER, layers)}


def to_raw_pixels(images: np.ndarray) -> np.ndarray:
    """Undo `Normalize((0.5,) * 3, (0.5,) * 3)` after `ToTensor()`: x = p / 127.5 - 1."""
    return np.rint((images.astype(np.float64) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def im2col(x: np.ndarray, k: int = 5) -> np.ndarray:
    """[n, c, h, w] -> [n, oh * ow, c * k * k], valid padding, stride 1."""
    n, c, h, w = x.shape
    oh, ow = h - k + 1, w - k + 1
    cols = np.empty((n, oh * ow, c * k * k), dtype=x.dtype)
    for ic in range(c):
        for ky in range(k):
            for kx in range(k):
                patch = x[:, ic, ky : ky + oh, kx : kx + ow].reshape(n, -1)
                cols[:, :, ic * k * k + ky * k + kx] = patch
    return cols


def conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Valid 5x5 convolution without bias: [n, c, h, w] -> [n, out, oh, ow]."""
    n, _, h, _ = x.shape
    oh = h - 4
    out = im2col(x) @ w.reshape(w.shape[0], -1).T
    return out.transpose(0, 2, 1).reshape(n, w.shape[0], oh, oh)


def pool(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def float_forward(layers, pixels: np.ndarray, collect: bool = False):
    """Reference float forward pass on raw pixels, with normalization folded into conv1."""
    acts = []
    x = pixels.astype(np.float64) / 127.5 - 1.0
    for name in ("conv1", "conv2"):
        w, b = layers[name]
        x = pool(np.maximum(conv(x, w) + b[None, :, None, None], 0.0))
        acts.append(x)
    x = x.reshape(x.shape[0], -1)
    for name in ("fc1", "fc2"):
        w, b = layers[name]
        x = np.maximum(x @ w.T + b, 0.0)
        acts.append(x)
    w, b = layers["fc3"]
    logits = x @ w.T + b
    return (logits, acts) if collect else logits


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = w.reshape(w.shape[0], -1)
    scale = np.abs(flat).max(axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def fixed_point(ratio: np.ndarray) -> np.ndarray:
    m = np.rint(ratio * (1 << SHIFT)).astype(np.int64)
    if (m <= 0).any() or (m >= 1 << 31).any():
        raise ValueError("requantization multiplier out of i32 range")
    return m.astype(np.int32)


def quantize(layers, pixels: np.ndarray):
    """Return [(name, q_weight, q_bias, multiplier)] for the integer guest."""
    # Fold x = p / 127.5 - 1 into conv1 so the guest consumes raw u8 pixels at scale 1.
    w1, b1 = layers["conv1"]
    folded = dict(layers)
    folded["conv1"] = (w1 / 127.5, b1 - w1.reshape(w1.shape[0], -1).sum(axis=1))

    _, acts = float_forward(layers, pixels, collect=True)
    act_scales = [max(float(a.max()), 1e-8) / 255.0 for a in acts]

    quantized = []
    in_scale = 1.0
    for i, name in enumerate(LAYER_ORDER):
        w, b = folded[name]
        q_w, w_scale = quantize_per_channel(w)
        acc_scale = w_scale * in_scale
        q_b = np.rint(b / acc_scale).astype(np.int32)
        if name == "fc3":
            # Logits only feed argmax, so any common positive scale works; halve it
            # so the largest multiplier stays below 2^31.
            multiplier = fixed_point(acc_scale / (2 * acc_scale.max()))
        else:
            # Floor the output scale so every multiplier stays below 2^31. It only binds
            # for a dead layer (ReLU zero on all calibration data), where any scale works.
            out_scale = max(act_scales[i], 2 * float(acc_scale.max()))
            multiplier = fixed_point(acc_scale / out_scale)
            in_scale = out_scale
        quantized.append((name, q_w, q_b, multiplier))
    return quantized


def requant(acc: np.ndarray, m: np.ndarray) -> np.ndarray:
    return ((acc * m.astype(np.int64) + (1 << (SHIFT - 1))) >> SHIFT).clip(0, 255)


def int_forward(quantized, pixels: np.ndarray) -> np.ndarray:
    """Exact integer reference of the guest's arithmetic. Returns predicted classes."""
    q = {name: (w, b, m) for name, w, b, m in quantized}
    x = pixels.astype(np.int64)
    for name in ("conv1", "conv2"):
        w, b, m = q[name]
        acc = conv(x, w.astype(np.int64)) + b.astype(np.int64)[None, :, None, None]
        x = pool(requant(acc, m[None, :, None, None]))
    x = x.reshape(x.shape[0], -1)
    for name in ("fc1", "fc2"):
        w, b, m = q[name]
        x = requant(x @ w.astype(np.int64).T + b.astype(np.int64), m)
    w, b, m = q["fc3"]
    logits = (x @ w.astype(np.int64).T + b.astype(np.int64)) * m.astype(np.int64)
    return logits.argmax(axis=1)


def write_model(quantized, path: Path) -> None:
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", SHIFT))
        for _, w, b, m in quantized:
            f.write(w.astype("<i1").tobytes())
            f.write(b.astype("<i4").tobytes())
            f.write(m.astype("<i4").tobytes())


def main(export_dir: Path) -> None:
    layers = load_float_layers(export_dir / "model.onnx")
    data = np.load(export_dir / "eval_data.npz")
    pixels = to_raw_pixels(data["images"])
    labels = data["labels"]

    quantized = quantize(layers, pixels)
    write_model(quantized, export_dir / INT8_MODEL_NAME)

    float_pred = float_forward(layers, pixels).argmax(axis=1)
    int_pred = int_forward(quantized, pixels)
    np.save(export_dir / INT8_PREDICTIONS_NAME, int_pred.astype(np.int64))

    n = len(labels)
    print(f"wrote {export_dir / INT8_MODEL_NAME} ({(export_dir / INT8_MODEL_NAME).stat().st_size} bytes)")
    print(f"samples:            {n}")
    print(f"float accuracy:     {(float_pred == labels).mean():.4f}")
    print(f"int8 accuracy:      {(int_pred == labels).mean():.4f}")
    print(f"top-1 agreement:    {(float_pred == int_pred).mean():.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/eval/partition-0"))
