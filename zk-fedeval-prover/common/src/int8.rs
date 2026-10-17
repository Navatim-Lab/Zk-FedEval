//! Integer-only version of the CNN, shared by the int8 SP1 guest (`program-int8`)
//! and the host's native twin so both compute bit-identical results.
//!
//! The model file is written by `zk_fedeval.quantize`: int8 weights per output channel,
//! i32 biases, and i32 fixed-point requantization multipliers. Activations are u8.
//!
//! SP1 has no floating-point unit, so this replaces ~120 emulated cycles per
//! multiply-add with a few integer instructions. Two output channels share one 64-bit
//! multiply: weights are offset to unsigned (`w + 128`) and packed as `lo | hi << 32`,
//! then each lane is corrected by `128 * sum(inputs)`. Every lane sum stays below
//! 2^32 (at most 400 * 255 * 255), so lanes never carry into each other.

use crate::{IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH};

pub const MAGIC: &[u8; 8] = b"ZKFEI8\x00\x01";
pub const PIXELS_PER_IMAGE: usize = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;

/// (outputs, inputs per output) for conv1, conv2, fc1, fc2, fc3.
const LAYER_SHAPES: [(usize, usize); 5] = [(6, 75), (16, 150), (120, 400), (84, 120), (100, 84)];

struct Layer {
    outputs: usize,
    inputs: usize,
    /// `[outputs / 2][inputs]` packed channel pairs.
    packed: Vec<u64>,
    /// Conv layers only: the same weights as `[window row][pair][5]`, so the conv
    /// kernel reads every pair's weights for one 5-pixel row contiguously.
    by_row: Vec<u64>,
    bias: Vec<i64>,
    multiplier: Vec<i64>,
}

impl Layer {
    fn parse(bytes: &[u8], at: &mut usize, outputs: usize, inputs: usize) -> Result<Self, &'static str> {
        let weights = take(bytes, at, outputs * inputs)?;
        let bias = read_i32s(take(bytes, at, outputs * 4)?);
        let multiplier = read_i32s(take(bytes, at, outputs * 4)?);

        let unsigned = |w: &u8| (*w as i8 as i64 + 128) as u64;
        let mut packed = Vec::with_capacity(outputs / 2 * inputs);
        for pair in weights.chunks_exact(2 * inputs) {
            let (lo, hi) = pair.split_at(inputs);
            packed.extend(lo.iter().zip(hi).map(|(l, h)| unsigned(l) | (unsigned(h) << 32)));
        }
        Ok(Self { outputs, inputs, packed, by_row: Vec::new(), bias, multiplier })
    }

    /// Fill `by_row` from `packed`; `inputs` is `channels * 25` for a 5x5 conv.
    fn with_rows(mut self) -> Self {
        let pairs = self.outputs / 2;
        let rows = self.inputs / 5;
        self.by_row = Vec::with_capacity(pairs * self.inputs);
        for row in 0..rows {
            for pair in 0..pairs {
                let at = pair * self.inputs + row * 5;
                self.by_row.extend_from_slice(&self.packed[at..at + 5]);
            }
        }
        self
    }

    /// Split a packed accumulator into the pre-activation values (`w · x + b`) of output
    /// channels `2 * pair` and `2 * pair + 1`, given `x_sum = sum(x)`.
    #[inline(always)]
    fn unpack(&self, pair: usize, acc: u64, x_sum: i64) -> (i64, i64) {
        let correction = 128 * x_sum;
        (
            (acc & 0xffff_ffff) as i64 - correction + self.bias[2 * pair],
            (acc >> 32) as i64 - correction + self.bias[2 * pair + 1],
        )
    }

    /// Packed dot product of `x` with one channel pair, four inputs per step so the
    /// compiler can drop per-element bounds checks.
    #[inline(always)]
    fn pair(&self, pair: usize, x: &[u64], x_sum: i64) -> (i64, i64) {
        let w = &self.packed[pair * self.inputs..(pair + 1) * self.inputs];
        let (mut xs, mut ws) = (x.chunks_exact(4), w.chunks_exact(4));
        let mut acc = 0u64;
        for (a, b) in (&mut xs).zip(&mut ws) {
            acc += a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        }
        for (a, b) in xs.remainder().iter().zip(ws.remainder()) {
            acc += a * b;
        }
        self.unpack(pair, acc, x_sum)
    }
}

pub struct Int8Model {
    shift: u32,
    layers: [Layer; 5],
}

/// Reusable buffers so a batch allocates once.
pub struct Scratch {
    conv_out: Vec<u8>,
    pooled1: Vec<u8>,
    pooled2: Vec<u8>,
    dense_in: Vec<u64>,
    hidden1: Vec<u8>,
    hidden2: Vec<u8>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            conv_out: vec![0; 6 * 28 * 28],
            pooled1: vec![0; 6 * 14 * 14],
            pooled2: vec![0; 400],
            dense_in: vec![0; 400],
            hidden1: vec![0; 120],
            hidden2: vec![0; 84],
        }
    }
}

impl Int8Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("not an int8 model file (bad magic); run `python -m zk_fedeval.quantize`");
        }
        let shift = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let mut at = 12;
        let mut parse = |i: usize| Layer::parse(bytes, &mut at, LAYER_SHAPES[i].0, LAYER_SHAPES[i].1);
        let layers = [parse(0)?.with_rows(), parse(1)?.with_rows(), parse(2)?, parse(3)?, parse(4)?];
        if at != bytes.len() {
            return Err("int8 model file has trailing bytes");
        }
        Ok(Self { shift, layers })
    }

    /// Count predictions matching `labels` over `images` (`n * 3072` raw pixels, CHW order).
    pub fn count_correct(&self, images: &[u8], labels: &[u8]) -> u32 {
        assert_eq!(images.len(), labels.len() * PIXELS_PER_IMAGE, "images/labels length mismatch");
        let mut scratch = Scratch::default();
        images
            .chunks_exact(PIXELS_PER_IMAGE)
            .zip(labels)
            .filter(|(image, label)| self.predict(image, &mut scratch) == **label as usize)
            .count() as u32
    }

    pub fn predict(&self, image: &[u8], s: &mut Scratch) -> usize {
        let [conv1, conv2, fc1, fc2, fc3] = &self.layers;
        self.conv_relu_pool::<3>(conv1, image, 3, 32, &mut s.conv_out, &mut s.pooled1);
        self.conv_relu_pool::<8>(conv2, &s.pooled1, 6, 14, &mut s.conv_out, &mut s.pooled2);
        self.dense_relu(fc1, &s.pooled2, &mut s.dense_in, &mut s.hidden1);
        self.dense_relu(fc2, &s.hidden1, &mut s.dense_in, &mut s.hidden2);

        // Logits only feed argmax, so compare them on a common scale without requantizing.
        let x = widen(&s.hidden2, &mut s.dense_in);
        let x_sum = x.iter().sum::<u64>() as i64;
        let (mut best, mut best_logit) = (0, i64::MIN);
        for pair in 0..fc3.outputs / 2 {
            let (lo, hi) = fc3.pair(pair, x, x_sum);
            for (c, acc) in [(2 * pair, lo), (2 * pair + 1, hi)] {
                let logit = acc * fc3.multiplier[c];
                if logit > best_logit {
                    (best, best_logit) = (c, logit);
                }
            }
        }
        best
    }

    #[inline(always)]
    fn requant(&self, acc: i64, multiplier: i64) -> u8 {
        ((acc * multiplier + (1 << (self.shift - 1))) >> self.shift).clamp(0, 255) as u8
    }

    /// Valid 5x5 conv, requantize (which also applies ReLU), then 2x2 max-pool, for a
    /// layer of `PAIRS` channel pairs.
    ///
    /// Each 5-pixel window row is loaded once and fed to every pair, and with `PAIRS`
    /// known at compile time the accumulators stay in registers.
    fn conv_relu_pool<const PAIRS: usize>(&self, layer: &Layer, x: &[u8], channels: usize, size: usize, out: &mut [u8], pooled: &mut [u8]) {
        assert_eq!(layer.outputs, 2 * PAIRS, "conv kernel instantiated for the wrong layer");
        let conv_size = size - 4;
        let plane = conv_size * conv_size;
        for oy in 0..conv_size {
            for ox in 0..conv_size {
                let mut sum = 0u64;
                let mut acc = [0u64; PAIRS];
                for ic in 0..channels {
                    for ky in 0..5 {
                        let at = ic * size * size + (oy + ky) * size + ox;
                        let r: &[u8; 5] = x[at..at + 5].try_into().unwrap();
                        let r = r.map(u64::from);
                        sum += r[0] + r[1] + r[2] + r[3] + r[4];
                        let w = &layer.by_row[(ic * 5 + ky) * PAIRS * 5..][..PAIRS * 5];
                        for (pair, acc) in acc.iter_mut().enumerate() {
                            let k: &[u64; 5] = w[pair * 5..pair * 5 + 5].try_into().unwrap();
                            *acc += r[0] * k[0] + r[1] * k[1] + r[2] * k[2] + r[3] * k[3] + r[4] * k[4];
                        }
                    }
                }
                for (pair, &acc) in acc.iter().enumerate() {
                    let (lo, hi) = layer.unpack(pair, acc, sum as i64);
                    out[2 * pair * plane + oy * conv_size + ox] = self.requant(lo, layer.multiplier[2 * pair]);
                    out[(2 * pair + 1) * plane + oy * conv_size + ox] = self.requant(hi, layer.multiplier[2 * pair + 1]);
                }
            }
        }
        let half = conv_size / 2;
        for c in 0..layer.outputs {
            let o = &out[c * plane..(c + 1) * plane];
            for py in 0..half {
                for px in 0..half {
                    let at = 2 * py * conv_size + 2 * px;
                    pooled[c * half * half + py * half + px] =
                        o[at].max(o[at + 1]).max(o[at + conv_size]).max(o[at + conv_size + 1]);
                }
            }
        }
    }

    fn dense_relu(&self, layer: &Layer, x: &[u8], widened: &mut [u64], out: &mut [u8]) {
        let x = widen(&x[..layer.inputs], widened);
        let x_sum = x.iter().sum::<u64>() as i64;
        for pair in 0..layer.outputs / 2 {
            let (lo, hi) = layer.pair(pair, x, x_sum);
            out[2 * pair] = self.requant(lo, layer.multiplier[2 * pair]);
            out[2 * pair + 1] = self.requant(hi, layer.multiplier[2 * pair + 1]);
        }
    }
}

fn widen<'a>(x: &[u8], buf: &'a mut [u64]) -> &'a [u64] {
    for (w, v) in buf.iter_mut().zip(x) {
        *w = *v as u64;
    }
    &buf[..x.len()]
}

fn take<'a>(bytes: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8], &'static str> {
    let slice = bytes.get(*at..*at + len).ok_or("int8 model file is truncated")?;
    *at += len;
    Ok(slice)
}

fn read_i32s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so the test needs no extra dependencies.
    fn bytes(seed: &mut u64, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (*seed >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn packed_pair_matches_plain_dot_product() {
        let mut seed = 7;
        let (outputs, inputs) = (4, 400);
        let mut file = bytes(&mut seed, outputs * inputs);
        let bias: Vec<i32> = vec![-5, 0, 17, -1 << 20];
        file.extend(bias.iter().flat_map(|b| b.to_le_bytes()));
        file.extend([1i32; 4].iter().flat_map(|m| m.to_le_bytes()));
        let layer = Layer::parse(&file, &mut 0, outputs, inputs).unwrap();

        for x in [vec![255u8; inputs], vec![0u8; inputs], bytes(&mut seed, inputs)] {
            let wide: Vec<u64> = x.iter().map(|v| *v as u64).collect();
            let sum = wide.iter().sum::<u64>() as i64;
            for pair in 0..outputs / 2 {
                let (lo, hi) = layer.pair(pair, &wide, sum);
                for (o, got) in [(2 * pair, lo), (2 * pair + 1, hi)] {
                    let want: i64 = (0..inputs).map(|i| file[o * inputs + i] as i8 as i64 * x[i] as i64).sum::<i64>()
                        + bias[o] as i64;
                    assert_eq!(got, want, "output {o}");
                }
            }
        }
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(Int8Model::from_bytes(&[0; 64]).is_err());
    }
}
