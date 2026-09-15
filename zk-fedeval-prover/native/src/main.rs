mod config;
mod dataset;
mod evaluator;
mod model;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

use config::ProverConfig;
use dataset::EvalDataset;
use evaluator::Evaluator;
use model::CnnModel;

type Backend = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let config = ProverConfig::from_env();
    let device = NdArrayDevice::default();

    let dataset = EvalDataset::load(&config.eval_data_path)?;
    println!(
        "loaded {} evaluation samples from {}",
        dataset.len(),
        config.eval_data_path.display()
    );

    let model = CnnModel::<Backend>::load(&device);
    let evaluator = Evaluator::new(model, device);

    let report = evaluator.run(&dataset);
    println!("{report}");

    Ok(())
}
