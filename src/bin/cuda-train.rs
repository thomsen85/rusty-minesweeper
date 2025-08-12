use burn::{backend::Autodiff, optim::AdamConfig};
use burn_cuda::Cuda;
use minesweeper::{
    ai::{
        model::ModelConfig,
        train::{train, TrainingConfig},
    },
    constants,
};

pub type CudaBackend = Cuda<f32, i32>;
pub type CudaAutodiffBackend = Autodiff<CudaBackend>;

fn main() {
    let device = dbg!(Default::default());
    let artifact_dir = std::env::var("ARTIFACT_DIR").unwrap_or_else(|_| "artifacts".to_string());
    train::<CudaAutodiffBackend>(
        &artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(constants::COLS, constants::ROWS, 512),
            AdamConfig::new(),
        ),
        device,
    );
}
