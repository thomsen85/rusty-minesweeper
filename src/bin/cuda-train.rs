use burn::backend::Autodiff;
use burn_cuda::Cuda;
use minesweeper::ai::ai::train;

pub type CudaBackend = Cuda<f32, i32>;
pub type CudaAutodiffBackend = Autodiff<CudaBackend>;

fn main() {
    let device = dbg!(Default::default());
    train::<CudaAutodiffBackend>(device);
}
