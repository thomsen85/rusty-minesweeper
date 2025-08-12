use burn_cuda::CudaDevice;
use minesweeper::ai::{train, MyAutodiffBackend};

fn main() {
    // let device = dbg!(CudaDevice::default());
    let device = dbg!(Default::default());
    train::<MyAutodiffBackend>(device);
}
