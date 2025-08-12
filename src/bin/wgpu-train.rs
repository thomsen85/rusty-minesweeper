use minesweeper::ai::ai::{train, MyAutodiffBackend};

pub type WgpuBackend = Wgpu<f32, i32>;
pub type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

fn main() {
    let device = dbg!(Default::default());
    train::<WgpuAutodiffBackend>(device);
}
