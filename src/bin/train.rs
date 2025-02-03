use minesweeper::ai::{train, MyAutodiffBackend};

fn main() {
    let device = dbg!(Default::default());
    train::<MyAutodiffBackend>(device);
}
