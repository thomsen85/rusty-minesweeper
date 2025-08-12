use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{
    constants::{COLS, ROWS},
    game::Minesweeper,
};

#[derive(Clone, Default)]
pub struct MinesweeperBatcher {}

#[derive(Clone, Debug)]
pub struct MinesweeperBatch<B: Backend> {
    pub boards: Tensor<B, 4>,
    pub mines: Tensor<B, 3>,
}

impl<B: Backend> Batcher<Minesweeper, MinesweeperBatch<B>> for MinesweeperBatcher {
    fn batch(&self, items: Vec<Minesweeper>) -> MinesweeperBatch<B> {
        let device = &B::Device::default();
        let mut input_boards = Vec::new();
        let mut mines = Vec::new();
        let items_len = items.len();

        for item in items {
            let board_opened = item
                .opened
                .map(|row| row.map(|val| if val { 1. } else { 0. }));

            let board_vals_masked = item
                .grid
                .iter()
                .zip(board_opened)
                .map(|(grid_row, opened_row)| {
                    grid_row
                        .iter()
                        .zip(opened_row)
                        .map(|(grid_val, opened_val)| {
                            use crate::game::Square::*;
                            let grid_val_f = match grid_val {
                                Empty => 0.,
                                Nearby(v) => *v as f32,
                                Mine => -1.,
                            };
                            // assert!(!(opened_val == 1. && grid_val_f == -1.));
                            grid_val_f * opened_val
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>();

            input_boards.push(
                Tensor::<_, 3>::from_data(
                    [board_opened, board_vals_masked.try_into().unwrap()],
                    device,
                )
                .unsqueeze::<4>(),
            );
            mines.push(
                Tensor::<_, 2>::from_data(
                    item.grid.map(|row| {
                        row.map(|val| {
                            if matches!(val, crate::game::Square::Mine) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                    }),
                    device,
                )
                .unsqueeze::<3>(),
            );

            // Could mask out all mines that can not be possibly known
            // let mut mines = Vec::new();
            // for row_i in 0..ROWS {
            //     let mut mine_row = Vec::new();
            //     for col_i in 0..COLS {
            //         let val = item.grid[row_i][col_i];
            //         use crate::game::Square::*;
            //         if !matches!(val, Mine) {
            //             mine_row.push(0);
            //             continue;
            //         }
            //
            //         for row_delta in -1..=1 {
            //             for col_delta in -1..=1 {
            //                 if row_delta == 0 && col_delta == 0 {
            //                     continue;
            //                 }
            //
            //                 if !(0..ROWS as i32).contains
            //             }
            //         }
            //     }
            // }
        }

        // let boards = Tensor::<_, 4>::cat(input_boards, 0).to_device(device);
        // let mines = Tensor::<_, 4>::cat(mines, 0).to_device(device);
        let boards = Tensor::<_, 4>::cat(input_boards, 0);
        let mines = Tensor::<_, 3>::cat(mines, 0);

        assert_eq!(boards.dims(), [items_len, 2, ROWS, COLS]);
        assert_eq!(mines.dims(), [items_len, ROWS, COLS]);

        MinesweeperBatch { boards, mines }
    }
}
