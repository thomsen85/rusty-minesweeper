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
            // mines.push(
            //     Tensor::<_, 2>::from_data(
            //         item.grid.map(|row| {
            //             row.map(|val| {
            //                 if matches!(val, crate::game::Square::Mine) {
            //                     1.0
            //                 } else {
            //                     0.0
            //                 }
            //             })
            //         }),
            //         device,
            //     )
            //     .unsqueeze::<3>(),
            // );

            let mines_on_board: [[f32; COLS]; ROWS] = item
                .grid
                .iter()
                .enumerate()
                .map(|(row_i, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(col_i, val)| {
                            use crate::game::Square::*;
                            let has_opened_square_around = (-1..=1).any(|row_delta| {
                                (-1..=1).any(|col_delta| {
                                    let new_row = row_i as i32 + row_delta;
                                    let new_col = col_i as i32 + col_delta;

                                    if !(0..COLS as i32).contains(&new_col)
                                        || !(0..ROWS as i32).contains(&new_row)
                                    {
                                        return false;
                                    }

                                    board_opened
                                        .get(new_row as usize)
                                        .and_then(|opened_row| opened_row.get(new_col as usize))
                                        .is_some_and(|&opened_val| opened_val == 1.0)
                                })
                            });

                            if matches!(val, Mine) && has_opened_square_around {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<f32>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            mines.push(Tensor::<_, 2>::from_data(mines_on_board, device).unsqueeze::<3>());

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
