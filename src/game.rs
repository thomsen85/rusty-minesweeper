use crate::constants::*;
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub enum Square {
    Empty,
    Nearby(u32),
    Mine,
}

pub struct Minesweeper {
    grid: [[Square; COLS]; ROWS],
    pub opened: [[bool; COLS]; ROWS],
    marked: [[bool; COLS]; ROWS],
}

impl Minesweeper {
    pub fn new_with_mines(mines: usize) -> Self {
        assert!(mines < COLS * ROWS);

        let mut mines_left = mines;
        let mut rng = rand::rng();
        let mut grid = [[Square::Empty; COLS]; ROWS];

        loop {
            if mines_left == 0 {
                break;
            }
            let new_pos_row = rng.random_range(0..ROWS);
            let new_pos_col = rng.random_range(0..COLS);

            let pos = &mut grid[new_pos_row][new_pos_col];

            if matches!(pos, Square::Mine) {
                continue;
            }

            *pos = Square::Mine;
            mines_left -= 1;

            for row_delta in -1..=1 {
                for col_delta in -1..=1 {
                    if row_delta == 0 && col_delta == 0
                        || row_delta > new_pos_row as i32
                        || col_delta > new_pos_col as i32
                    {
                        continue;
                    }

                    if let Some(Some(val)) = grid
                        .get_mut((new_pos_row as i32 - row_delta) as usize)
                        .map(|row| row.get_mut((new_pos_col as i32 - col_delta) as usize))
                    {
                        *val = match val {
                            Square::Empty => Square::Nearby(1),
                            Square::Nearby(v) => Square::Nearby(*v + 1),
                            Square::Mine => Square::Mine,
                        }
                    }
                }
            }
        }
        Minesweeper {
            grid,
            opened: [[false; COLS]; ROWS],
            marked: [[false; COLS]; ROWS],
        }
    }

    /// Will return None if an already opend square is clicked
    pub fn click(&mut self, row: usize, col: usize) -> Square {
        assert!(!self.opened[row][col]);

        match self.grid[row][col] {
            Square::Nearby(_) => {
                self.opened[row][col] = true;
            }
            Square::Empty => {
                let mut stack = vec![(row, col)];
                let neighbours = (-1..=1)
                    .flat_map(|a| {
                        (-1..=1)
                            .filter_map(move |b| if a == 0 && b == 0 { None } else { Some((a, b)) })
                    })
                    .collect::<Vec<_>>();

                while let Some((curr_row, curr_col)) = stack.pop() {
                    if self.opened[curr_row][curr_col]
                        || matches!(self.grid[curr_row][curr_col], Square::Mine)
                    {
                        continue;
                    }

                    self.opened[curr_row][curr_col] = true;

                    if matches!(self.grid[curr_row][curr_col], Square::Nearby(_)) {
                        continue;
                    }

                    for (n_row, n_col) in &neighbours {
                        let next_row = curr_row as i32 + n_row;
                        let next_col = curr_col as i32 + n_col;
                        if next_row < 0
                            || next_row >= ROWS as i32
                            || next_col < 0
                            || next_col >= COLS as i32
                        {
                            continue;
                        }

                        stack.push((next_row as usize, next_col as usize));
                    }
                }
            }
            Square::Mine => {
                self.opened[row][col] = true;
            }
        }
        self.grid[row][col]
    }

    pub fn square_state(&self, row: usize, col: usize) -> Square {
        self.grid[row][col]
    }

    pub fn is_square_open(&self, row: usize, col: usize) -> bool {
        self.opened[row][col]
    }

    pub fn is_square_marked(&self, row: usize, col: usize) -> bool {
        self.marked[row][col]
    }

    pub fn mark(&mut self, row: usize, col: usize) {
        if self.opened[row][col] {
            return;
        }

        self.marked[row][col] = !self.marked[row][col];
    }

    /// Checks if all squares except the bombs are opened
    pub fn is_board_completed(&self) -> bool {
        // Crating a bitmap for each. if (res == 0) than won
        // BOMB:   0 1 0 1
        // OPENED: 0 0 1 1
        // RES:    1 0 0 1
        // This is a XNOR gate or just equality of booleans;

        let bomb_bit_map = self
            .grid
            .iter()
            .flatten()
            .map(|state| matches!(state, Square::Mine));

        let opened_bit_map = self.opened.iter().flatten();

        bomb_bit_map
            .zip(opened_bit_map)
            .map(|(a, b)| a == *b)
            .all(|a| !a)
    }

    /// This is meant for a structrure to be able to train on
    pub fn get_category_vec(&self) -> StateArray {
        // First is bitmap of opened
        // Second is the empty squares
        // Last is 1-8 state for squares where 0 is empty and 1-8 is nearby state
        // Size is 10 * ROW * COL

        self.opened
            .iter()
            .flatten()
            .copied()
            .chain(
                self.grid
                    .iter()
                    .flatten()
                    .map(|state| matches!(state, Square::Empty)),
            )
            .chain((1..=8).flat_map(|i| {
                self.grid
                    .iter()
                    .flatten()
                    .zip(self.opened.iter().flatten())
                    .map(move |(state, &open)| {
                        open && matches!(state, Square::Nearby(x) if *x == i)
                    })
            }))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|v: Vec<bool>| {
                panic!(
                    "Expected a Vec of length {} but it was {}",
                    10 * ROWS * COLS,
                    v.len()
                )
            })
    }
}

pub type StateArray = [bool; 10 * ROWS * COLS];
