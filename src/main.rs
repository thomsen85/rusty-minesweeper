mod constants;
mod utils;

use std::collections::HashMap;

use constants::*;
use nannou::{
    prelude::*,
    rand::{thread_rng, Rng},
};

fn main() {
    nannou::app(model).run();
}

#[derive(Clone, Copy, Debug)]
enum Square {
    Empty,
    Nearby(u32),
    Mine,
}

#[derive(Clone, Copy, Debug)]
enum GameState {
    Playing,
    Lost,
    Won,
}

struct Model {
    game_state: GameState,
    grid: [[Square; COLS]; ROWS],
    opened: [[bool; COLS]; ROWS],
    marked: [[bool; COLS]; ROWS],
    textures: HashMap<&'static str, wgpu::Texture>,
    first_click: bool,
}

fn init_grid(mines: usize) -> [[Square; COLS]; ROWS] {
    assert!(mines < COLS * ROWS);

    let mut mines_left = mines;
    let mut rng = thread_rng();
    let mut grid = [[Square::Empty; COLS]; ROWS];

    loop {
        if mines_left == 0 {
            break;
        }
        let new_pos_row = rng.gen_range(0..ROWS);
        let new_pos_col = rng.gen_range(0..COLS);

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
    grid
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(SCREEN_WIDTH, SCREEN_HEIGHT)
        .resizable(false)
        .title("Bombsearcher")
        .view(view)
        .event(event)
        .build()
        .unwrap();

    let assets = app.assets_path().unwrap();
    let bomb_path = assets.join("bomb.png");
    let flag_path = assets.join("flag.png");
    let bomb_texture = wgpu::Texture::from_path(app, bomb_path).unwrap();
    let flag_texture = wgpu::Texture::from_path(app, flag_path).unwrap();

    Model {
        game_state: GameState::Playing,
        grid: init_grid(MINES),
        opened: [[false; COLS]; ROWS],
        marked: [[false; COLS]; ROWS],
        textures: HashMap::from([("bomb", bomb_texture), ("flag", flag_texture)]),
        first_click: true,
    }
}

fn event(app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        WindowEvent::MousePressed(MouseButton::Left) => {
            if let Some((row, col)) = utils::x_y_to_row_col(app.mouse.x, app.mouse.y) {
                // Player will always hit Empty on first click :D
                if model.marked[row][col] {
                    return;
                }

                while model.first_click && !matches!(model.grid[row][col], Square::Empty) {
                    model.grid = init_grid(MINES);
                }
                model.first_click = false;

                match model.grid[row][col] {
                    Square::Nearby(_) => {
                        model.opened[row][col] = true;
                    }
                    Square::Empty => {
                        let mut stack = vec![(row, col)];
                        let neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                        while let Some((curr_row, curr_col)) = stack.pop() {
                            if model.opened[curr_row][curr_col]
                                || matches!(model.grid[curr_row][curr_col], Square::Mine)
                            {
                                continue;
                            }

                            model.opened[curr_row][curr_col] = true;

                            if matches!(model.grid[curr_row][curr_col], Square::Nearby(_)) {
                                continue;
                            }

                            for (n_row, n_col) in neighbours {
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
                        model.opened[row][col] = true;
                        model.game_state = GameState::Lost;
                    }
                };
            }
        }
        WindowEvent::MousePressed(MouseButton::Right) => {
            if let Some((row, col)) = utils::x_y_to_row_col(app.mouse.x, app.mouse.y) {
                if model.opened[row][col] {
                    return;
                }

                model.marked[row][col] = !model.marked[row][col];
            }
        }
        _ => {}
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(GRAY);

    for row in 0..ROWS {
        for col in 0..COLS {
            let (x, y) = utils::row_col_to_x_y(row, col);

            if !model.opened[row][col] {
                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(WHITE);

                if model.marked[row][col] {
                    draw.texture(model.textures.get("flag").unwrap())
                        .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                        .x_y(x, y);
                }
            } else {
                let square_state = model.grid[row][col];
                let background_color = match square_state {
                    Square::Empty | Square::Nearby(_) => Rgb::new(0.3, 0.3, 0.3),
                    Square::Mine => Rgb::new(0.8, 0.3, 0.3),
                };

                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(background_color);

                match square_state {
                    Square::Empty => {}
                    Square::Nearby(v) => {
                        draw.text(&v.to_string())
                            .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                            .x_y(x, y)
                            .font_size(24)
                            .color(WHITE);
                    }
                    Square::Mine => {
                        draw.texture(model.textures.get("bomb").unwrap())
                            .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                            .x_y(x, y);
                    }
                }
            }
        }
    }
    draw.to_frame(app, &frame).unwrap();
}
