mod constants;
mod minesweeper;
mod utils;

use std::collections::HashMap;

use constants::*;
use minesweeper::{Minesweeper, Square};
use nannou::prelude::*;

fn main() {
    nannou::app(model).run();
}

#[derive(Clone, Copy, Debug)]
enum GameState {
    Playing,
    Lost,
    Won,
}

struct Model {
    game_state: GameState,
    minesweeper: Minesweeper,
    textures: HashMap<&'static str, wgpu::Texture>,
    first_click: bool,
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
        minesweeper: Minesweeper::new_with_mines(MINES),
        textures: HashMap::from([("bomb", bomb_texture), ("flag", flag_texture)]),
        first_click: true,
    }
}

fn event(app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        WindowEvent::MousePressed(MouseButton::Left) => {
            if let Some((row, col)) = utils::x_y_to_row_col(app.mouse.x, app.mouse.y) {
                // Player will always hit Empty on first click :D
                while model.first_click
                    && !matches!(model.minesweeper.square_state(row, col), Square::Empty)
                {
                    model.minesweeper = Minesweeper::new_with_mines(MINES);
                }
                model.first_click = false;

                if model.minesweeper.is_square_marked(row, col) {
                    return;
                }
                model.minesweeper.click(row, col);

                if model.minesweeper.is_board_completed() {
                    println!("Yey");
                }
            }
        }
        WindowEvent::MousePressed(MouseButton::Right) => {
            if let Some((row, col)) = utils::x_y_to_row_col(app.mouse.x, app.mouse.y) {
                if model.minesweeper.is_square_open(row, col) {
                    return;
                }

                model.minesweeper.mark(row, col);
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

            if !model.minesweeper.is_square_open(row, col) {
                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(WHITE);

                if model.minesweeper.is_square_marked(row, col) {
                    draw.texture(model.textures.get("flag").unwrap())
                        .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                        .x_y(x, y);
                }
            } else {
                let square_state = model.minesweeper.square_state(row, col);
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
