mod ai;
mod constants;
mod game;
mod utils;

use std::collections::HashMap;

use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::CompactRecorder,
    tensor::{ElementConversion, Tensor},
};
use burn_cuda::Cuda;
use constants::*;
use game::{Minesweeper, Square};
use nannou::prelude::*;

use crate::ai::{batcher::MinesweeperBatcher, model::ModelConfig};

fn main() {
    nannou::app(model).run();
}

#[derive(Clone, Copy, Debug)]
enum GameState {
    Playing,
    Lost,
    Won,
}

type MyBackend = Cuda<f32, i32>;
struct Model {
    game_state: GameState,
    minesweeper: Minesweeper,
    textures: HashMap<&'static str, wgpu::Texture>,
    first_click: bool,
    ai_model: ai::model::Model<MyBackend>,
    ai_prediction: Option<Vec<f32>>,
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

    let device = Default::default();
    let ai_model =
        ModelConfig::new(constants::ROWS, constants::COLS, 512).init::<MyBackend>(&device);

    let ai_model = ai_model
        .load_file(
            "artifacts/checkpoint/model-100.mpk",
            &CompactRecorder::new(),
            &device,
        )
        .expect("Coulnd load file");

    Model {
        game_state: GameState::Playing,
        minesweeper: Minesweeper::new_with_mines(MINES),
        textures: HashMap::from([("bomb", bomb_texture), ("flag", flag_texture)]),
        first_click: true,
        ai_model,
        ai_prediction: None,
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
                dbg!(model.minesweeper.get_category_vec());
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
        WindowEvent::KeyPressed(Key::M) => {
            // let device = Default::default();
            // let opened_float_map: [f32; constants::ROWS * constants::COLS] = model
            //     .minesweeper
            //     .opened
            //     .iter()
            //     .flatten()
            //     .map(|&a| if a { 0. } else { 1. })
            //     .collect::<Vec<_>>()
            //     .try_into()
            //     .unwrap();
            //
            // let possibility_mask = Tensor::from_data(opened_float_map, &device);
            let board_state = MinesweeperBatcher::default()
                .batch(vec![model.minesweeper.clone()])
                .boards;

            let forward = model.ai_model.forward(board_state);
            let prediction = forward.reshape([constants::ROWS, constants::COLS]);
            model.ai_prediction = Some(
                prediction
                    .to_data()
                    .iter()
                    .map(|val: f32| dbg!(val))
                    .collect(),
            );
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
                let color = if let Some(prediction) = &model.ai_prediction {
                    let v = prediction[row * constants::COLS + col].clamp(0.0, 1.0);
                    Rgb::new(1., 1. - v, 1. - v)
                } else {
                    Rgb::new(1., 1., 1.)
                };
                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(color);

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
