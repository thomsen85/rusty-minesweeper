use nannou::{
    prelude::*,
    rand::{thread_rng, Rng},
};

const ROWS: usize = 20;
const COLS: usize = 20;

const SCREEN_WIDTH: u32 = 1000;
const SCREEN_HEIGHT: u32 = 1000;

const SCREEN_PADDING: f32 = 100.;

const SQUARE_MARGIN: f32 = 2.;

const SQUARE_WIDTH: f32 =
    ((SCREEN_WIDTH as f32 - SCREEN_PADDING * 2.) / COLS as f32) - SQUARE_MARGIN;

const SQUARE_HEIGHT: f32 =
    ((SCREEN_HEIGHT as f32 - SCREEN_PADDING * 2.) / ROWS as f32) - SQUARE_MARGIN;

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
    textures: Vec<wgpu::Texture>,
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
        .view(view) // The function that will be called for presenting graphics to a frame.
        .event(event) // The function that will be called when the window receives events.
        .build()
        .unwrap();

    let assets = app.assets_path().unwrap();
    let img_path = assets.join("bomb.png");
    let bomb_texture = wgpu::Texture::from_path(app, img_path).unwrap();

    Model {
        game_state: GameState::Playing,
        grid: init_grid(25),
        opened: [[true; COLS]; ROWS],
        textures: vec![bomb_texture],
    }
}

fn event(app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        WindowEvent::MousePressed(MouseButton::Left) => {
            if let Some((row, col)) = x_y_to_row_col(app.mouse.x, app.mouse.y) {
                match model.grid[row][col] {
                    Square::Nearby(_) => {
                        model.opened[row][col] = true;
                    }
                    Square::Empty => {}
                    Square::Mine => {
                        model.game_state = GameState::Lost;
                    }
                };
            }
        }
        WindowEvent::MousePressed(MouseButton::Right) => { // Marking
        }
        _ => {}
    }
}

fn row_col_to_x_y(row: usize, col: usize) -> (f32, f32) {
    (
        SQUARE_WIDTH / 2. + (SQUARE_WIDTH + SQUARE_MARGIN) * col as f32 - SCREEN_WIDTH as f32 / 2.
            + SCREEN_PADDING,
        SQUARE_HEIGHT / 2. + (SQUARE_HEIGHT + SQUARE_MARGIN) * row as f32
            - SCREEN_HEIGHT as f32 / 2.
            + SCREEN_PADDING,
    )
}

fn x_y_to_row_col(x: f32, y: f32) -> Option<(usize, usize)> {
    let col =
        ((-x - SCREEN_WIDTH as f32 / 2. + SCREEN_PADDING) / -(SQUARE_WIDTH + SQUARE_MARGIN)) as i32;

    let row = ((-y - SCREEN_HEIGHT as f32 / 2. + SCREEN_PADDING) / -(SQUARE_HEIGHT + SQUARE_MARGIN))
        as i32;

    if (0..COLS as i32).contains(&col) && (0..ROWS as i32).contains(&row) {
        return Some((row as usize, col as usize));
    }

    None
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(GRAY);

    for row in 0..ROWS {
        for col in 0..COLS {
            let (x, y) = row_col_to_x_y(row, col);
            if !model.opened[row][col] {
                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(WHITE);
            } else {
                let square_state = model.grid[row][col];
                draw.rect()
                    .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                    .x_y(x, y)
                    .color(Rgb::new(0.3, 0.3, 0.3));

                match square_state {
                    Square::Empty => {
                        // draw.rect()
                        //     .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                        //     .x_y(x, y)
                        //     .color(Rgb::new(0.5, 0.5, 0.5));
                    }
                    Square::Nearby(v) => {
                        draw.text(&v.to_string())
                            .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                            .x_y(x, y)
                            .font_size(24)
                            .color(WHITE);
                    }
                    Square::Mine => {
                        draw.texture(model.textures.first().unwrap())
                            .w_h(SQUARE_WIDTH, SQUARE_HEIGHT)
                            .x_y(x, y);
                    }
                }
            }
        }
    }
    draw.to_frame(app, &frame).unwrap();
}
