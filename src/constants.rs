pub const ROWS: usize = 20;
pub const COLS: usize = 20;

pub const MINES: usize = 10;

pub const SCREEN_WIDTH: u32 = 1000;
pub const SCREEN_HEIGHT: u32 = 1000;

pub const SCREEN_PADDING: f32 = 100.;

pub const SQUARE_MARGIN: f32 = 2.;

pub const SQUARE_WIDTH: f32 =
    ((SCREEN_WIDTH as f32 - SCREEN_PADDING * 2.) / COLS as f32) - SQUARE_MARGIN;

pub const SQUARE_HEIGHT: f32 =
    ((SCREEN_HEIGHT as f32 - SCREEN_PADDING * 2.) / ROWS as f32) - SQUARE_MARGIN;
