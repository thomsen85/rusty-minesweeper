use crate::constants::*;

pub(crate) fn row_col_to_x_y(row: usize, col: usize) -> (f32, f32) {
    (
        SQUARE_WIDTH / 2. + (SQUARE_WIDTH + SQUARE_MARGIN) * col as f32 - SCREEN_WIDTH as f32 / 2.
            + SCREEN_PADDING,
        SQUARE_HEIGHT / 2. + (SQUARE_HEIGHT + SQUARE_MARGIN) * row as f32
            - SCREEN_HEIGHT as f32 / 2.
            + SCREEN_PADDING,
    )
}

pub(crate) fn x_y_to_row_col(x: f32, y: f32) -> Option<(usize, usize)> {
    let col =
        ((-x - SCREEN_WIDTH as f32 / 2. + SCREEN_PADDING) / -(SQUARE_WIDTH + SQUARE_MARGIN)) as i32;

    let row = ((-y - SCREEN_HEIGHT as f32 / 2. + SCREEN_PADDING) / -(SQUARE_HEIGHT + SQUARE_MARGIN))
        as i32;

    if (0..COLS as i32).contains(&col) && (0..ROWS as i32).contains(&row) {
        return Some((row as usize, col as usize));
    }

    None
}
