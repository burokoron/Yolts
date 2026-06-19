use crate::{Color, Square};

pub(crate) fn current_view(square: Square, side_to_move: Color) -> Square {
    match side_to_move {
        Color::Black => square,
        Color::White => square.flip(),
    }
}

pub(crate) fn plane_square_index(square: Square) -> usize {
    // Square::array_index()はfile-majorだが、networkとmove labelはrank-majorである。
    usize::from(square.rank() - 1) * 9 + usize::from(square.file() - 1)
}
