use shogi_core::{Color, Hand, PieceKind, Square};

use crate::Position;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeclarationRuleConfig {
    pub point_threshold_black: u16,
    pub point_threshold_white: u16,
    pub rook_bishop_point: u16,
    pub other_piece_point: u16,
    pub min_pieces_in_enemy_camp_excluding_king: u8,
}

impl DeclarationRuleConfig {
    #[must_use]
    pub const fn csa27() -> Self {
        Self {
            point_threshold_black: 28,
            point_threshold_white: 27,
            rook_bishop_point: 5,
            other_piece_point: 1,
            min_pieces_in_enemy_camp_excluding_king: 10,
        }
    }

    #[must_use]
    pub const fn point_threshold(self, color: Color) -> u16 {
        match color {
            Color::Black => self.point_threshold_black,
            Color::White => self.point_threshold_white,
        }
    }
}

impl Default for DeclarationRuleConfig {
    fn default() -> Self {
        Self::csa27()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeclarationWinInfo {
    pub side: Color,
    pub points: u16,
    pub pieces_in_enemy_camp: u8,
}

impl Position {
    #[must_use]
    pub fn declaration_win(&self, config: DeclarationRuleConfig) -> Option<DeclarationWinInfo> {
        let position: &yasai::Position = self.as_ref();
        let side = position.side_to_move();
        if position.in_check() {
            return None;
        }
        let king_in_enemy_camp = Square::all().any(|square| {
            position.piece_at(square).is_some_and(|piece| {
                piece.color() == side
                    && piece.piece_kind() == PieceKind::King
                    && square.relative_rank(side) <= 3
            })
        });
        if !king_in_enemy_camp {
            return None;
        }

        let mut points = 0_u16;
        let mut pieces_in_enemy_camp = 0_u8;
        for square in Square::all() {
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            if piece.color() != side
                || piece.piece_kind() == PieceKind::King
                || square.relative_rank(side) > 3
            {
                continue;
            }
            pieces_in_enemy_camp += 1;
            points += piece_point(piece.piece_kind(), config);
        }
        if pieces_in_enemy_camp < config.min_pieces_in_enemy_camp_excluding_king {
            return None;
        }

        let hand = position.hand(side);
        for kind in Hand::all_hand_pieces() {
            let count = hand.count(kind).unwrap_or_default();
            points += u16::from(count) * piece_point(kind, config);
        }
        (points >= config.point_threshold(side)).then_some(DeclarationWinInfo {
            side,
            points,
            pieces_in_enemy_camp,
        })
    }
}

const fn piece_point(kind: PieceKind, config: DeclarationRuleConfig) -> u16 {
    match kind {
        PieceKind::Bishop | PieceKind::Rook | PieceKind::ProBishop | PieceKind::ProRook => {
            config.rook_bishop_point
        }
        PieceKind::Pawn
        | PieceKind::Lance
        | PieceKind::Knight
        | PieceKind::Silver
        | PieceKind::Gold
        | PieceKind::ProPawn
        | PieceKind::ProLance
        | PieceKind::ProKnight
        | PieceKind::ProSilver => config.other_piece_point,
        PieceKind::King => 0,
    }
}
