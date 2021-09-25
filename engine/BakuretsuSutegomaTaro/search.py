"""
探索アルゴリズム
"""

import dataclasses
import random
import typing

import cshogi


@dataclasses.dataclass
class NegaAlpha:
    """
    ネガアルファ探索
    """

    def __post_init__(self) -> None:
        self.num_searched = 0  # 探索局面数
        self.best_move_pv = "resign"  # 最善手PV
        self.max_board_number = 0

    def search(
        self, board: typing.Any, depth: float, alpha: int, beta: int
    ) -> typing.Any:
        # 選択的探索深さ計測用
        if board.move_number > self.max_board_number:
            self.max_board_number = board.move_number

        # 終局なら
        if board.is_game_over():
            return -30000 + board.turn

        # 探索深さ制限なら
        if depth <= 0:
            return random.randint(-500, 500)

        # 王手ならちょっと延長
        if board.is_check():
            depth += 0.5

        # 合法手展開
        best_value = -1000000
        for move in board.legal_moves:
            move = cshogi.move_to_usi(move)
            board.push_usi(move)
            self.num_searched += 1
            value = -self.search(board=board, depth=depth - 1, alpha=-beta, beta=-alpha)
            board.pop()

            if best_value < value:
                best_value = value
                if depth == 3:
                    self.best_move_pv = move
                    info_text = f"info depth {int(depth)} "
                    info_text += (
                        f"seldepth {int(self.max_board_number - board.move_number)} "
                    )
                    info_text += f"nodes {self.num_searched} score cp {value} "
                    info_text += f"pv {self.best_move_pv}"
                    print(info_text, flush=True)

        return best_value
