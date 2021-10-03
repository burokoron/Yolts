"""
探索アルゴリズム
"""

import dataclasses
import time
import typing

import cshogi


@dataclasses.dataclass
class NegaAlpha:
    """
    ネガアルファ探索
    """

    max_depth: float  # 探索深さ
    max_time: float  # 最大思考時間(マージンなし)
    pieces_in_hand_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(持ち駒)
    pieces_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(盤面)

    def __post_init__(self) -> None:
        self.num_searched = 0  # 探索局面数
        self.best_move_pv = "resign"  # 最善手PV
        self.max_board_number = 0  # 最大手数
        self.start_time = time.perf_counter() * 1000  # 思考開始時刻

    def search(
        self, board: typing.Any, depth: float, alpha: int, beta: int
    ) -> typing.Any:
        # 選択的探索深さ計測用
        if board.move_number > self.max_board_number:
            self.max_board_number = board.move_number

        # 終局なら
        if board.is_game_over():
            return -30000 + board.move_number

        # 同一局面なら
        if board.is_draw() and self.best_move_pv != "resign":
            return 0

        # 探索深さ制限なら
        if depth <= 0:
            value = 0
            bp, bw = board.pieces_in_hand
            ap = bp + bw
            for i, piece in enumerate(ap):
                value += self.pieces_in_hand_dict[i][piece]
            pieces = board.pieces
            for i, piece in enumerate(pieces):
                value += self.pieces_dict[i][piece]
            if board.turn == 0:
                return value
            else:
                return -value

        # 時間制限なら
        if time.perf_counter() * 1000 - self.start_time >= self.max_time:
            return alpha

        # 王手ならちょっと延長
        if board.is_check():
            depth += 0.0

        # 合法手展開
        best_value = alpha
        for move in board.legal_moves:
            move = cshogi.move_to_usi(move)
            board.push_usi(move)
            self.num_searched += 1
            value = -self.search(
                board=board, depth=depth - 1, alpha=-beta, beta=-best_value
            )
            board.pop()

            if best_value < value:
                best_value = value
                if depth >= self.max_depth:
                    elapsed_time = int(time.perf_counter() * 1000 - self.start_time)
                    if elapsed_time < self.max_time:
                        self.best_move_pv = move

                        info_text = f"info depth {int(self.max_depth)} "
                        info_text += "seldepth "
                        info_text += (
                            f"{int(self.max_board_number - board.move_number)} "
                        )
                        info_text += f"time {elapsed_time} "
                        info_text += f"nodes {self.num_searched} score cp {value} "
                        info_text += f"pv {self.best_move_pv} "
                        if elapsed_time != 0:
                            info_text += (
                                f"nps {int(self.num_searched / (elapsed_time / 1000))}"
                            )
                        print(info_text, flush=True)

            if best_value >= beta:
                break

        return best_value
