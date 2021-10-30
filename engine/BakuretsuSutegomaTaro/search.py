"""
探索アルゴリズム
"""

import dataclasses
import operator
import time
import typing

import cshogi


@dataclasses.dataclass
class NegaAlpha:
    """
    ネガアルファ探索

    Args:
        max_depth: float
            探索深さ
        start_time: float
            思考開始時刻
        max_time: float
            最大思考時間(マージンなし)
        pieces_in_hand_dict: typing.Dict[int, typing.Dict[int, int]]
            評価パラメータ(持ち駒)
        pieces_dict: typing.Dict[int, typing.Dict[int, int]]
            評価パラメータ(盤面)
        move_ordering_dict: typing.Dict[str, typing.Dict[str, int]]
            ムーブオーダリング辞書
        any_to_ordering_dict: typing.Dict[int, typing.Dict[str, float]]
            ムーブオーダリング辞書(着手位置)
        num_searched: int = 0
            探索局面数
        best_move_pv: str = "resign"
            最善手PV
        max_board_number: int = 0
            最大手数
    """

    max_depth: float  # 探索深さ
    start_time: float  # 思考開始時刻
    max_time: float  # 最大思考時間(マージンなし)
    pieces_in_hand_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(持ち駒)
    pieces_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(盤面)
    move_ordering_dict: typing.Dict[str, typing.Dict[str, int]]  # ムーブオーダリング辞書(指し手)
    any_to_ordering_dict: typing.Dict[int, typing.Dict[str, float]]  # ムーブオーダリング辞書(着手位置)
    num_searched: int = 0  # 探索局面数
    best_move_pv: str = "resign"  # 最善手PV
    max_board_number: int = 0  # 最大手数

    def evaluate(self, board: typing.Any) -> int:
        """
        評価値の計算

        Args:
            board: typing.Any
                cshogiの局面
        """

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

    def search(
        self, board: typing.Any, depth: float, alpha: int, beta: int
    ) -> typing.Any:
        """
        ネガアルファ探索

        Args:
            board: typing.Any
                cshogiの局面
            depth: float
                残り探索深さ
            alpha: int
                アルファ値
            beta: int
            　　ベータ値

        Returns:
            best_value: typing.Any
                ベスト評価値
        """

        # 選択的探索深さ計測用
        if board.move_number > self.max_board_number:
            self.max_board_number = board.move_number

        # 終局なら
        if board.is_game_over():
            return -30000 + board.move_number

        # 同一局面なら
        if board.is_draw() == 1 and self.max_depth > depth:
            return 0

        # 探索深さ制限なら
        if depth <= 0:
            return self.evaluate(board=board)

        # 時間制限なら
        if time.perf_counter() * 1000 - self.start_time >= self.max_time:
            return alpha

        # 王手ならちょっと延長
        if board.is_check() and self.max_depth > depth:
            depth += 0.0

        # 合法手展開
        best_value = alpha
        sfen = board.sfen()
        move_list = []
        # 現局面における合法手の評価値が高かった順に並べる
        if sfen in self.move_ordering_dict:
            sorted_move_list = sorted(
                self.move_ordering_dict[sfen].items(), key=lambda x: x[1], reverse=True
            )
            move_list = [move[0] for move in sorted_move_list]
        else:
            self.move_ordering_dict[sfen] = {}
        # 同一深度での着手位置で評価値の高かった順に並べる
        all_move_list = [cshogi.move_to_usi(move) for move in board.legal_moves]
        all_move_list = [
            [move, self.any_to_ordering_dict[board.move_number][move[2:4]]]
            for move in all_move_list
        ]
        all_move_list = sorted(all_move_list, key=operator.itemgetter(1), reverse=True)
        all_move_list = [move[0] for move in all_move_list]
        move_list += [move for move in all_move_list if move not in move_list]

        for move in move_list:
            board.push_usi(move)
            self.num_searched += 1
            value = -self.search(
                board=board, depth=depth - 1, alpha=-beta, beta=-best_value
            )
            board.pop()

            # ムーブオーダリング登録
            self.any_to_ordering_dict[board.move_number][move[2:4]] *= 0.9
            self.any_to_ordering_dict[board.move_number][move[2:4]] += value * 0.1

            # 評価値更新なら
            if best_value < value:
                # ムーブオーダリング登録
                self.move_ordering_dict[sfen][move] = value
                best_value = value

                # 探索開始局面ならログ更新&最善手更新
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

            # ベータカット
            if best_value >= beta:
                break

        return best_value


@dataclasses.dataclass
class IterativeDeepeningSearch:
    """
    反復深化探索

    Args:
        max_depth: float
            探索深さ
        start_time: float
            思考開始時刻[sec]
        max_time: float
            最大思考時間[sec] (マージンなし)
        pieces_in_hand_dict: typing.Dict[int, typing.Dict[int, int]]
            評価パラメータ(持ち駒)
        pieces_dict: typing.Dict[int, typing.Dict[int, int]]
            評価パラメータ(盤面)
    """

    max_depth: float  # 探索深さ
    start_time: float  # 思考開始時刻
    max_time: float  # 最大思考時間(マージンなし)
    pieces_in_hand_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(持ち駒)
    pieces_dict: typing.Dict[int, typing.Dict[int, int]]  # 評価パラメータ(盤面)

    def search(self, board: typing.Any) -> str:
        """
        反復深化探索

        Args:
            board: typing.Any
                cshogiの局面

        Returns:
            best_move: str
                最善手
        """

        # 初期化
        info_text = ""
        na = NegaAlpha(
            max_depth=self.max_depth + 1,
            start_time=self.start_time,
            max_time=self.max_time,
            pieces_in_hand_dict=self.pieces_in_hand_dict,
            pieces_dict=self.pieces_dict,
            move_ordering_dict={},
            any_to_ordering_dict={},
        )

        # 反復深化探索
        value = 0
        for depth in range(1, int(self.max_depth) + 1):
            na.max_depth = depth
            na.best_move_pv = "resign"
            na.any_to_ordering_dict[board.move_number + depth - 1] = {}
            for i in "123456789":
                for j in "abcdefghi":
                    na.any_to_ordering_dict[board.move_number + depth - 1][i + j] = 0
            value = na.search(board=board, depth=depth, alpha=-1000000, beta=1000000)

            # 時間切れでないなら更新
            if abs(value) != 1000000:
                elapsed_time = int(time.perf_counter() * 1000 - self.start_time)
                info_text = f"info depth {int(depth)} "
                info_text += f"seldepth {int(na.max_board_number - board.move_number)} "
                info_text += f"time {elapsed_time} "
                info_text += (
                    f"nodes {na.num_searched} score cp {value} pv {na.best_move_pv} "
                )
                if elapsed_time != 0:
                    info_text += f"nps {int(na.num_searched / (elapsed_time / 1000))}"
                print(info_text)
                self.best_move_pv = na.best_move_pv
            else:
                print(info_text)
                break

            # 次深度が時間内に終わりそうにないなら
            elapsed_time = int(time.perf_counter() * 1000 - self.start_time)
            if self.max_time < elapsed_time * 7:
                break

        return self.best_move_pv


def main() -> None:
    import pickle

    pieces_dict, pieces_in_hand_dict = pickle.load(open("eval.pkl", "rb"))

    board = cshogi.Board()
    ids = IterativeDeepeningSearch(
        max_depth=7,
        start_time=time.perf_counter() * 1000,
        max_time=1000000000000,
        pieces_in_hand_dict=pieces_in_hand_dict,
        pieces_dict=pieces_dict,
    )
    best_move = ids.search(board=board)
    print(f"bestmove {best_move}")


if __name__ == "__main__":
    main()
