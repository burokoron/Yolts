"""
評価関数がランダムの将棋エンジン
"""

import dataclasses
import typing

import cshogi

from .search import NegaAlpha


@dataclasses.dataclass
class BakuretsuSutegomaTaro:
    """
    評価関数がランダムの将棋エンジン
    """

    def __post_init__(self) -> None:
        self.engine_name = "爆裂駒捨太郎"
        self.version = "Version 1.1.0"
        self.author = "burokoron"

    def start(self) -> None:
        """
        対局するためのメイン関数
        """

        while True:
            # GUIからのコマンドを受け取って対応
            inputs = input().split(" ")

            if inputs[0] == "usi":  # エンジン名を返答
                self.usi()
            elif inputs[0] == "isready":  # 対局準備
                self.isready()
            elif inputs[0] == "setoption":  # エンジンのパラメータ設定
                self.setoption()
            elif inputs[0] == "usinewgame":  # 新規対局準備
                self.usinewgame()
            elif inputs[0] == "position":  # 現局面の反映
                self.position(inputs=inputs[1:])
            elif inputs[0] == "go":  # 思考して指し手を返答
                self.go(inputs=inputs[1:])
            elif inputs[0] == "stop":  # 思考停止コマンド
                self.stop()
            elif inputs[0] == "ponderhit":  # 先読みが当たった場合
                self.ponderhit()
            elif inputs[0] == "quit":  # 強制終了
                self.quit()
            elif inputs[0] == "gameover":  # 対局終了
                self.gameover()
            else:
                # 未対応のコマンドなら
                print(f"Can not be recognized = {inputs}")

    def usi(self) -> None:
        """
        エンジン名(バージョン番号付き)を返答
        """

        print(f"id name {self.engine_name} {self.version}")
        print(f"id author {self.author}")
        print("usiok")

    def isready(self) -> None:
        """
        対局の準備をする
        """

        # 特に準備はなし
        # 準備OKと返答
        print("readyok")

    def setoption(self) -> None:
        """
        エンジンのパラメータを設定する
        """

        pass

    def usinewgame(self) -> None:
        """
        新規対局の準備をする
        """

        pass

    def position(self, inputs: typing.List[str]) -> None:
        """
        現局面の反映

        Args:
            sfen: str
                positionコマンドで送られてくる局面情報
        """

        move_list = []
        if inputs[0] == "startpos":  # 平手局面なら
            self.board = cshogi.Board()
            if len(inputs) != 1:  # 指し手があるなら
                move_list = inputs[2:]
        elif inputs[0] == "sfen":  # 指定局面なら
            self.board = cshogi.Board(" ".join(inputs[1:5]))
            if len(inputs) != 4:  # 指し手があるなら
                move_list = inputs[6:]

        # 指し手を再現
        for move in move_list:
            self.board.push_usi(move)

    def go(self, inputs: typing.List[str]) -> None:
        """
        評価値はランダムだが全探索する
        """

        go_info = {}
        for i in range(0, len(inputs), 2):
            go_info[inputs[i]] = int(inputs[i + 1])

        max_depth = 3.0
        margin_time = 1000
        if self.board.turn == 0:
            max_time = go_info["btime"]
            if "binc" in go_info:
                max_time = go_info["binc"]
        else:
            max_time = go_info["wtime"]
            if "winc" in go_info:
                max_time = go_info["winc"]
        if "byoyomi" in go_info:
            max_time += go_info["byoyomi"]
        max_time -= margin_time

        na = NegaAlpha(max_depth=max_depth, max_time=max_time)
        value = na.search(
            board=self.board, depth=max_depth, alpha=-1000000, beta=1000000
        )

        info_text = f"info depth {int(max_depth)} "
        info_text += f"seldepth {int(na.max_board_number - self.board.move_number)} "
        info_text += f"nodes {na.num_searched} score cp {value} pv {na.best_move_pv}"
        print(info_text)
        print(f"bestmove {na.best_move_pv}")

    def stop(self) -> None:
        """
        思考停止コマンドに対応する
        """

        # そもそも思考中に思考停止コマンドを受け付けない
        pass

    def ponderhit(self) -> None:
        """
        先読みが当たった場合に対応する
        """

        # そもそも先読みしない
        pass

    def quit(self) -> None:
        """
        強制終了
        """

        # すぐさま終了はできないが終了はする
        exit(0)

    def gameover(self) -> None:
        """
        対局終了通知に対応する
        """

        # たぶん対応する必要はない
        pass
