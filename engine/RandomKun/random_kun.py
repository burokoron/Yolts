"""
評価関数がランダムの将棋エンジン
"""

import dataclasses
import typing
import random

import cshogi


@dataclasses.dataclass
class RandomKun:
    """
    評価関数がランダムの将棋エンジン
    """

    def __post_init__(self) -> None:
        self.engine_name = "RandomKun"
        self.version = "Version 1.0.0"
        self.author = "burokoron"

    def start(self) -> None:
        """
        対局するためのメイン関数
        """

        while(True):
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
        合法手をランダムで返す
        """
        move_list = []
        for move in self.board.legal_moves:
            move_list.append(cshogi.move_to_usi(move))

        if len(move_list) != 0:
            best_move = random.choice(move_list)
            print(f"bestmove {best_move}")
        else:
            print("bestmove resign")

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
