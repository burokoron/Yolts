import dataclasses
import os
import pickle
import subprocess
import typing

import cshogi
from tqdm import tqdm


@dataclasses.dataclass
class Engine:
    path: str

    def __post_init__(self) -> None:
        self.engine: typing.Any = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(self.path),
        )
        self.name: typing.Union[None, str] = None
        self.author: typing.Union[None, str] = None

    def usi(self, verbose: bool = False) -> None:
        cmd = "usi\n"
        if verbose:
            assert self.engine is not None
            assert self.engine.stdin is not None

            print(f"In: {cmd}", end="")
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()

        while True:
            assert self.engine is not None
            assert self.engine.stdout is not None

            self.engine.stdout.flush()
            out = self.engine.stdout.readline().replace(b"\n", b"").decode("cp932")

            if verbose:
                print(f"Out: {out}")

            if out == "":
                raise EOFError()
            elif out == "usiok":
                break
            elif " ".join(out.split(" ")[:2]) == "id name":
                self.name = " ".join(out.split(" ")[2:])
            elif " ".join(out.split(" ")[:2]) == "id author":
                self.author = " ".join(out.split(" ")[2:])

    def isready(self, verbose: bool = False) -> None:
        cmd = "isready\n"
        if verbose:
            print(f"In: {cmd}", end="")

        assert self.engine is not None
        assert self.engine.stdin is not None

        self.engine.stdin.write(cmd.encode("cp932"))
        self.engine.stdin.flush()

        while True:
            if self.engine.stdout is not None:
                self.engine.stdout.flush()
                out = self.engine.stdout.readline().replace(b"\n", b"").decode("cp932")
            else:
                raise AttributeError()
            if verbose:
                print(f"Out: {out}")

            if out == "":
                raise EOFError()
            if out == "readyok":
                break

    def setoption(self, name: str, value: str, verbose: bool = False) -> None:
        cmd = f"setoption name {name} value {value}\n"
        if verbose:
            print(f"In: {cmd}", end="")

        assert self.engine is not None
        assert self.engine.stdin is not None

        self.engine.stdin.write(cmd.encode("cp932"))
        self.engine.stdin.flush()

    def usinewgame(self) -> None:
        print("can not use command")

    def position(
        self, sfen: str = "startpos", moves: str = "", verbose: bool = False
    ) -> None:
        cmd = f"position {sfen}"
        if moves == "":
            cmd += "\n"
        else:
            cmd += f" moves{moves}\n"
        if verbose:
            print(f"In: {cmd}", end="")

        assert self.engine is not None
        assert self.engine.stdin is not None

        self.engine.stdin.write(cmd.encode("cp932"))
        self.engine.stdin.flush()

    def go(
        self,
        btime: int,
        wtime: int,
        byoyomi: int = -1,
        binc: int = -1,
        winc: int = -1,
        verbose: bool = False,
    ) -> typing.Tuple[
        str, typing.Union[int, None], typing.Union[int, None], typing.Union[int, None]
    ]:
        cmd = f"go btime {btime} wtime {wtime} "
        if byoyomi >= 0:
            cmd += f"byoyomi {byoyomi}\n"
        else:
            if binc >= 0:
                cmd += f"binc {binc} "
            if winc >= 0:
                cmd += f"winc {winc}\n"
        if verbose:
            print(f"In: {cmd}", end="")

        assert self.engine is not None
        assert self.engine.stdin is not None

        self.engine.stdin.write(cmd.encode("cp932"))
        self.engine.stdin.flush()

        time = None
        nodes = None
        value = None
        while True:
            if self.engine.stdout is not None:
                self.engine.stdout.flush()
                out = self.engine.stdout.readline().replace(b"\n", b"").decode("cp932")
            else:
                raise AttributeError()
            if verbose:
                print(f"Out: {out}")

            if out == "":
                raise EOFError()
            elif out.split(" ")[0] == "info":
                out_list = out.split(" ")
                for i in range(len(out_list)):
                    if out_list[i] == "time":
                        time = int(out_list[i + 1])
                    if out_list[i] == "nodes":
                        nodes = int(out_list[i + 1])
                    if out_list[i] == "cp":
                        value = int(out_list[i + 1])
            elif out.split(" ")[0] == "bestmove":
                return (out.split(" ")[1], time, nodes, value)

    def stop(self) -> None:
        print("can not use command")

    def ponderhit(self) -> None:
        print("can not use command")

    def quit(self, verbose: bool = False) -> None:
        cmd = "quit\n"
        if verbose:
            print(f"In: {cmd}")

        assert self.engine is not None
        assert self.engine.stdin is not None

        self.engine.stdin.write(cmd.encode("cp932"))
        self.engine.stdin.flush()

        self.engine.wait()
        self.engine = None
        self.name = None
        self.author = None

    def gameover(self) -> None:
        print("can not use command")


def main(engine_path: str, kifu_path: str) -> None:
    # エンジンの起動
    engine = Engine(path=engine_path)
    engine.usi(verbose=True)
    print(f"engine: {engine.name}")
    print(f"author: {engine.author}")
    engine.setoption(name="DepthLimit", value="4", verbose=False)
    engine.isready(verbose=False)

    time_list = []
    nodes_list = []

    # 棋譜の読み込み
    same_sfen: typing.Set[str] = set()
    file_list = [file for file in os.listdir(kifu_path) if file.split(".")[-1] == "pkl"]
    for file in tqdm(file_list[:1], desc="file"):
        with open(file, "rb") as f:
            kifu_dict = pickle.load(f)
        for key in tqdm(kifu_dict, desc="kifu"):
            moves = kifu_dict[key]["moves"]
            values = kifu_dict[key]["value"]

            board: typing.Any = cshogi.Board()
            for move, value in zip(moves, values):
                board.push_usi(move)
                if value is not None:
                    sfen = board.sfen()
                    if " ".join(sfen.split(" ")[:-1]) in same_sfen:
                        continue
                    else:
                        same_sfen.add(" ".join(sfen.split(" ")[:-1]))
                    # 探索のベンチマーク
                    if int(key) % 9 == 0:
                        engine.position(sfen=f"sfen {sfen}", moves="", verbose=False)
                        _, time, nodes, _ = engine.go(
                            btime=0, wtime=0, byoyomi=10000, verbose=False
                        )
                        if time is not None and nodes is not None:
                            time_list.append(time)
                            nodes_list.append(nodes)

    # エンジンの終了
    engine.quit()

    print("time mean: ", sum(time_list) / len(time_list))
    print("nodes mean: ", sum(nodes_list) / len(nodes_list))


if __name__ == "__main__":
    engine_path = "engine_path"  # 学習対象のエンジンパス
    kifu_path = "./"  # 棋譜があるルートフォルダ

    main(engine_path=engine_path, kifu_path=kifu_path)
