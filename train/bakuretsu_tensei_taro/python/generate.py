import dataclasses
import datetime
import os
import pickle
import subprocess
import typing
import random

import cshogi


@dataclasses.dataclass
class Engine:
    path: str

    def __post_init__(self):
        self.engine = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(self.path),
        )

    def usi(self, verbose: bool = False):
        cmd = "usi\n"
        if verbose:
            print(f"In: {cmd}", end="")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

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
            elif out == "usiok":
                break
            elif " ".join(out.split(" ")[:2]) == "id name":
                self.name = " ".join(out.split(" ")[2:])
            elif " ".join(out.split(" ")[:2]) == "id author":
                self.author = " ".join(out.split(" ")[2:])

    def isready(self, verbose: bool = False):
        cmd = "isready\n"
        if verbose:
            print(f"In: {cmd}", end="")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

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

    def setoption(self, name: str, value: str, verbose: bool = False):
        cmd = f"setoption name {name} value {value}\n"
        if verbose:
            print(f"In: {cmd}", end="")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()

    def usinewgame(self):
        print("can not use command")

    def position(self, sfen: str = "startpos", moves: str = "", verbose: bool = False):
        cmd = f"position {sfen}"
        if moves == "":
            cmd += "\n"
        else:
            cmd += f" moves{moves}\n"
        if verbose:
            print(f"In: {cmd}", end="")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def go(
        self,
        btime: int,
        wtime: int,
        byoyomi: int = -1,
        binc: int = -1,
        winc: int = -1,
        verbose: bool = False,
    ):
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
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

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
                out = out.split(" ")
                for i in range(len(out)):
                    if out[i] == "cp":
                        value = int(out[i + 1])
                        break
            elif out.split(" ")[0] == "bestmove":
                return (out.split(" ")[1], value)

    def stop(self):
        print("can not use command")

    def ponderhit(self):
        print("can not use command")

    def quit(self, verbose: bool = False):
        cmd = "quit\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

        self.engine.wait()
        self.engine = None
        self.name = None
        self.author = None

    def gameover(self):
        print("can not use command")

    def extra_load(self, path: str, verbose: bool = False):
        cmd = f"extra load {path}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_save(self, path: str, verbose: bool = False):
        cmd = f"extra save {path}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_make(self, sfen: str, verbose: bool = False):
        cmd = f"extra make {sfen}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

        if self.engine.stdout is not None:
            self.engine.stdout.flush()
            out = self.engine.stdout.readline().replace(b"\n", b"").decode("cp932")
        else:
            raise AttributeError()
        if verbose:
            print(f"Out: {out}")

        return out

    def extra_entry(self, sfen: str, winner: str, verbose: bool = False):
        cmd = f"extra entry {sfen} {winner}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()


def main(
        train_engine_path: str,
        target_engine_path: typing.List[str],
        choice_weights: typing.List[float],
        games: int):
    # 学習対象のエンジンの起動
    train_engine = Engine(path=train_engine_path)
    train_engine.usi(verbose=False)
    print(f"engine: {train_engine.name}")
    print(f"author: {train_engine.author}")
    train_engine.setoption(name="DepthLimit", value="4", verbose=False)
    train_engine.isready(verbose=False)
    train_engine.extra_load("book.json")

    # 仮想敵のエンジンの起動
    game_results: typing.Dict[str, typing.Dict[str, int]] = {}
    target_engine_list: typing.List[Engine] = []
    for path in target_engine_path:
        target_engine_list.append(Engine(path=path))
        target_engine_list[-1].usi(verbose=False)
        print(f"engine: {target_engine_list[-1].name}")
        print(f"author: {target_engine_list[-1].author}")
        target_engine_list[-1].setoption(name="DepthLimit", value="4", verbose=False)
        target_engine_list[-1].isready(verbose=False)
        if target_engine_list[-1].name is not None:
            game_results[target_engine_list[-1].name] = {"wins": 0, "loses": 0, "draw": 0}
        else:
            raise ValueError

    # 対局
    game_sfen: typing.Dict[
        int, typing.Dict[str, typing.List[typing.Union[int, str, None]]]
    ] = {}
    for i in range(games):
        target_engine = random.choices(target_engine_list, weights=choice_weights, k=1)[0]
        board: typing.Any = cshogi.Board()  # type:ignore
        moves = ""
        winner = ""
        make = True
        entry = True
        entry_position: typing.List[str] = []
        game_sfen[i] = {"moves": [], "value": [], "winner": []}
        while winner == "":
            if board.move_number % 2 == i % 2:
                train_engine.position(sfen="startpos", moves=moves, verbose=False)
                if make:
                    sfen = " ".join(board.sfen().split(" ")[:-1])
                    bestmove = train_engine.extra_make(sfen=sfen, verbose=False)
                    if bestmove == "None":
                        make = False
                        entry_position.append(sfen)
                        bestmove, value = train_engine.go(
                            btime=0, wtime=0, byoyomi=10000, verbose=False
                        )
                        if value is None:
                            game_sfen[i]["value"].append(value)
                        elif board.move_number % 2 == 1:
                            game_sfen[i]["value"].append(value)
                        else:
                            game_sfen[i]["value"].append(-value)
                    else:
                        game_sfen[i]["value"].append(None)
                else:
                    bestmove, value = train_engine.go(
                        btime=0, wtime=0, byoyomi=10000, verbose=False
                    )
                    if value is None:
                        game_sfen[i]["value"].append(value)
                    elif board.move_number % 2 == 1:
                        game_sfen[i]["value"].append(value)
                    else:
                        game_sfen[i]["value"].append(-value)
            else:
                if entry:
                    sfen = " ".join(board.sfen().split(" ")[:-1])
                    entry_position.append(sfen)
                    if not make:
                        entry = False
                target_engine.position(sfen="startpos", moves=moves, verbose=False)
                bestmove, _ = target_engine.go(
                    btime=0, wtime=0, byoyomi=10000, verbose=False
                )
                game_sfen[i]["value"].append(None)

            board.push_usi(bestmove)
            moves += f" {bestmove}"
            game_sfen[i]["moves"].append(bestmove)
            if board.is_game_over():
                if board.move_number % 2 == 1:
                    winner = "w"
                else:
                    winner = "b"
            if board.is_draw() == 1:
                winner = "d"
            if bestmove == "win":
                if board.move_number % 2 == 1:
                    winner = "b"
                else:
                    winner = "w"
            if board.move_number > 512:
                winner = "d"
        if target_engine.name is not None:
            if winner == "b":
                if i % 2 == 1:
                    game_results[target_engine.name]["wins"] += 1
                else:
                    game_results[target_engine.name]["loses"] += 1
            elif winner == "w":
                if i % 2 == 0:
                    game_results[target_engine.name]["wins"] += 1
                else:
                    game_results[target_engine.name]["loses"] += 1
            else:
                game_results[target_engine.name]["draw"] += 1
        else:
            raise ValueError
        game_sfen[i]["winner"].append(winner)

        for sfen in entry_position:
            train_engine.extra_entry(sfen=sfen, winner=winner, verbose=False)

        print(game_results)
        train_wins = 0
        target_wins = 0
        draw = 0
        for key in game_results:
            train_wins += game_results[key]["wins"]
            target_wins += game_results[key]["loses"]
            draw += game_results[key]["draw"]
        rate = (train_wins + draw * 0.5) / (train_wins + target_wins + draw)
        print(f"train_engine win_rate: {rate}")

    train_engine.extra_save(path="book.json", verbose=False)
    train_engine.quit(verbose=False)
    for i in range(len(target_engine_list)):
        target_engine_list[i].quit(verbose=False)

    print(game_results)
    date = datetime.datetime.today()
    with open(
        f"sfen_{date.year}{date.month}{date.day}{date.hour}{date.minute}.pkl", "wb"
    ) as f:
        pickle.dump(game_sfen, f)


if __name__ == "__main__":
    train_engine_path = "train_engine_path"  # 学習対象のエンジンパス
    target_engine_path = [  # 仮想敵のエンジンパス
        "target_engine_path1",
        "target_engine_path2",
    ]
    choice_weights = [1 / 0.698, 1 / 0.339]  # 仮想敵のエンジンの選択重み
    games = 400  # 対局数

    main(
        train_engine_path=train_engine_path,
        target_engine_path=target_engine_path,
        choice_weights=choice_weights,
        games=games,
    )
