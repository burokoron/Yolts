import dataclasses
import datetime
import os
import pickle
import subprocess
import threading
import time
import typing

import cshogi
from rich import color
from rich.progress import Progress, TaskID

game_results: typing.Dict[str, typing.Dict[str, int]]
game_sfen: typing.Dict[int, typing.Dict[str, typing.List[typing.Union[int, str, None]]]]
game_idx: int
target_engine_path2name: typing.Dict[str, str]


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
    ) -> typing.Tuple[str, typing.Union[int, None]]:
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
                    if out_list[i] == "cp":
                        value = int(out_list[i + 1])
                        break
            elif out.split(" ")[0] == "bestmove":
                return (out.split(" ")[1], value)

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

    def extra_load(self, path: str, verbose: bool = False) -> None:
        cmd = f"extra load {path}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_save(self, path: str, verbose: bool = False) -> None:
        cmd = f"extra save {path}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_make(self, sfen: str, verbose: bool = False) -> typing.Any:
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

    def extra_entry(self, sfen: str, winner: str, verbose: bool = False) -> None:
        cmd = f"extra entry {sfen} {winner}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("cp932"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()


def generate(
    lock: threading.Lock,
    book_engine: Engine,
    train_engine_path: str,
    target_engine_path: str,
    games: int,
) -> None:
    # 学習対象のエンジンの起動
    train_engine = Engine(path=train_engine_path)
    train_engine.usi(verbose=True)
    print(f"train engine: {train_engine.name}")
    print(f"author: {train_engine.author}")
    train_engine.setoption(name="DepthLimit", value="4", verbose=False)
    train_engine.isready(verbose=False)
    train_engine.extra_load("book.json")
    # 仮想敵のエンジンの起動
    global game_results
    target_engine = Engine(path=target_engine_path)
    target_engine.usi(verbose=True)
    print(f"target engine: {target_engine.name}")
    print(f"author: {target_engine.author}")
    target_engine.setoption(name="DepthLimit", value="4", verbose=False)
    target_engine.isready(verbose=False)
    if target_engine.name is not None:
        lock.acquire()
        target_engine_path2name[target_engine_path] = target_engine.name
        game_results[target_engine_path] = {
            "wins": 0,
            "loses": 0,
            "draw": 0,
        }
        lock.release()
    else:
        raise ValueError

    # 対局
    global game_sfen
    global game_idx
    for _ in range(games):
        board: typing.Any = cshogi.Board()
        moves = ""
        winner = ""
        make = True
        entry = True
        entry_position: typing.List[str] = []
        lock.acquire()
        i = game_idx
        game_sfen[i] = {"moves": [], "value": [], "winner": []}
        game_idx += 1
        lock.release()
        while winner == "":
            if board.move_number % 2 == i % 2:
                train_engine.position(sfen="startpos", moves=moves, verbose=False)
                if make:
                    sfen = " ".join(board.sfen().split(" ")[:-1])
                    lock.acquire()
                    book_engine.position(sfen="startpos", moves=moves, verbose=False)
                    bestmove = book_engine.extra_make(sfen=sfen, verbose=False)
                    lock.release()
                    if bestmove == "None":
                        make = False
                        entry_position.append(sfen)
                        bestmove, value = train_engine.go(
                            btime=0, wtime=0, byoyomi=10000, verbose=False
                        )
                        lock.acquire()
                        if value is None:
                            game_sfen[i]["value"].append(value)
                        elif board.move_number % 2 == 1:
                            game_sfen[i]["value"].append(value)
                        else:
                            game_sfen[i]["value"].append(-value)
                        lock.release()
                    else:
                        lock.acquire()
                        game_sfen[i]["value"].append(None)
                        lock.release()
                else:
                    bestmove, value = train_engine.go(
                        btime=0, wtime=0, byoyomi=10000, verbose=False
                    )
                    lock.acquire()
                    if value is None:
                        game_sfen[i]["value"].append(value)
                    elif board.move_number % 2 == 1:
                        game_sfen[i]["value"].append(value)
                    else:
                        game_sfen[i]["value"].append(-value)
                    lock.release()
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
                lock.acquire()
                game_sfen[i]["value"].append(None)
                lock.release()

            board.push_usi(bestmove)
            moves += f" {bestmove}"
            lock.acquire()
            game_sfen[i]["moves"].append(bestmove)
            lock.release()
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

        lock.acquire()
        if winner == "b":
            if i % 2 == 1:
                game_results[target_engine_path]["wins"] += 1
            else:
                game_results[target_engine_path]["loses"] += 1
        elif winner == "w":
            if i % 2 == 0:
                game_results[target_engine_path]["wins"] += 1
            else:
                game_results[target_engine_path]["loses"] += 1
        else:
            game_results[target_engine_path]["draw"] += 1
        game_sfen[i]["winner"].append(winner)

        for sfen in entry_position:
            book_engine.extra_entry(sfen=sfen, winner=winner, verbose=False)
        lock.release()

    target_engine.quit(verbose=False)


def ganerete_progress(
    target_engine_path: typing.List[str], choice_weights: typing.List[float], games: int
) -> None:
    global game_results
    global target_engine_path2name

    while len(game_results) != len(target_engine_path):
        time.sleep(1)

    ignore_color = ["black", "magenta", "cyan", "white", "bright_black"]
    color_list = [c for c in color.ANSI_COLOR_NAMES.keys() if c not in ignore_color]
    task_list: typing.List[TaskID] = []
    game_progress: typing.Dict[str, int] = {}
    with Progress(auto_refresh=False) as progress:
        for i in range(len(target_engine_path)):
            task_list.append(
                progress.add_task(
                    f"[{color_list[i]}]{target_engine_path2name[target_engine_path[i]]}",
                    total=int(choice_weights[i] / sum(choice_weights) * games),
                )
            )
            game_progress[target_engine_path[i]] = 0

        while not progress.finished:
            for i in range(len(target_engine_path)):
                wins = game_results[target_engine_path[i]]["wins"]
                loses = game_results[target_engine_path[i]]["loses"]
                draw = game_results[target_engine_path[i]]["draw"]
                advance = float(
                    wins + loses + draw - game_progress[target_engine_path[i]]
                )
                progress.update(task_list[i], advance=advance)
                game_progress[target_engine_path[i]] = wins + loses + draw
            progress.refresh()
            time.sleep(1)


def main(
    train_engine_path: str,
    target_engine_path: typing.List[str],
    choice_weights: typing.List[float],
    games: int,
) -> None:
    # 学習対象のエンジン(定跡用)の起動
    book_engine = Engine(path=train_engine_path)
    book_engine.usi(verbose=True)
    print(f"book engine: {book_engine.name}")
    print(f"author: {book_engine.author}")
    book_engine.setoption(name="DepthLimit", value="4", verbose=False)
    book_engine.isready(verbose=False)
    book_engine.extra_load("book.json")

    # 仮想敵のエンジンごとにスレッドを起動
    global game_sfen
    global game_idx
    global game_results
    global target_engine_path2name
    game_sfen = {}
    game_idx = 0
    game_results = {}
    target_engine_path2name = {}
    lock = threading.Lock()
    for i in range(len(target_engine_path)):
        # スレッド実行
        threading.Thread(
            target=generate,
            args=[
                lock,
                book_engine,
                train_engine_path,
                target_engine_path[i],
                int(choice_weights[i] / sum(choice_weights) * games),
            ],
        ).start()
    threading.Thread(
        target=ganerete_progress, args=[target_engine_path, choice_weights, games]
    ).start()

    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    book_engine.extra_save(path="book.json", verbose=False)
    book_engine.quit(verbose=False)

    train_wins = 0
    target_wins = 0
    draw = 0
    for path in target_engine_path:
        name = target_engine_path2name[path]
        wins = game_results[path]["wins"]
        loses = game_results[path]["loses"]
        draw = game_results[path]["draw"]
        train_wins += wins
        target_wins += loses
        draw += draw
        print(f"{name}, wins={wins}, loses={loses}, draw={draw}")
    rate = (train_wins + draw * 0.5) / (train_wins + target_wins + draw)
    print(f"train_engine win_rate: {rate}")

    date = datetime.datetime.today()
    with open(
        f"sfen_{date.year:04}{date.month:02}{date.day:02}{date.hour:02}{date.minute:02}.pkl",
        "wb",
    ) as f:
        pickle.dump(game_sfen, f)


if __name__ == "__main__":
    train_engine_path = "engine_path_00"  # 学習対象のエンジンパス
    target_engine_path = [  # 仮想敵のエンジンパス
        "engine_path_01",
        "engine_path_02",
        "engine_path_03",
        "engine_path_04",
        "engine_path_05",
        "engine_path_06",
        "engine_path_07",
        "engine_path_08",
        "engine_path_09",
        "engine_path_10",
    ]
    choice_weights = [
        1 / 0.933,
        1 / 0.895,
        1 / 0.896,
        1 / 0.934,
        1 / 0.596,
        1 / 0.494,
        1 / 0.464,
        1 / 0.496,
        1 / 0.517,
        1 / 0.389,
    ]  # 仮想敵のエンジンの選択重み
    games = 1000  # 対局数

    main(
        train_engine_path=train_engine_path,
        target_engine_path=target_engine_path,
        choice_weights=choice_weights,
        games=games,
    )
