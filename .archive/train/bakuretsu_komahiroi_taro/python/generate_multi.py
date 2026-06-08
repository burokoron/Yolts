import dataclasses
import datetime
import os
import pickle
import subprocess
import threading
import time
import typing

import cshogi
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
            self.engine.stdin.write(cmd.encode("utf-8"))
            self.engine.stdin.flush()

        while True:
            assert self.engine is not None
            assert self.engine.stdout is not None

            self.engine.stdout.flush()
            out = self.engine.stdout.readline().replace(b"\n", b"").decode("utf-8")

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

        self.engine.stdin.write(cmd.encode("utf-8"))
        self.engine.stdin.flush()

        while True:
            if self.engine.stdout is not None:
                self.engine.stdout.flush()
                out = self.engine.stdout.readline().replace(b"\n", b"").decode("utf-8")
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

        self.engine.stdin.write(cmd.encode("utf-8"))
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

        self.engine.stdin.write(cmd.encode("utf-8"))
        self.engine.stdin.flush()

    def go(
        self,
        btime: int,
        wtime: int,
        byoyomi: int = -1,
        binc: int = -1,
        winc: int = -1,
        verbose: bool = False,
    ) -> typing.Tuple[typing.Union[str, None], typing.Union[int, None]]:
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

        self.engine.stdin.write(cmd.encode("utf-8"))
        self.engine.stdin.flush()

        value = None
        outs = []
        while True:
            if self.engine.stdout is not None:
                self.engine.stdout.flush()
                out = self.engine.stdout.readline().replace(b"\n", b"").decode("utf-8")
                outs.append(out)
            else:
                raise AttributeError()
            if verbose:
                print(f"Out: {out}")

            if out == "":
                print(outs)
                # raise EOFError()
                return None, None
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

        self.engine.stdin.write(cmd.encode("utf-8"))
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
            self.engine.stdin.write(cmd.encode("utf-8"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_save(self, path: str, verbose: bool = False) -> None:
        cmd = f"extra save {path}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("utf-8"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

    def extra_make(self, sfen: str, verbose: bool = False) -> typing.Any:
        cmd = f"extra make {sfen}\n"
        if verbose:
            print(f"In: {cmd}")
        if self.engine is not None and self.engine.stdin is not None:
            self.engine.stdin.write(cmd.encode("utf-8"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()

        if self.engine.stdout is not None:
            self.engine.stdout.flush()
            out = self.engine.stdout.readline().replace(b"\n", b"").decode("utf-8")
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
            self.engine.stdin.write(cmd.encode("utf-8"))
            self.engine.stdin.flush()
        else:
            raise AttributeError()


def boot_engine(engine_path: str, mode: str) -> Engine:
    engine = Engine(path=engine_path)
    engine.usi(verbose=True)
    print(f"{mode} engine: {engine.name}")
    engine.setoption(name="DepthLimit", value="4", verbose=True)
    engine.setoption(name="UseBook", value="false", verbose=True)
    if mode == "train":
        engine.setoption(name="SearchMode", value="Absolute-27-Point", verbose=True)
    engine.isready(verbose=True)

    return engine


def generate(
    lock: threading.Lock,
    book_train_engine: Engine,
    train_engine_path: str,
    target_engine_path: str,
    games: int,
) -> None:
    # 学習対象のエンジンの起動（定跡はbook_train_engineが担当するため、ここでは読み込まない）
    train_engine = boot_engine(engine_path=train_engine_path, mode="train")
    # 仮想敵のエンジンの起動
    global game_results
    target_engine = boot_engine(engine_path=target_engine_path, mode="target")
    if target_engine.name is not None:
        lock.acquire()
        target_engine_path2name[target_engine_path] = target_engine.name
        if target_engine_path not in game_results:
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
        make = True  # 学習側の定跡を使うフラグ
        make_target = True  # 仮想敵側の定跡を使うフラグ
        entry = True
        entry_position: typing.List[str] = []
        lock.acquire()
        i = game_idx
        game_sfen[i] = {"moves": [], "value": [], "winner": []}
        game_idx += 1
        lock.release()
        while winner == "":
            if board.move_number % 2 == i % 2:
                if make:
                    sfen = " ".join(board.sfen().split(" ")[:-1])
                    lock.acquire()
                    book_train_engine.position(
                        sfen="startpos", moves=moves, verbose=False
                    )
                    bestmove = book_train_engine.extra_make(sfen=sfen, verbose=False)
                    entry_position.append(sfen)
                    lock.release()
                    if bestmove == "None":
                        make = False
                        bestmove = None
                        value = None
                        while bestmove is None:
                            train_engine.position(
                                sfen="startpos", moves=moves, verbose=False
                            )
                            bestmove, value = train_engine.go(
                                btime=0, wtime=0, byoyomi=10000, verbose=False
                            )
                            if bestmove is None:
                                print(board)
                                train_engine = boot_engine(
                                    engine_path=train_engine_path, mode="train"
                                )
                        lock.acquire()
                        if board.move_number % 2 == 1:
                            game_sfen[i]["value"].append(value)
                        else:
                            assert value is not None
                            game_sfen[i]["value"].append(-value)
                        lock.release()
                    else:
                        lock.acquire()
                        game_sfen[i]["value"].append(None)
                        lock.release()
                else:
                    bestmove = None
                    while bestmove is None:
                        train_engine.position(
                            sfen="startpos", moves=moves, verbose=False
                        )
                        bestmove, value = train_engine.go(
                            btime=0, wtime=0, byoyomi=10000, verbose=False
                        )
                        if bestmove is None:
                            print(board)
                            train_engine = boot_engine(
                                engine_path=train_engine_path, mode="train"
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
                # 仮想敵の手番
                sfen = " ".join(board.sfen().split(" ")[:-1])
                if entry:
                    entry_position.append(sfen)
                    if not make:
                        entry = False
                bestmove = None
                value = None
                # まずターゲット側の定跡を参照（共有定跡を使用）
                if make_target:
                    lock.acquire()
                    book_train_engine.position(
                        sfen="startpos", moves=moves, verbose=False
                    )
                    target_book_move = book_train_engine.extra_make(
                        sfen=sfen, verbose=False
                    )
                    lock.release()
                    if target_book_move != "None":
                        bestmove = target_book_move
                    else:
                        make_target = False

                # 定跡がなければエンジンで探索
                while bestmove is None:
                    target_engine.position(sfen="startpos", moves=moves, verbose=False)
                    bestmove, value = target_engine.go(
                        btime=0, wtime=0, byoyomi=10000, verbose=False
                    )
                    if bestmove is None:
                        print(board)
                        target_engine = boot_engine(
                            engine_path=target_engine_path, mode="target"
                        )
                lock.acquire()
                if value is None:
                    game_sfen[i]["value"].append(value)
                elif board.move_number % 2 == 1:
                    game_sfen[i]["value"].append(value)
                else:
                    game_sfen[i]["value"].append(-value)
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

        for position in entry_position:
            book_train_engine.extra_entry(sfen=position, winner=winner, verbose=False)
        lock.release()

    train_engine.quit(verbose=False)
    target_engine.quit(verbose=False)


def generate_progress(target_engine_path: str, games: int) -> None:
    global game_results
    global target_engine_path2name

    # 待機: エンジン名が取得されて結果テーブルが初期化されるまで
    while target_engine_path not in game_results:
        time.sleep(1)

    task: TaskID
    done = 0
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn(
            " | W:{task.fields[wins]} L:{task.fields[loses]} D:{task.fields[draw]}"
        ),
        TimeRemainingColumn(),
        TextColumn(" (elapsed: "),
        TimeElapsedColumn(),
        TextColumn(")"),
        auto_refresh=False,
    ) as progress:
        task = progress.add_task(
            f"[green]{target_engine_path2name[target_engine_path]}",
            total=games,
            wins=0,
            loses=0,
            draw=0,
        )

        while not progress.finished:
            wins = game_results[target_engine_path]["wins"]
            loses = game_results[target_engine_path]["loses"]
            draw = game_results[target_engine_path]["draw"]
            total_done = wins + loses + draw
            advance = float(total_done - done)
            progress.update(
                task,
                advance=advance,
                wins=wins,
                loses=loses,
                draw=draw,
            )
            done = total_done
            progress.refresh()
            time.sleep(1)


def main(
    train_engine_path: str,
    target_engine_path: str,
    parallel: int,
    games: int,
    save_path: str,
) -> None:
    # 学習用定跡エンジンの起動（定跡のみ使用）
    book_train_engine = Engine(path=train_engine_path)
    book_train_engine.usi(verbose=True)
    print(f"book train engine: {book_train_engine.name}")
    print(f"author: {book_train_engine.author}")
    book_train_engine.setoption(name="DepthLimit", value="4", verbose=True)
    book_train_engine.setoption(
        name="SearchMode", value="Absolute-27-Point", verbose=True
    )
    book_train_engine.isready(verbose=True)

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
    # 並列数に応じてゲーム数を割り当て
    base = games // parallel
    rem = games % parallel
    for i in range(parallel):
        n_games = base + (1 if i < rem else 0)
        if n_games == 0:
            continue
        threading.Thread(
            target=generate,
            args=[
                lock,
                book_train_engine,
                train_engine_path,
                target_engine_path,
                n_games,
            ],
        ).start()
    threading.Thread(target=generate_progress, args=[target_engine_path, games]).start()

    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()

    book_train_engine.extra_save(path="book.json", verbose=False)
    book_train_engine.quit(verbose=False)

    wins = game_results[target_engine_path]["wins"]
    loses = game_results[target_engine_path]["loses"]
    draw = game_results[target_engine_path]["draw"]
    name = target_engine_path2name[target_engine_path]
    print(f"{name}, wins={wins}, loses={loses}, draw={draw}")
    rate = (wins + draw * 0.5) / (
        wins + loses + draw if (wins + loses + draw) > 0 else 1
    )
    print(f"train_engine win_rate: {rate}")

    date = datetime.datetime.today()
    file_name = f"sfen_{date.year:04}{date.month:02}{date.day:02}{date.hour:02}{date.minute:02}.pkl"
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/{file_name}", "wb") as f:
        pickle.dump(game_sfen, f)


if __name__ == "__main__":
    # 学習対象のエンジンパス
    train_engine_path = "path"
    # 仮想敵のエンジンパス（単一）
    target_engine_path = "path"
    parallel = 24  # 並列数
    games = 5000  # 合計対局数
    save_path = "./kifu"  # 生成棋譜を保存するフォルダ

    main(
        train_engine_path=train_engine_path,
        target_engine_path=target_engine_path,
        parallel=parallel,
        games=games,
        save_path=save_path,
    )
