"""
爆裂駒捨太郎の評価関数の学習
"""

import argparse
import os
import json
import typing

import cshogi
import numpy as np
from tqdm import tqdm


def main(path: str) -> None:
    file_list = os.listdir(path)

    # 先後の局面数をカウントし、重みを計算
    turn_weight = np.zeros(2)
    for file in tqdm(file_list, desc="file"):
        with open(f"{path}/{file}") as f:
            position_list = f.read().split("\n")
        for position in tqdm(position_list, desc="position"):
            board = cshogi.Board(position)
            turn_weight[board.turn] += 1

    turn_weight /= np.mean(turn_weight)
    turn_weight = 1.0 / turn_weight

    # 1駒関係において情報量の多いパターンは価値があるとした評価パラメータの計算
    # 持ち駒評価パラメータ計算用
    pieces_in_hand_dict: typing.Dict[
        int, typing.Dict[int, typing.Union[int, float]]
    ] = {
        0: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
        },
        1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        5: {0: 0, 1: 0, 2: 0},
        6: {0: 0, 1: 0, 2: 0},
        7: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
        },
        8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        10: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        11: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        12: {0: 0, 1: 0, 2: 0},
        13: {0: 0, 1: 0, 2: 0},
    }
    pieces_in_hand_total_dict: typing.Dict[
        int, typing.Dict[int, typing.Union[int, float]]
    ] = {
        0: {
            0: 1.0,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 1,
            17: 1,
            18: 1,
        },
        1: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        2: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        3: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        4: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        5: {0: 1, 1: 1, 2: 1},
        6: {0: 1, 1: 1, 2: 1},
        7: {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 1,
            17: 1,
            18: 1,
        },
        8: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        9: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        10: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        11: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        12: {0: 1, 1: 1, 2: 1},
        13: {0: 1, 1: 1, 2: 1},
    }
    # 盤面の評価パラメータ計算用
    pieces_dict: typing.Dict[int, typing.Dict[int, typing.Union[int, float]]] = {}
    pieces_total_dict: typing.Dict[int, typing.Dict[int, typing.Union[int, float]]] = {}
    for i in range(81):
        pieces_dict[i] = {}
        pieces_total_dict[i] = {}
        for j in range(31):
            pieces_dict[i][j] = 0
            pieces_total_dict[i][j] = 1

    # 詰将棋局面の駒種&位置&勝ち負け出現数カウント
    for file in tqdm(file_list, desc="file"):
        with open(f"{path}/{file}") as f:
            position_list = f.read().split("\n")
        for position in tqdm(position_list, desc="position"):
            board = cshogi.Board(position)
            bp, bw = board.pieces_in_hand
            ap = bp + bw
            for i, piece in enumerate(ap):
                if board.turn == 0:
                    pieces_in_hand_dict[i][piece] += turn_weight[0]
                else:
                    pieces_in_hand_dict[i][piece] -= turn_weight[1]
                pieces_in_hand_total_dict[i][piece] += 1
            pieces = board.pieces
            for i, piece in enumerate(pieces):
                if board.turn == 0:
                    pieces_dict[i][piece] += turn_weight[0]
                else:
                    pieces_dict[i][piece] -= turn_weight[1]
                pieces_total_dict[i][piece] += 1

    # 出現頻度で正規化
    max_num: typing.Union[int, float] = 0
    pieces_in_hand_features = 0
    pieces_features = 0
    for i in pieces_in_hand_total_dict.keys():
        pieces_in_hand_features += 1
        for j in pieces_in_hand_total_dict[i].keys():
            if max_num < pieces_in_hand_total_dict[i][j]:
                max_num = pieces_in_hand_total_dict[i][j]
    for i in pieces_total_dict.keys():
        pieces_features += 1
        for j in pieces_total_dict[i].keys():
            if max_num < pieces_total_dict[i][j]:
                max_num = pieces_total_dict[i][j]

    for i in pieces_in_hand_total_dict.keys():
        for j in pieces_in_hand_total_dict[i].keys():
            pieces_in_hand_total_dict[i][j] /= max_num
    for i in pieces_total_dict.keys():
        for j in pieces_total_dict[i].keys():
            pieces_total_dict[i][j] /= max_num

    # 情報量に基づいて評価値に変換
    for i in pieces_in_hand_total_dict.keys():
        for j in pieces_in_hand_total_dict[i].keys():
            pieces_in_hand_total_dict[i][j] = -np.log2(pieces_in_hand_total_dict[i][j])
    for i in pieces_total_dict.keys():
        for j in pieces_total_dict[i].keys():
            pieces_total_dict[i][j] = -np.log2(pieces_total_dict[i][j])

    # 先手勝ちなら評価値プラス、負けならマイナス
    # いい感じの整数値になるように×100する
    for i in pieces_in_hand_dict.keys():
        for j in pieces_in_hand_dict[i].keys():
            if pieces_in_hand_dict[i][j] != 0:
                if pieces_in_hand_dict[i][j] < 0:
                    pieces_in_hand_dict[i][j] = -int(
                        pieces_in_hand_total_dict[i][j] * 100
                    )
                else:
                    pieces_in_hand_dict[i][j] = int(
                        pieces_in_hand_total_dict[i][j] * 100
                    )
    for i in pieces_dict.keys():
        for j in pieces_dict[i].keys():
            if pieces_dict[i][j] != 0:
                if pieces_dict[i][j] < 0:
                    pieces_dict[i][j] = -int(
                        pieces_total_dict[i][j]
                        * 100
                        * pieces_in_hand_features
                        / pieces_features
                    )
                else:
                    pieces_dict[i][j] = int(
                        pieces_total_dict[i][j]
                        * 100
                        * pieces_in_hand_features
                        / pieces_features
                    )

    # 評価パラメータを保存
    eval_dict = {"pieces_dict": pieces_dict, "pieces_in_hand_dict": pieces_in_hand_dict}
    with open("eval.json", "w") as f:
        json.dump(eval_dict, f)


if __name__ == "__main__":
    """
    Args:
        2コマンド目:str
            学習局面が保存されているフォルダパス
    """

    parser = argparse.ArgumentParser(description="爆裂駒捨太郎Rの学習")
    parser.add_argument("folder_path", help="学習局面が保存されているフォルダパス")
    args = parser.parse_args()

    main(args.folder_path)
