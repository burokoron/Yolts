import json
import os
import pickle
import typing

import cshogi
import numpy as np
from sklearn import linear_model, metrics
from tqdm import tqdm


def sfen2features(
    sfen: str, value: int, matting_value: int
) -> typing.Tuple[typing.List[int], int]:
    board: typing.Any = cshogi.Board(sfen)
    bp, wp = board.pieces_in_hand
    pieces = board.pieces
    features: typing.List[int] = []
    for i in range(len(pieces)):
        if pieces[i] == 5:
            pieces[i] = 6
        elif pieces[i] == 6:
            pieces[i] = 7
        elif pieces[i] == 7:
            pieces[i] = 5
        elif pieces[i] == 21:
            pieces[i] = 22
        elif pieces[i] == 22:
            pieces[i] = 23
        elif pieces[i] == 23:
            pieces[i] = 21

        if pieces[i] == 0:
            features += [0] * 30
        else:
            tmp = [0] * 30
            tmp[pieces[i] - 1] = 1
            features += tmp
    for i in range(len(bp)):
        if i == 0:
            if bp[i] == 0:
                features += [0] * 18
            else:
                tmp = [0] * 18
                tmp[bp[i] - 1] = 1
                features += tmp
        elif 1 <= i <= 4:
            if bp[i] == 0:
                features += [0] * 4
            else:
                tmp = [0] * 4
                tmp[bp[i] - 1] = 1
                features += tmp
        elif 5 <= i:
            if bp[i] == 0:
                features += [0] * 2
            else:
                tmp = [0] * 2
                tmp[bp[i] - 1] = 1
                features += tmp
    for i in range(len(wp)):
        if i == 0:
            if wp[i] == 0:
                features += [0] * 18
            else:
                tmp = [0] * 18
                tmp[wp[i] - 1] = 1
                features += tmp
        elif 1 <= i <= 4:
            if wp[i] == 0:
                features += [0] * 4
            else:
                tmp = [0] * 4
                tmp[wp[i] - 1] = 1
                features += tmp
        elif 5 <= i:
            if wp[i] == 0:
                features += [0] * 2
            else:
                tmp = [0] * 2
                tmp[wp[i] - 1] = 1
                features += tmp

    value = min([max([value, -matting_value]), matting_value])

    return features, value


def sfen2features_reverse(
    sfen: str, value: int, matting_value: int
) -> typing.Tuple[typing.List[int], int]:
    board: typing.Any = cshogi.Board(sfen)
    bp, wp = board.pieces_in_hand
    pieces = board.pieces
    features: typing.List[int] = []
    for i in range(len(pieces) - 1, -1, -1):
        if pieces[i] == 5:
            pieces[i] = 6
        elif pieces[i] == 6:
            pieces[i] = 7
        elif pieces[i] == 7:
            pieces[i] = 5
        elif pieces[i] == 21:
            pieces[i] = 22
        elif pieces[i] == 22:
            pieces[i] = 23
        elif pieces[i] == 23:
            pieces[i] = 21

        if 1 <= pieces[i] <= 14:
            pieces[i] += 16
        elif 17 <= pieces[i]:
            pieces[i] -= 16

        if pieces[i] == 0:
            features += [0] * 30
        else:
            tmp = [0] * 30
            tmp[pieces[i] - 1] = 1
            features += tmp
    for i in range(len(wp)):
        if i == 0:
            if wp[i] == 0:
                features += [0] * 18
            else:
                tmp = [0] * 18
                tmp[wp[i] - 1] = 1
                features += tmp
        elif 1 <= i <= 4:
            if wp[i] == 0:
                features += [0] * 4
            else:
                tmp = [0] * 4
                tmp[wp[i] - 1] = 1
                features += tmp
        elif 5 <= i:
            if wp[i] == 0:
                features += [0] * 2
            else:
                tmp = [0] * 2
                tmp[wp[i] - 1] = 1
                features += tmp
    for i in range(len(bp)):
        if i == 0:
            if bp[i] == 0:
                features += [0] * 18
            else:
                tmp = [0] * 18
                tmp[bp[i] - 1] = 1
                features += tmp
        elif 1 <= i <= 4:
            if bp[i] == 0:
                features += [0] * 4
            else:
                tmp = [0] * 4
                tmp[bp[i] - 1] = 1
                features += tmp
        elif 5 <= i:
            if bp[i] == 0:
                features += [0] * 2
            else:
                tmp = [0] * 2
                tmp[bp[i] - 1] = 1
                features += tmp

    value = min([max([-value, -matting_value]), matting_value])

    return features, value


def main(root_path: str, train_ratio: float, matting_value: int) -> None:
    x_train: typing.Any = []
    y_train: typing.Any = []
    x_test: typing.Any = []
    y_test: typing.Any = []
    same_sfen: typing.Set[str] = set()

    file_list = [file for file in os.listdir(root_path) if file.split(".")[-1] == "pkl"]
    for i, file in enumerate(file_list):
        if file.split(".")[-1] != "pkl":
            continue
        with open(file, "rb") as fb:
            kifu_dict = pickle.load(fb)
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
                    features, value = sfen2features(
                        sfen=sfen, value=value, matting_value=matting_value
                    )
                    if int(key) % 9 == 0:
                        x_test.append(features)
                        y_test.append(value)
                    elif i < int(len(file_list) * train_ratio):
                        x_train.append(features)
                        y_train.append(value)
                    features, value = sfen2features_reverse(
                        sfen=sfen, value=value, matting_value=matting_value
                    )
                    if int(key) % 9 == 0:
                        x_test.append(features)
                        y_test.append(value)
                    elif i < int(len(file_list) * train_ratio):
                        x_train.append(features)
                        y_train.append(value)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print(x_train.shape)
    print(x_test.shape)

    alpha_list = [0.00005, 0.000025, 0.00001, 0.0000075, 0.000005]
    max_iter_list = [1000]
    score_list: typing.List[str] = []
    for alpha in alpha_list:
        for max_iter in max_iter_list:
            clf = linear_model.SGDRegressor(
                alpha=alpha,
                fit_intercept=False,
                max_iter=max_iter,
                verbose=1,
                random_state=42,
                n_iter_no_change=max_iter,
            )
            clf.fit(X=x_train, y=y_train)

            y_predict: typing.Any = clf.predict(X=x_test)
            score = metrics.mean_squared_error(y_true=y_test, y_pred=y_predict)
            score_list.append(f"alpha = {alpha}, max_iter = {max_iter}, MSE = {score}")

            assert clf.coef_ is not None

            pieces_dict = {}
            pieces_in_hand_dict = {}
            sq = 0
            piece = 0
            for param in clf.coef_:
                if sq < 81:
                    if piece == 0:
                        pieces_dict[str(sq)] = {str(piece): 0}
                        piece += 1
                        pieces_dict[str(sq)][str(piece)] = int(param)
                    else:
                        pieces_dict[str(sq)][str(piece)] = int(param)
                    piece += 1
                    if piece > 30:
                        sq += 1
                        piece = 0
                elif sq == 81 or sq == 88:
                    if piece == 0:
                        pieces_in_hand_dict[str(sq - 81)] = {str(piece): 0}
                        piece += 1
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    else:
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    piece += 1
                    if piece > 18:
                        sq += 1
                        piece = 0
                elif 82 <= sq <= 85 or 89 <= sq <= 92:
                    if piece == 0:
                        pieces_in_hand_dict[str(sq - 81)] = {str(piece): 0}
                        piece += 1
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    else:
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    piece += 1
                    if piece > 4:
                        sq += 1
                        piece = 0
                elif 86 <= sq <= 87 or 93 <= sq <= 94:
                    if piece == 0:
                        pieces_in_hand_dict[str(sq - 81)] = {str(piece): 0}
                        piece += 1
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    else:
                        pieces_in_hand_dict[str(sq - 81)][str(piece)] = int(param)
                    piece += 1
                    if piece > 2:
                        sq += 1
                        piece = 0

            # 評価パラメータを保存
            eval_dict: typing.Any = {
                "pieces_dict": pieces_dict,
                "pieces_in_hand_dict": pieces_in_hand_dict,
            }
            with open(f"eval_{alpha}_{max_iter}_{score}.json", "w") as ft:
                json.dump(eval_dict, ft)


if __name__ == "__main__":
    root_path = "./"  # 学習棋譜があるルートフォルダ
    matting_value = 6842  # 勝ち(負け)を読み切ったときの評価値
    train_ratio = 0.9  # 学習に使用する棋譜ファイルの割合

    main(root_path=root_path, train_ratio=train_ratio, matting_value=matting_value)
