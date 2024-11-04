import os
from tqdm import tqdm
import pickle
import typing
import matplotlib.pyplot as plt
import numpy as np


def main(root_path: str, limit_value: int) -> None:
    file_list = [file for file in os.listdir(root_path) if file.split(".")[-1] == "pkl"]

    value_win_list = [0] * (limit_value * 2 + 1)
    value_lose_list = [0] * (limit_value * 2 + 1)
    for file in tqdm(file_list, desc="file"):
        with open(f"{root_path}/{file}", "rb") as f:
            kifu_dict = pickle.load(f)
        for key in tqdm(kifu_dict, desc="kifu", leave=False):
            values = kifu_dict[key]["value"]
            winner = kifu_dict[key]["winner"][0]

            my_turn = ""
            for i, value in enumerate(values):
                if value is not None:
                    if i % 2 == 0:
                        my_turn = "b"
                    else:
                        my_turn = "w"

            for value in values:
                if value is not None:
                    if -limit_value <= value <= limit_value:
                        if my_turn == "w":
                            value = -value
                        if winner == "d":
                            value_win_list[value + limit_value] += 0.5
                            value_lose_list[value + limit_value] += 0.5
                        elif winner == my_turn:
                            value_win_list[value + limit_value] += 1
                        else:
                            value_lose_list[value + limit_value] += 1

    x: typing.List[float] = []
    y: typing.List[float] = []
    for i, (win, lose) in enumerate(zip(value_win_list, value_lose_list)):
        if win + lose != 0:
            x.append(i - limit_value)
            win_rate = win / (win + lose)
            y.append((win_rate - 0.5) * 2)

    best_mse = 1e9
    mse = 0
    scale = 1
    while mse < best_mse:
        for i in range(len(x)):
            mse += (np.tanh(x[i] / scale) - y[i]) ** 2
        print(scale, mse)
        if mse < best_mse:
            best_mse = mse
            mse = 0
            scale += 1

    pred_y: typing.List[float] = []
    for i in range(len(x)):
        pred_y.append(np.tanh(x[i] / scale))

    plt.scatter(x, y)
    plt.scatter(x, pred_y)
    plt.show()


if __name__ == "__main__":
    root_path = "./kifu"  # 学習棋譜があるルートフォルダ
    limit_value = 5000  # スケーリング計算の対象とする評価値の下限と上限

    main(root_path=root_path, limit_value=limit_value)
