import json
import os
import pickle
import typing

import cshogi
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm


class MakeFeatures:
    def __init__(self) -> None:
        self.features_table = np.zeros((2, 81, 81, 95, 31), dtype=np.uint32)
        idx = 0
        for i in range(2):
            for j in range(81):
                for k in range(81):
                    for m in range(95):
                        for n in range(31):
                            self.features_table[i][j][k][m][n] = idx
                            idx += 1

    def sfen2features(
        self, sfen: str, value: int, matting_value: int
    ) -> typing.Tuple[typing.List[int], int]:
        board: typing.Any = cshogi.Board(sfen)
        bp, wp = board.pieces_in_hand
        pieces = board.pieces
        features: typing.List[int] = []
        bking_pos = None
        wking_pos = None
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

            if pieces[i] == 8:
                bking_pos = i
            elif pieces[i] == 24:
                wking_pos = i

        assert bking_pos is not None
        assert wking_pos is not None
        for i in range(len(pieces)):
            if pieces[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][i][
                        pieces[i]
                    ]
                )
        for i in range(len(bp)):
            if bp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][
                        81 + i
                    ][bp[i]]
                )
        for i in range(len(wp)):
            if wp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][
                        88 + i
                    ][wp[i]]
                )

        value = min([max([value, -matting_value]), matting_value])

        return features, value

    def sfen2features_reverse(
        self, sfen: str, value: int, matting_value: int
    ) -> typing.Tuple[typing.List[int], int]:
        board: typing.Any = cshogi.Board(sfen)
        bp, wp = board.pieces_in_hand
        pieces = board.pieces
        features: typing.List[int] = []
        bking_pos = None
        wking_pos = None
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

            if pieces[i] == 8:
                bking_pos = i
            elif pieces[i] == 24:
                wking_pos = i

        assert bking_pos is not None
        assert wking_pos is not None
        for i in range(len(pieces) - 1, -1, -1):
            if pieces[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][i][
                        pieces[i]
                    ]
                )
        for i in range(len(wp)):
            if wp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][
                        81 + i
                    ][wp[i]]
                )
        for i in range(len(bp)):
            if bp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos % 9][wking_pos % 9][
                        88 + i
                    ][bp[i]]
                )

        value = min([max([-value, -matting_value]), matting_value])

        return features, value


def mlp() -> keras.models.Model:
    inputs = keras.layers.Input(shape=95, name="inputs")
    x = keras.layers.Embedding(
        input_dim=38644290,
        output_dim=1,
        embeddings_initializer=tf.keras.initializers.Zeros(),
        mask_zero=True,
        input_length=95,
    )(inputs)
    outputs = tf.math.reduce_sum(input_tensor=x, axis=1)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def main(
    root_path: str, train_ratio: float, matting_value: int, value_scale: int
) -> None:
    x_train: typing.Any = []
    y_train: typing.Any = []
    x_test: typing.Any = []
    y_test: typing.Any = []
    same_sfen: typing.Set[str] = set()

    file_list = [file for file in os.listdir(root_path) if file.split(".")[-1] == "pkl"]
    mf = MakeFeatures()
    for i, file in enumerate(tqdm(file_list, desc="file")):
        if file.split(".")[-1] != "pkl":
            continue
        with open(file, "rb") as f:
            kifu_dict = pickle.load(f)
        for key in tqdm(kifu_dict, desc="kifu", leave=False):
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
                    features, value = mf.sfen2features(
                        sfen=sfen, value=value, matting_value=matting_value
                    )
                    if int(key) % 9 == 0:
                        x_test.append(features)
                        y_test.append(value)
                    elif i < int(len(file_list) * train_ratio):
                        x_train.append(features)
                        y_train.append(value)
                    features, value = mf.sfen2features_reverse(
                        sfen=sfen, value=value, matting_value=matting_value
                    )
                    if int(key) % 9 == 0:
                        x_test.append(features)
                        y_test.append(value)
                    elif i < int(len(file_list) * train_ratio):
                        x_train.append(features)
                        y_train.append(value)

    x_train = np.array(x_train)
    y_train = np.array(y_train) / value_scale
    x_test = np.array(x_test)
    y_test = np.array(y_test) / value_scale
    print(x_train.shape)
    print(x_test.shape)

    # 学習
    model = mlp()
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanAbsoluteError()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=1, verbose=1, min_lr=0.0
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=65535,
        epochs=1000,
        validation_data=(x_test, y_test),
        shuffle=False,
        callbacks=[reduce_lr, early_stop],
    )
    model.save("mlp")

    # jsonで保存
    embedding = model.layers[1]
    weights = embedding.get_weights()[0][:, 0]
    params = [float(weight) for weight in weights]

    eval_json: typing.TextIO = open("eval.json", "w")
    json.dump({"params": params}, eval_json)


if __name__ == "__main__":
    root_path = "./"  # 学習棋譜があるルートフォルダ
    matting_value = 13544  # 勝ち(負け)を読み切ったときの評価値
    value_scale = 512  # 学習時の評価値スケーリングパラメータ
    train_ratio = 0.9  # 学習に使用する棋譜ファイルの割合

    main(
        root_path=root_path,
        train_ratio=train_ratio,
        matting_value=matting_value,
        value_scale=value_scale,
    )
