import json
import os
import pickle
import shutil
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

        bking_pos = bking_pos // 27 * 9 + bking_pos % 9
        wking_pos = wking_pos // 27 * 9 + wking_pos % 9
        for i in range(len(pieces)):
            if pieces[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][i][pieces[i]]
                )
        for i in range(len(bp)):
            if bp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][81 + i][bp[i]]
                )
        for i in range(len(wp)):
            if wp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][88 + i][wp[i]]
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

        bking_pos = bking_pos // 27 * 9 + bking_pos % 9
        wking_pos = wking_pos // 27 * 9 + wking_pos % 9
        for i in range(len(pieces) - 1, -1, -1):
            if pieces[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][i][pieces[i]]
                )
        for i in range(len(wp)):
            if wp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][81 + i][wp[i]]
                )
        for i in range(len(bp)):
            if bp[i] == 0:
                features.append(0)
            else:
                features.append(
                    self.features_table[0][bking_pos][wking_pos][88 + i][bp[i]]
                )

        value = min([max([-value, -matting_value]), matting_value])

        return features, value

    def flip_features(self, features: typing.List[int]) -> typing.List[int]:
        outputs: typing.List[int] = []

        for i in range(72, -1, -9):
            for j in range(9):
                outputs.append(features[i + j])
        for i in range(81, 95):
            outputs.append(features[i])

        return features


class TrainSequence(keras.utils.Sequence):  # type:ignore
    def __init__(
        self,
        root_path: str,
        train_file_number: int,
        matting_value: int,
        value_scale: int,
    ):
        self.mf = MakeFeatures()
        self.root_path = root_path
        self.train_file_number = train_file_number
        self.matting_value = matting_value
        self.value_scale = value_scale

    def __len__(self) -> int:
        return self.train_file_number

    def __getitem__(self, idx: int) -> typing.Any:

        batch_x: typing.Any = []
        batch_y: typing.Any = []

        with open(f"{self.root_path}/train_{idx}.pkl", "rb") as f:
            x_train, y_train = pickle.load(f)
        for (sfen, value) in zip(x_train, y_train):
            x, y = self.mf.sfen2features(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append(y)

            x = self.mf.flip_features(x)
            batch_x.append(x)
            batch_y.append(y)

            x, y = self.mf.sfen2features_reverse(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append(y)

            x = self.mf.flip_features(x)
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y) / self.value_scale


class ValidationSequence(keras.utils.Sequence):  # type:ignore
    def __init__(
        self,
        root_path: str,
        test_file_number: int,
        matting_value: int,
        value_scale: int,
    ):
        self.mf = MakeFeatures()
        self.root_path = root_path
        self.test_file_number = test_file_number
        self.matting_value = matting_value
        self.value_scale = value_scale

    def __len__(self) -> int:
        return self.test_file_number

    def __getitem__(self, idx: int) -> typing.Any:

        batch_x: typing.Any = []
        batch_y: typing.Any = []

        with open(f"{self.root_path}/test_{idx}.pkl", "rb") as f:
            x_test, y_test = pickle.load(f)
        for (sfen, value) in zip(x_test, y_test):
            x, y = self.mf.sfen2features(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append(y)

            x = self.mf.flip_features(x)
            batch_x.append(x)
            batch_y.append(y)

            x, y = self.mf.sfen2features_reverse(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append(y)

            x = self.mf.flip_features(x)
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y) / self.value_scale


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
    root_path: str,
    tmp_path: str,
    train_ratio: float,
    matting_value: int,
    value_scale: int,
    batch_size: int,
) -> None:
    x_train: typing.Any = []
    y_train: typing.Any = []
    x_test: typing.Any = []
    y_test: typing.Any = []
    same_sfen: typing.Set[str] = set()

    file_list = [file for file in os.listdir(root_path) if file.split(".")[-1] == "pkl"]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_file_number = 0
    test_file_number = 0
    os.mkdir(tmp_path)
    for i, file in enumerate(tqdm(file_list, desc="file")):
        if file.split(".")[-1] != "pkl":
            continue
        with open(f"{root_path}/{file}", "rb") as f:
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
                    if int(key) % 10 == 0:
                        x_test.append(sfen)
                        y_test.append(value)
                    elif i < int(len(file_list) * train_ratio):
                        x_train.append(sfen)
                        y_train.append(value)
                    # バッチサイズ分データが溜まったらファイル保存する
                    if len(x_train) == batch_size // 4:
                        with open(
                            f"{tmp_path}/train_{train_file_number}.pkl", "wb"
                        ) as f:
                            pickle.dump((x_train, y_train), f)
                        x_train = []
                        y_train = []
                        train_file_number += 1
                    if len(x_test) == batch_size // 4:
                        with open(f"{tmp_path}/test_{test_file_number}.pkl", "wb") as f:
                            pickle.dump((x_test, y_test), f)
                        x_test = []
                        y_test = []
                        test_file_number += 1
    with open(f"{tmp_path}/train_{train_file_number}.pkl", "wb") as f:
        pickle.dump((x_train, y_train), f)
    with open(f"{tmp_path}/test_{test_file_number}.pkl", "wb") as f:
        pickle.dump((x_test, y_test), f)

    train_generator = TrainSequence(
        root_path=tmp_path,
        train_file_number=train_file_number + 1,
        matting_value=matting_value,
        value_scale=value_scale,
    )
    validation_generator = ValidationSequence(
        root_path=tmp_path,
        test_file_number=test_file_number + 1,
        matting_value=matting_value,
        value_scale=value_scale,
    )

    print(train_file_number * batch_size + len(x_train) * 4)
    print(test_file_number * batch_size + len(x_test) * 4)

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
        train_generator,
        epochs=1000,
        validation_data=validation_generator,
        callbacks=[reduce_lr, early_stop],
        max_queue_size=24,
        workers=12,
        use_multiprocessing=True,
    )
    model.save("mlp")

    # jsonで保存
    embedding = model.layers[1]
    weights = embedding.get_weights()[0][:, 0]
    params = [float(weight) for weight in weights]

    eval_json: typing.TextIO = open("eval.json", "w")
    json.dump({"params": params}, eval_json)

    # 一時保存用フォルダの削除
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    root_path = "./kifu"  # 学習棋譜があるルートフォルダ
    tmp_path = "./tmp"  # 一時保存データ用のフォルダ
    matting_value = 19089  # 勝ち(負け)を読み切ったときの評価値
    value_scale = 512  # 学習時の評価値スケーリングパラメータ
    train_ratio = 0.9  # 学習に使用する棋譜ファイルの割合
    batch_size = 65536  # バッチサイズ

    main(
        root_path=root_path,
        tmp_path=tmp_path,
        train_ratio=train_ratio,
        matting_value=matting_value,
        value_scale=value_scale,
        batch_size=batch_size,
    )
