import json
import os
import pickle
import shutil
import time
import typing

import cshogi
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm


class MakeFeatures:
    def __init__(self) -> None:
        self.features_table = np.zeros((2, 95, 31, 95, 31), dtype=np.uint32)
        idx = 0
        for i in range(2):
            for j in range(95):
                for k in range(31):
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

        for i, pi in enumerate(pieces):
            for j, pj in enumerate(pieces):
                if i > j:
                    continue
                if pi == 0 or pj == 0:
                    features.append(0)
                else:
                    features.append(self.features_table[0][i][pi][j][pj])
        for i, p in enumerate(bp):
            if p == 0:
                features.append(0)
            else:
                features.append(self.features_table[0][0][0][81 + i][p])
        for i, p in enumerate(wp):
            if p == 0:
                features.append(0)
            else:
                features.append(self.features_table[0][0][0][88 + i][p])

        value = min([max([value, -matting_value]), matting_value])

        return features, value

    def sfen2features_reverse(
        self, sfen: str, value: int, matting_value: int
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

        pieces_rev = list(reversed(pieces))
        for i, pi in enumerate(pieces_rev):
            for j, pj in enumerate(pieces_rev):
                if i > j:
                    continue
                if pi == 0 or pj == 0:
                    features.append(0)
                else:
                    features.append(self.features_table[0][i][pi][j][pj])
        for i, p in enumerate(wp):
            if p == 0:
                features.append(0)
            else:
                features.append(self.features_table[0][0][0][81 + i][p])
        for i, p in enumerate(bp):
            if p == 0:
                features.append(0)
            else:
                features.append(self.features_table[0][0][0][88 + i][p])

        value = min([max([-value, -matting_value]), matting_value])

        return features, value


class TrainDataset(Dataset):
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
        for sfen, value in zip(x_train, y_train):
            x, y = self.mf.sfen2features(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append([y])

            x, y = self.mf.sfen2features_reverse(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append([y])

        return (
            torch.tensor(batch_x, dtype=torch.int32),
            torch.tensor(batch_y) / self.value_scale,
        )


class ValidationDataset(Dataset):
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
        for sfen, value in zip(x_test, y_test):
            x, y = self.mf.sfen2features(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append([y])

            x, y = self.mf.sfen2features_reverse(
                sfen=sfen, value=value, matting_value=self.matting_value
            )
            batch_x.append(x)
            batch_y.append([y])

        return (
            torch.tensor(batch_x, dtype=torch.int32),
            torch.tensor(batch_y) / self.value_scale,
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(inputs)
        outputs = torch.sum(input=x, dim=1)
        return outputs


def main(
    root_path: str,
    tmp_path: str,
    checkpoint_path: str,
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
    # """
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
                    if len(x_train) == batch_size // 2:
                        with open(
                            f"{tmp_path}/train_{train_file_number}.pkl", "wb"
                        ) as f:
                            pickle.dump((x_train, y_train), f)
                        x_train = []
                        y_train = []
                        train_file_number += 1
                    if len(x_test) == batch_size // 2:
                        with open(f"{tmp_path}/test_{test_file_number}.pkl", "wb") as f:
                            pickle.dump((x_test, y_test), f)
                        x_test = []
                        y_test = []
                        test_file_number += 1
    with open(f"{tmp_path}/train_{train_file_number}.pkl", "wb") as f:
        pickle.dump((x_train, y_train), f)
    with open(f"{tmp_path}/test_{test_file_number}.pkl", "wb") as f:
        pickle.dump((x_test, y_test), f)

    del same_sfen
    # """
    # train_file_number = 6817
    # test_file_number = 1289
    train_dataset = TrainDataset(
        root_path=tmp_path,
        train_file_number=train_file_number + 1,
        matting_value=matting_value,
        value_scale=value_scale,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_dataset = ValidationDataset(
        root_path=tmp_path,
        test_file_number=test_file_number + 1,
        matting_value=matting_value,
        value_scale=value_scale,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    print(train_file_number * batch_size + len(x_train) * 2)
    print(test_file_number * batch_size + len(x_test) * 2)

    # 学習
    model = MLP(input_dim=17346050, output_dim=1)
    model.embedding.weight.data.zero_()
    summary(model, input_data=torch.zeros([1, 3335], dtype=torch.int32))
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.2, patience=0, verbose=True
    )
    # early_stopping用
    best_epoch = 0
    best_validation_loss = 1e9
    early_stopping_threshold = 0.0001
    early_stopping_patience = 1
    os.mkdir(checkpoint_path)
    for epoch in range(1000):
        # 学習
        model.train()
        train_loss = 0.0
        start_time = time.time()
        with tqdm(train_data_loader, leave=False) as pbar:
            for i, data in enumerate(pbar):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.set_description(f"train epoch: {epoch + 1}")
                pbar.set_postfix(ordered_dict={"loss": train_loss / (i + 1)})
        # 評価
        model.eval()
        validation_loss = 0.0
        with tqdm(validation_data_loader, leave=False) as pbar:
            for i, data in enumerate(pbar):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                pbar.set_description(f"validation epoch: {epoch + 1}")
                pbar.set_postfix(ordered_dict={"loss": validation_loss / (i + 1)})
        end_time = time.time()
        elapsed_time = end_time - start_time
        train_loss /= len(train_data_loader)
        validation_loss /= len(validation_data_loader)
        print(f"epoch: {epoch + 1}, train loss: {train_loss}, ", end="")
        print(f"validation loss: {validation_loss}, elapsed time: {elapsed_time}s")
        torch.save(
            model.state_dict(), f"{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pt"
        )
        scheduler.step(validation_loss)
        if validation_loss + early_stopping_threshold < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
        elif best_epoch + early_stopping_patience < epoch:
            print("Early Stopping!!")
            break
    torch.save(model.state_dict(), "model.pt")

    # jsonで保存
    weights = model.embedding.weight.cpu().detach().numpy()[:, 0]
    params = [float(weight) for weight in weights]
    eval_json: typing.TextIO = open("eval.json", "w")
    json.dump({"params": params}, eval_json)

    # 一時保存用フォルダの削除
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    root_path = "./kifu"  # 学習棋譜があるルートフォルダ
    tmp_path = "./tmp"  # 一時保存データ用のフォルダ
    checkpoint_path = "./checkpoint"  # モデルチェックポイントを保存するフォルダ
    matting_value = 10075  # 勝ち(負け)を読み切ったときの評価値
    value_scale = 512  # 学習時の評価値スケーリングパラメータ
    train_ratio = 0.9  # 学習に使用する棋譜ファイルの割合
    batch_size = 8192  # バッチサイズ

    main(
        root_path=root_path,
        tmp_path=tmp_path,
        checkpoint_path=checkpoint_path,
        train_ratio=train_ratio,
        matting_value=matting_value,
        value_scale=value_scale,
        batch_size=batch_size,
    )
