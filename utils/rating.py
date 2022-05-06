"""
イロ・レーティングの計算シミュレーション
"""

import random
import typing


def update(
    my_elo_rating: float,
    fixed_rating_players: typing.Dict[str, typing.List[float]],
    iteration: int,
) -> typing.Tuple[float, float]:
    min_rating = my_elo_rating
    max_reting = my_elo_rating
    for _ in range(iteration):
        for i in range(len(fixed_rating_players["elo_rating"])):
            w = 1 / (
                10 ** (abs(fixed_rating_players["elo_rating"][i] - my_elo_rating) / 400)
                + 1
            )
            rand = random.random()
            if rand < fixed_rating_players["win_rate"][i]:
                if my_elo_rating < fixed_rating_players["elo_rating"][i]:
                    w = 1 - w
                my_elo_rating += 1 * w
                if my_elo_rating > max_reting:
                    max_reting = my_elo_rating
            else:
                if my_elo_rating > fixed_rating_players["elo_rating"][i]:
                    w = 1 - w
                my_elo_rating -= 1 * w
                if my_elo_rating < min_rating:
                    min_rating = my_elo_rating

    return min_rating, max_reting


if __name__ == "__main__":
    """
    QRL Software rating基準のレーティングを計算する
    """

    # QRLのついた基準ソフトとレーティング計算したソフトの設定
    player: typing.Dict[str, typing.Dict[str, typing.Any]] = {
        "RandamKun V 1.0.0": {
            "opponent": {
                "十六式いろは SDT4": 0.528,
                "GSE 0.1.6": 0.014,
                "LesserKai": 0.0,
            },
            "rating": 112,
        },
        "十六式いろは SDT4": {
            "opponent": {},
            "rating": 196,
        },
        "爆裂駒捨太郎 V 1.0.0": {
            "opponent": {
                "RandamKun V 1.0.0": 0.936,
                "十六式いろは SDT4": 0.871,
                "GSE 0.1.6": 0.008,
                "LesserKai": 0.001,
            },
            "rating": 308,
        },
        "爆裂駒捨太郎 V 1.1.0": {
            "opponent": {
                "RandamKun V 1.0.0": 0.999,
                "十六式いろは SDT4": 0.997,
                "爆裂駒捨太郎 V 1.0.0": 0.942,
                "GSE 0.1.6": 0.012,
                "LesserKai": 0.0,
                "Bonanza 6.0 D1": 0,
            },
            "rating": 425,
        },
        "GSE 0.1.6": {
            "opponent": {},
            "rating": 426,
        },
        "爆裂駒捨太郎 V 2.0.0": {
            "opponent": {
                "RandamKun V 1.0.0": 1.0,
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 0.981,
                "爆裂駒捨太郎 V 1.1.0": 0.949,
                "GSE 0.1.6": 0.019,
                "LesserKai": 0.004,
                "Bonanza 6.0 D1": 0.003,
            },
            "rating": 504,
        },
        "爆裂駒捨太郎 V 2.1.0": {
            "opponent": {
                "RandamKun V 1.0.0": 1.0,
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 0.998,
                "爆裂駒捨太郎 V 1.1.0": 0.994,
                "GSE 0.1.6": 0.033,
                "爆裂駒捨太郎 V 2.0.0": 0.979,
                "LesserKai": 0.015,
                "Bonanza 6.0 D1": 0.004,
                "Bonanza 6.0 D2": 0.0,
            },
            "rating": 584,
        },
        "爆裂駒捨太郎 V 2.2.0": {
            "opponent": {
                "RandamKun V 1.0.0": 1.0,
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 0.999,
                "爆裂駒捨太郎 V 1.1.0": 1.0,
                "GSE 0.1.6": 0.06,
                "爆裂駒捨太郎 V 2.0.0": 0.971,
                "爆裂駒捨太郎 V 2.1.0": 0.931,
                "LesserKai": 0.007,
                "Bonanza 6.0 D1": 0.011,
                "Bonanza 6.0 D2": 0.0,
            },
            "rating": 639,
        },
        "爆裂駒捨太郎 V 2.3.0": {
            "opponent": {
                "RandamKun V 1.0.0": 0.997,
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 1.0,
                "爆裂駒捨太郎 V 1.1.0": 0.996,
                "GSE 0.1.6": 0.133,
                "爆裂駒捨太郎 V 2.0.0": 0.998,
                "爆裂駒捨太郎 V 2.1.0": 0.972,
                "爆裂駒捨太郎 V 2.2.0": 0.86,
                "LesserKai": 0.04,
                "Bonanza 6.0 D1": 0.012,
                "Bonanza 6.0 D2": 0.002,
            },
            "rating": 701,
        },
        "爆裂駒捨太郎R V 1.0.0": {
            "opponent": {
                "RandamKun V 1.0.0": 1.0,
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 1.0,
                "爆裂駒捨太郎 V 1.1.0": 1.0,
                "GSE 0.1.6": 0.216,
                "爆裂駒捨太郎 V 2.0.0": 1.0,
                "爆裂駒捨太郎 V 2.1.0": 0.989,
                "爆裂駒捨太郎 V 2.2.0": 0.89,
                "爆裂駒捨太郎 V 2.3.0": 0.67,
                "LesserKai": 0.116,
                "Bonanza 6.0 D1": 0.023,
                "Bonanza 6.0 D2": 0.015,
                "技巧2 D1": 0.005,
            },
            "rating": 744,
        },
        "LesserKai": {
            "opponent": {},
            "rating": 772,
        },
        "Bonanza 6.0 D1": {
            "opponent": {},
            "rating": 1101,
        },
        "Bonanza 6.0 D2": {
            "opponent": {},
            "rating": 1325,
        },
        "技巧2 D1": {
            "opponent": {},
            "rating": 1544,
        },
    }

    # イテレーション
    iteration = 1000000

    for target in player.keys():
        if player[target]["rating"] != 0:
            continue
        # 基準エンジンのレーティングとそれに対する勝率
        fixed_rating_players: typing.Dict[str, typing.List[float]] = {
            "elo_rating": [],
            "win_rate": [],
        }
        for opponent in player[target]["opponent"].keys():
            fixed_rating_players["elo_rating"].append(player[opponent]["rating"])
            fixed_rating_players["win_rate"].append(
                player[target]["opponent"][opponent]
            )
        if len(fixed_rating_players["elo_rating"]) == 0:
            continue
        my_elo_rating = min(fixed_rating_players["elo_rating"])
        assert my_elo_rating != 0, f"my_elo_rating == 0\n{fixed_rating_players}"

        print(f"Calculating ratings for {target}...")
        c = my_elo_rating
        while c > 1:
            a, b = update(
                my_elo_rating=my_elo_rating,
                fixed_rating_players=fixed_rating_players,
                iteration=iteration,
            )
            c = abs(my_elo_rating - (a + b) / 2)
            my_elo_rating = (a + b) / 2

            player[target]["rating"] = int(my_elo_rating)
            for _target in player.keys():
                print(f"{_target}: {player[_target]['rating']}")
            print()

    for target in player.keys():
        print(f"{target}: {player[target]['rating']}")
