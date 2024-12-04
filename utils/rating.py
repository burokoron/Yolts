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
            "rating": 127,
        },
        "十六式いろは SDT4": {
            "opponent": {},
            "rating": 215,
        },
        "爆裂駒捨太郎 V 1.0.0": {
            "opponent": {
                "RandamKun V 1.0.0": 0.936,
                "十六式いろは SDT4": 0.871,
                "GSE 0.1.6": 0.008,
                "LesserKai": 0.001,
            },
            "rating": 321,
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
            "rating": 437,
        },
        "GSE 0.1.6": {
            "opponent": {},
            "rating": 437,
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
            "rating": 516,
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
            "rating": 595,
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
            "rating": 650,
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
            "rating": 712,
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
            "rating": 754,
        },
        "LesserKai": {
            "opponent": {},
            "rating": 779,
        },
        "爆裂転生太郎 V 1.0.1": {
            "opponent": {
                "十六式いろは SDT4": 1.0,
                "爆裂駒捨太郎 V 1.0.0": 1.0,
                "爆裂駒捨太郎 V 1.1.0": 1.0,
                "GSE 0.1.6": 0.495,
                "爆裂駒捨太郎 V 2.0.0": 1.0,
                "爆裂駒捨太郎 V 2.1.0": 0.998,
                "爆裂駒捨太郎 V 2.2.0": 0.979,
                "爆裂駒捨太郎 V 2.3.0": 0.947,
                "爆裂駒捨太郎R V 1.0.0": 0.953,
                "LesserKai": 0.537,
                "Bonanza 6.0 D1": 0.109,
                "Bonanza 6.0 D2": 0.056,
                "技巧2 D1": 0.022,
                "Bonanza 6.0 D3": 0.014,
                "海底 WCSC28": 0.003,
            },
            "rating": 931,
        },
        "爆裂転生太郎 V 2.0.2": {
            "opponent": {
                "爆裂駒捨太郎 V 1.0.0": 1.0,
                "爆裂駒捨太郎 V 1.1.0": 1.0,
                "GSE 0.1.6": 0.635,
                "爆裂駒捨太郎 V 2.0.0": 0.998,
                "爆裂駒捨太郎 V 2.1.0": 1.0,
                "爆裂駒捨太郎 V 2.2.0": 1.0,
                "爆裂駒捨太郎 V 2.3.0": 0.974,
                "爆裂駒捨太郎R V 1.0.0": 0.972,
                "LesserKai": 0.629,
                "爆裂転生太郎 V 1.0.1": 0.829,
                "Bonanza 6.0 D1": 0.252,
                "Bonanza 6.0 D2": 0.060,
                "技巧2 D1": 0.027,
                "Bonanza 6.0 D3": 0.015,
                "海底 WCSC28": 0.005,
                "Bonanza 6.0 D4": 0.002,
            },
            "rating": 1017,
        },
        "Bonanza 6.0 D1": {
            "opponent": {},
            "rating": 1109,
        },
        "爆裂駒拾太郎 V 1.0.0": {
            "opponent": {
                "爆裂駒捨太郎 V 2.0.0": 1.0,
                "爆裂駒捨太郎 V 2.1.0": 1.0,
                "爆裂駒捨太郎 V 2.2.0": 0.999,
                "爆裂駒捨太郎 V 2.3.0": 0.999,
                "爆裂駒捨太郎R V 1.0.0": 0.998,
                "LesserKai": 0.909,
                "爆裂転生太郎 V 1.0.1": 0.942,
                "爆裂転生太郎 V 2.0.2": 0.870,
                "Bonanza 6.0 D1": 0.437,
                "Bonanza 6.0 D2": 0.330,
                "技巧2 D1": 0.200,
                "Bonanza 6.0 D3": 0.100,
                "海底 WCSC28": 0.038,
                "Bonanza 6.0 D4": 0.024,
                "技巧2 D2": 0.044,
                "Bonanza 6.0 D5": 0.001,
                "技巧2 D3": 0.027,
            },
            "rating": 1238,
        },
        "爆裂駒拾太郎 V 1.1.0": {
            "opponent": {
                "爆裂駒捨太郎 V 2.0.0": 0.998,
                "爆裂駒捨太郎 V 2.1.0": 1.0,
                "爆裂駒捨太郎 V 2.2.0": 1.0,
                "爆裂駒捨太郎 V 2.3.0": 1.0,
                "爆裂駒捨太郎R V 1.0.0": 0.993,
                "LesserKai": 0.922,
                "爆裂転生太郎 V 1.0.1": 0.942,
                "爆裂転生太郎 V 2.0.2": 0.885,
                "Bonanza 6.0 D1": 0.606,
                "爆裂駒拾太郎 V 1.0.0": 0.675,
                "Bonanza 6.0 D2": 0.438,
                "技巧2 D1": 0.215,
                "Bonanza 6.0 D3": 0.231,
                "海底 WCSC28": 0.054,
                "Bonanza 6.0 D4": 0.026,
                "技巧2 D2": 0.051,
                "Bonanza 6.0 D5": 0.026,
                "技巧2 D3": 0.035,
            },
            "rating": 1310,
        },
        "Bonanza 6.0 D2": {
            "opponent": {},
            "rating": 1326,
        },
        "爆裂駒拾太郎 V 1.2.0": {
            "opponent": {
                "爆裂駒捨太郎 V 2.2.0": 1.0,
                "爆裂駒捨太郎 V 2.3.0": 0.994,
                "爆裂駒捨太郎R V 1.0.0": 0.983,
                "LesserKai": 0.951,
                "爆裂転生太郎 V 1.0.1": 0.979,
                "爆裂転生太郎 V 2.0.2": 0.973,
                "Bonanza 6.0 D1": 0.717,
                "爆裂駒拾太郎 V 1.0.0": 0.836,
                "爆裂駒拾太郎 V 1.1.0": 0.656,
                "Bonanza 6.0 D2": 0.520,
                "技巧2 D1": 0.333,
                "Bonanza 6.0 D3": 0.293,
                "海底 WCSC28": 0.105,
                "Bonanza 6.0 D4": 0.089,
                "技巧2 D2": 0.129,
                "Bonanza 6.0 D5": 0.017,
                "技巧2 D3": 0.056,
                "Bonanza 6.0 D6": 0.024,
                "技巧2 D4": 0.009,
            },
            "rating": 1405,
        },
        "爆裂駒拾太郎 V 2.2.0": {
            "opponent": {
                "爆裂駒捨太郎 V 2.2.0": 0.998,
                "爆裂駒捨太郎 V 2.3.0": 0.998,
                "爆裂駒捨太郎R V 1.0.0": 1.0,
                "LesserKai": 0.961,
                "爆裂転生太郎 V 1.0.1": 0.969,
                "爆裂転生太郎 V 2.0.2": 0.946,
                "Bonanza 6.0 D1": 0.686,
                "爆裂駒拾太郎 V 1.0.0": 0.816,
                "爆裂駒拾太郎 V 1.1.0": 0.764,
                "Bonanza 6.0 D2": 0.496,
                "爆裂駒拾太郎 V 1.2.0": 0.650,
                "技巧2 D1": 0.305,
                "Bonanza 6.0 D3": 0.223,
                "海底 WCSC28": 0.187,
                "Bonanza 6.0 D4": 0.067,
                "技巧2 D2": 0.170,
                "Bonanza 6.0 D5": 0.026,
                "技巧2 D3": 0.059,
                "Bonanza 6.0 D6": 0.007,
                "技巧2 D4": 0.028,
                "技巧2 D5": 0.017,
            },
            "rating": 1420,
        },
        "技巧2 D1": {
            "opponent": {},
            "rating": 1545,
        },
        "Bonanza 6.0 D3": {
            "opponent": {},
            "rating": 1562,
        },
        "爆裂駒拾太郎 V 2.2.3": {
            "opponent": {
                "爆裂転生太郎 V 1.0.1": 0.996,
                "爆裂転生太郎 V 2.0.2": 0.993,
                "Bonanza 6.0 D1": 0.846,
                "爆裂駒拾太郎 V 1.0.0": 0.949,
                "爆裂駒拾太郎 V 1.1.0": 0.934,
                "Bonanza 6.0 D2": 0.685,
                "爆裂駒拾太郎 V 1.2.0": 0.889,
                "爆裂駒拾太郎 V 2.2.0": 0.814,
                "技巧2 D1": 0.548,
                "Bonanza 6.0 D3": 0.419,
                "海底 WCSC28": 0.396,
                "Bonanza 6.0 D4": 0.185,
                "技巧2 D2": 0.319,
                "Bonanza 6.0 D5": 0.073,
                "技巧2 D3": 0.153,
                "Bonanza 6.0 D6": 0.020,
                "技巧2 D4": 0.065,
                "技巧2 D5": 0.033,
                "Bonanza 6.0 D7": 0.006,
                "技巧2 D6": 0.012,
            },
            "rating": 1598,
        },
        "海底 WCSC28": {
            "opponent": {},
            "rating": 1619,
        },
        "爆裂駒拾太郎 V 2.2.4": {
            "opponent": {
                "爆裂転生太郎 V 1.0.1": 0.999,
                "爆裂転生太郎 V 2.0.2": 0.993,
                "Bonanza 6.0 D1": 0.853,
                "爆裂駒拾太郎 V 1.0.0": 0.985,
                "爆裂駒拾太郎 V 1.1.0": 0.973,
                "Bonanza 6.0 D2": 0.686,
                "爆裂駒拾太郎 V 1.2.0": 0.939,
                "爆裂駒拾太郎 V 2.2.0": 0.904,
                "技巧2 D1": 0.578,
                "Bonanza 6.0 D3": 0.448,
                "爆裂駒拾太郎 V 2.2.3": 0.672,
                "海底 WCSC28": 0.478,
                "Bonanza 6.0 D4": 0.191,
                "技巧2 D2": 0.330,
                "Bonanza 6.0 D5": 0.091,
                "技巧2 D3": 0.171,
                "Bonanza 6.0 D6": 0.023,
                "技巧2 D4": 0.072,
                "技巧2 D5": 0.031,
                "Bonanza 6.0 D7": 0.013,
                "技巧2 D6": 0.016,
            },
            "rating": 1642,
        },
        "爆裂駒拾太郎 V 3.0.2": {
            "opponent": {
                "爆裂転生太郎 V 1.0.1": 0.999,
                "爆裂転生太郎 V 2.0.2": 0.995,
                "Bonanza 6.0 D1": 0.921,
                "爆裂駒拾太郎 V 1.0.0": 0.991,
                "爆裂駒拾太郎 V 1.1.0": 0.983,
                "Bonanza 6.0 D2": 0.788,
                "爆裂駒拾太郎 V 1.2.0": 0.948,
                "爆裂駒拾太郎 V 2.2.0": 0.923,
                "技巧2 D1": 0.685,
                "Bonanza 6.0 D3": 0.582,
                "爆裂駒拾太郎 V 2.2.3": 0.710,
                "海底 WCSC28": 0.547,
                "爆裂駒拾太郎 V 2.2.4": 0.643,
                "Bonanza 6.0 D4": 0.284,
                "技巧2 D2": 0.411,
                "Bonanza 6.0 D5": 0.124,
                "技巧2 D3": 0.243,
                "Bonanza 6.0 D6": 0.053,
                "技巧2 D4": 0.132,
                "技巧2 D5": 0.057,
                "Bonanza 6.0 D7": 0.025,
                "技巧2 D6": 0.038,
                "技巧2 D7": 0.025,
            },
            "rating": 0,
        },
        "Bonanza 6.0 D4": {
            "opponent": {},
            "rating": 1793,
        },
        "技巧2 D2": {
            "opponent": {},
            "rating": 1825,
        },
        "Bonanza 6.0 D5": {
            "opponent": {},
            "rating": 1962,
        },
        "技巧2 D3": {
            "opponent": {},
            "rating": 2022,
        },
        "Bonanza 6.0 D6": {
            "opponent": {},
            "rating": 2121,
        },
        "技巧2 D4": {
            "opponent": {},
            "rating": 2137,
        },
        "技巧2 D5": {
            "opponent": {},
            "rating": 2210,
        },
        "Bonanza 6.0 D7": {
            "opponent": {},
            "rating": 2250,
        },
        "技巧2 D6": {
            "opponent": {},
            "rating": 2328,
        },
        "技巧2 D7": {
            "opponent": {},
            "rating": 2463,
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
