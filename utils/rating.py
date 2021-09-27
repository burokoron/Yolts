"""
イロ・レーティングの計算シミュレーション
"""

import random
import typing

# 基準エンジンのレーティングとそれに対する勝率
fixed_rating_players: typing.Dict[str, typing.List[float]] = {
    "elo_rating": [166, 400, 750],
    "win_rate": [0.528, 0.014, 0.0],
}
# イテレーション
iteration = 1000000

my_elo_rating = min(fixed_rating_players["elo_rating"])


def update(my_elo_rating: float, iteration: int) -> typing.Tuple[float, float]:
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


c = my_elo_rating
while c > 1:
    a, b = update(my_elo_rating=my_elo_rating, iteration=iteration)
    c = abs(my_elo_rating - (a + b) / 2)
    my_elo_rating = (a + b) / 2
    print(my_elo_rating)

print("Elo Rating =", my_elo_rating)
