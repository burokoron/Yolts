import typing

import matplotlib.pyplot as plt
import json


def main(eval_file_path: str):
    with open(eval_file_path, "r") as f:
        eval_dict = json.load(f)
    ax_list: typing.List[typing.Any] = []
    fig = plt.figure(figsize=(10, 8))
    titles = [
        "fu", "kyo", "kei", "gin", "kin", "kaku", "hi", "gyoku",
        "to", "narikyo", "narikei", "narigin", "uma", "ryu",
    ]
    for i in range(1, 15):
        ax_list.append(fig.add_subplot(4, 4, i))
        ax_list[-1].set_title(titles[i-1])

    for p in eval_dict["pieces_dict"]["0"]:
        v = 0
        ig = 0
        score = [[0]*9 for _ in range(9)]
        for sq in eval_dict["pieces_dict"]:
            if eval_dict["pieces_dict"][sq][p] == 0:
                ig += 1
            else:
                v += eval_dict["pieces_dict"][sq][p]
            score[8 - int(sq) % 9][8 - int(sq) // 9] = eval_dict["pieces_dict"][sq][p]
        if len(eval_dict["pieces_dict"]) - ig != 0:
            v /= len(eval_dict["pieces_dict"]) - ig
        if 1 <= int(p) <= 14:
            print(f"{titles[int(p)-1]}: {int(v)}")
        else:
            print(p, int(v))
        if 1 <= int(p) <= 14:
            heatmap = ax_list[int(p) - 1].pcolor(score)
            fig.colorbar(heatmap, ax=ax_list[int(p) - 1])
    plt.tight_layout()
    plt.savefig("komawari")
    print()
    for p in eval_dict["pieces_in_hand_dict"]:
        print(p, eval_dict["pieces_in_hand_dict"][p]["1"])


if __name__ == "__main__":
    eval_file_path = "eval.json"  # 評価関数ファイル

    main(eval_file_path=eval_file_path)
