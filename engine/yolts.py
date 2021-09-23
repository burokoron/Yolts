"""
対局エンジンのメイン
"""

import argparse

from RandomKun import RandomKun


def main(engine_name: str) -> None:
    """
    使用するエンジンを起動する

    Args:
        engine_name: str
            エンジン名
    """

    if engine_name == "RandomKun":
        player = RandomKun()
        player.start()
    else:
        print(f"Not found engine = '{engine_name}'")


if __name__ == "__main__":
    """
    Args:
        2コマンド目：str
            使用するエンジン名
    """

    parser = argparse.ArgumentParser(description="将棋対局エンジン")
    parser.add_argument("engine_name", help="使用するエンジン名")
    args = parser.parse_args()

    main(args.engine_name)
