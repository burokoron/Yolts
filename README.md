# Yolts

詰将棋に意味はあるのか、はたまた無いのか、永遠に終わらない不毛な議論に終止符を打つ崇高なプロジェクト。  
~~その名もYolts (You only look Tsume-Shogi) プロジェクト。~~  
その名もYolts (You **not** only look Tsume-Shogi) プロジェクト。

## 方針

- ~~やねうらお氏公開の[詰将棋問題集500万問](https://yaneuraou.yaneu.com/2020/12/25/christmas-present/)を用いて将棋エンジンを開発する~~
- 真に入玉を目指す、だいたい入玉宣言で勝つ将棋エンジンを開発する
- 場合によってはその限りではない

## 使い方

1. [Releases](https://github.com/burokoron/Yolts/releases)から任意のバージョンのYoltsV*.zipをダウンロードする
2. ダウンロードしたzipファイルを展開する
3. フォルダ内にある任意の*.batファイルもしくは*.exeファイルを将棋所やShogiGUIで将棋エンジンとして登録する
4. 対局する

## イロレーティング

[QRL Software rating](https://www.qhapaq.org/shogi/)基準のレーティング
| エンジン名 | レーティング |
| :---: | ---: |
| [RandamKun V 1.0.0](https://github.com/burokoron/Yolts/releases/tag/v0.0.1) | 127 |
| [爆裂駒捨太郎 V 1.0.0](https://github.com/burokoron/Yolts/releases/tag/v0.1.0) | 321 |
| [爆裂駒捨太郎 V 1.1.0](https://github.com/burokoron/Yolts/releases/tag/v0.2.0) | 437 |
| [爆裂駒捨太郎 V 2.0.0](https://github.com/burokoron/Yolts/releases/tag/v1.0.0) | 516 |
| [爆裂駒捨太郎 V 2.1.0](https://github.com/burokoron/Yolts/releases/tag/v1.1.0) | 595 |
| [爆裂駒捨太郎 V 2.2.0](https://github.com/burokoron/Yolts/releases/tag/v1.2.0) | 650 |
| [爆裂駒捨太郎 V 2.2.1](https://github.com/burokoron/Yolts/releases/tag/v1.2.1) | - |
| [爆裂駒捨太郎 V 2.3.0](https://github.com/burokoron/Yolts/releases/tag/v1.3.0) | 712 |
| [爆裂駒捨太郎R V 0.5.0](https://github.com/burokoron/Yolts/releases/tag/v1.3.1) | - |
| [爆裂駒捨太郎R V 1.0.0](https://github.com/burokoron/Yolts/releases/tag/v1.4.0) | 754 |
| [爆裂転生太郎 V 1.0.1](https://github.com/burokoron/Yolts/releases/tag/v2.0.0) | 931 |
| [爆裂転生太郎 V 2.0.2](https://github.com/burokoron/Yolts/releases/tag/v2.1.2) | 1017 |
| [爆裂駒拾太郎 V 1.0.0](https://github.com/burokoron/Yolts/releases/tag/v3.0.0) | 1238 |
| [爆裂駒拾太郎 V 1.1.0](https://github.com/burokoron/Yolts/releases/tag/v3.1.0) | 1310 |
| [爆裂駒拾太郎 V 1.2.0](https://github.com/burokoron/Yolts/releases/tag/v3.2.0) | 1405 |
| [爆裂駒拾太郎 V 2.2.0](https://github.com/burokoron/Yolts/releases/tag/v3.6.0) | 1420 |
| [爆裂駒拾太郎 V 2.2.3](https://github.com/burokoron/Yolts/releases/tag/v3.6.3) | 1598 |
| [爆裂駒拾太郎 V 2.2.4](https://github.com/burokoron/Yolts/releases/tag/v3.6.4) | 1642 |
| [爆裂駒拾太郎 V 3.0.2](https://github.com/burokoron/Yolts/releases/tag/v4.0.2) | 1714 |
| [爆裂駒拾太郎 V 3.0.3](https://github.com/burokoron/Yolts/releases/tag/v4.0.3) | 1765 |

## 更新履歴

- 2025/06/25
  - 爆裂駒拾太郎 V 3.0.3のレーティングを計測(R1765)
- 2025/04/07
  - 爆裂駒拾太郎 V 3.0.3にアップデート
    - PPe4-4型評価関数(第0世代)に変更
  - 爆裂訓練太郎 V 4.0.0にアップデート
    - PPT/2e2型第0世代評価関数に変更
    - 静止探索において王手をかける手を生成するように変更
    - Late Move Reductionsの調整
- 2024/12/04
  - 爆裂駒拾太郎 V 3.0.2のレーティングを計測(R1714)
- 2024/11/18
  - 爆裂駒拾太郎 V 3.0.2にアップデート
    - Late Move Reductionsの調整
- 2024/11/10
  - 爆裂駒拾太郎 V 3.0.1にアップデート
    - 静止探索において王手をかける手を生成するように変更
- 2024/11/05
  - 爆裂駒拾太郎 V 3.0.0にアップデート
    - PPT/2e2型第0世代評価関数に変更
  - 爆裂訓練太郎 V 3.0.2にアップデート
    - 置換表を世代管理し、前回の探索結果を活用するように変更
    - 置換表のサイズを調整
    - Null Move Pruningのパラメータを調整
    - Late Move Reductionsのパラメータを調整
    - PPT/2型評価関数(第2.1世代)に変更
- 2024/6/30
  - 爆裂駒拾太郎 V 2.2.4のレーティングを計測(R1642)
- 2024/6/28
  - 爆裂駒拾太郎 V 2.2.4にアップデート
    - PPT/2型評価関数(第2.1世代)に変更
- 2024/5/3
  - 爆裂駒拾太郎 V 2.2.3のレーティングを計測(R1598)
- 2024/4/12
  - 爆裂駒拾太郎 V 2.2.3にアップデート
    - 置換表を世代管理し、前回の探索結果を活用するように変更
    - 置換表のサイズを調整
    - Null Move Pruningのパラメータを調整
    - Late Move Reductionsのパラメータを調整
- 2024/4/3
  - 爆裂駒拾太郎 V 2.2.2にアップデート
    - PPT/2型評価関数(第2世代)に変更
  - 爆裂訓練太郎 V 3.0.1にアップデート
    - PPT/2型評価関数(第1世代)に変更
    - 評価関数ファイルが読み込まれていない場合、isreadyコマンドで読み込むように変更
    - 置換表の実装をVectorに変更し、Vectorはエンジン起動直後にメモリ確保するように変更
    - Move Ordering部分の指し手並び替え用配列を静的確保するように変更
- 2024/3/10
  - 評価関数をスパーステンソルを用いて学習するように変更
- 2024/2/10
  - 爆裂駒拾太郎 V 2.2.1にアップデート
    - PPT/2型評価関数(第1世代)に変更
    - 評価関数ファイルが読み込まれていない場合、isreadyコマンドで読み込むように変更
    - 置換表の実装をVectorに変更し、Vectorはエンジン起動直後にメモリ確保するように変更
    - Move Ordering部分の指し手並び替え用配列を静的確保するように変更
  - 爆裂訓練太郎 V 3.0.0にアップデート
    - PPT/2型評価関数(第0世代)を実装
    - 棋譜生成時に千日手手順を選択しにくくなるように変更
    - Late Move Reductionsを実装
- 2024/2/7
  - 評価関数の学習を高速化
- 2024/1/2
  - 評価関数の学習に使用しているライブラリをTensorFlowからPyTorchに変更
- 2024/1/1
  - 爆裂転生太郎をアーカイブ
- 2023/12/3
  - 爆裂駒拾太郎 V 2.2.0のレーティングを計測(R1420)
- 2023/10/30
  - 爆裂駒拾太郎 V 2.2.0にアップデート
    - Late Move Reductionsを実装
- 2023/10/24
  - 爆裂駒拾太郎 V 2.1.0にアップデート
    - PPT/2型評価関数の差分計算を実装
- 2023/10/15
  - 爆裂駒捨太郎Rをアーカイブ
  - 爆裂駒拾太郎 V 2.0.0にアップデート
    - PPT/2型評価関数(第0世代)を実装
  - 爆裂訓練太郎 V 2.3.0にアップデート
    - K/3K/3PT/2型評価関数(第0世代)を実装
- 2023/07/24
  - 爆裂駒拾太郎 V 1.3.0にアップデート
    - K/3K/3PT/2型評価関数(第0世代)を実装
  - 爆裂訓練太郎 V 2.2.0にアップデート
    - K/9K/9PT/2型評価関数(第3世代)に変更
    - Null Move Pruningの実装
- 2023/07/02
  - 爆裂駒拾太郎 V 1.2.0のレーティングを計測(R1405)
- 2023/06/13
  - 爆裂駒拾太郎 V 1.2.0にアップデート
    - K/9K/9PT/2型評価関数(第3世代)に変更
    - Null Move Pruningの実装
  - 爆裂訓練太郎 V 2.1.1-rc1にアップデート
    - K/9K/9PT/2型評価関数(第2世代)に変更
- 2023/05/03
  - 爆裂駒拾太郎 V 1.1.0にアップデート
    - K/9K/9PT/2型評価関数(第0世代)に変更
    - レーティングを計測(R1310)
  - 爆裂訓練太郎 V 2.0.3にアップデート
    - K/27K/27PT/2型評価関数(第5世代)に変更
- 2023/04/22
  - 爆裂駒拾太郎 V 1.0.0のレーティングを計測(R1238)
- 2023/02/14
  - 爆裂駒拾太郎 V 1.0.0にアップデート
    - K/27K/27PT/2型評価関数(第1世代)に変更
  - 爆裂訓練太郎 V 2.0.0にアップデート
    - K/27K/27PT/2型評価関数(第0世代)に変更
    - 静止探索を実装
    - MVV-LVAを実装
    - Killer Heuristicを実装
- 2022/12/31
  - 爆裂駒拾太郎 V 0.3.0にアップデート
    - 静止探索を実装
    - MVV-LVAを実装
    - Killer Heuristicを実装
- 2022/12/15
  - 爆裂駒拾太郎 V 0.2.0にアップデート
    - K/27K/27PT/2型評価関数を実装
- 2022/12/04
  - 爆裂転生太郎 V 2.0.2のレーティングを計測(R1017)
- 2022/11/28
  - 爆裂駒拾太郎を追加
    - KKPT型評価関数を使用
- 2022/11/23
  - RandomKunと爆裂転生太郎をアーカイブ
- 2022/08/19
  - 爆裂転生太郎 V 2.0.2にアップデート
    - 強化学習による評価関数(第2世代)に変更
  - 爆裂訓練太郎 V 1.0.2にアップデート
    - 強化学習による評価関数(第1世代)に変更
    - 評価値が一定以上なら第0世代の評価関数に切り替えるように変更
- 2022/08/10
  - 爆裂転生太郎 V 2.0.1にアップデート
    - 合法手生成に使用しているyasai 0.4.0(一部改変)を0.5.0(一部改変)に変更
  - 爆裂訓練太郎 V 1.0.1にアップデート
    - 合法手生成に使用しているyasai 0.4.0(一部改変)を0.5.0(一部改変)に変更
- 2022/08/08
  - 爆裂転生太郎 V 2.0.0にアップデート
    - 強化学習による評価関数(第1世代)に更新
    - 爆裂訓練太郎 V 1.0.0の内容を反映
  - 爆裂訓練太郎 V 1.0.0にアップデート
    - Thompson Samplingを用いた定跡作成機能を追加
    - 学習用教師データ作成スクリプトを追加
    - 評価関数学習スクリプトを追加
    - Futility Pruningをコメントアウト
- 2022/07/27
  - 爆裂転生太郎 V 1.0.1のレーティングを計測(R931)
- 2022/07/08
  - 爆裂訓練太郎 V 0.0.3にアップデート
    - 合法手生成に使用しているyasai 0.1.3を0.4.0(一部改変)に変更
    - リファクタリング
- 2022/06/29
  - QRL Software ratingの最新レーティングで再計算
- 2022/06/23
  - 爆裂転生太郎 V 1.0.1にアップデート
    - 探索深度制限を設定に追加
  - 爆裂訓練太郎 V 0.0.2を追加
    - 爆裂転生太郎から相手玉を詰ますと負けになるように変更
    - 爆裂転生太郎に入玉宣言勝ち判定を追加
- 2022/06/05
  - 爆裂転生太郎 V 1.0.0を追加
    - ムーブオーダリングを手番、駒種、着手位置をキーにしたHistory Heuristicに変更
    - ムーブオーダリングで前回の最善手順から探索するように変更
- 2022/05/07
  - 爆裂駒捨太郎R V 1.0.0にアップデート
    - 局面管理に使用していたshogi-rsをyasaiに変更、NPSが約30倍に
    - 探索部のムーブオーダリングの調整
    - 探索部の枝刈および深度延長を調整
- 2022/04/03
  - 爆裂駒捨太郎R V 0.5.0を追加
    - Pythonで記述された爆裂駒捨太郎をRustで書き直したもの
- 2022/03/21
  - QRL Software ratingの最新レーティングで再計算
- 2021/11/21
  - 探索中に1手詰めチェックを追加
  - Mate Distance Pruningを追加
  - null move pruningを追加
  - Futility Pruningを追加
  - 指し手を出力する機能を追加
  - 爆裂駒捨太郎 V 2.3.0のレーティングを計測(R675)
- 2021/10/31
  - 爆裂駒捨太郎 V 2.2.1にアップデート
    - 優等局面、劣等局面、連続王手の千日手を評価値0にしていたバグを修正
    - 持ち時間制御を追加
  - 爆裂駒捨太郎 V 2.3.0にアップデート
    - 置換表の追加
      - 初期局面7手読みの探索局面数が約59.3%に減少(880904⇒522150)
- 2021/10/22
  - 爆裂駒捨太郎 V 2.2.0のレーティングを計測(R615)
- 2021/10/17
  - 爆裂駒捨太郎 V 2.2.0にアップデート
    - ムーブオーダリング改善
      - 各局面の指し手と評価値登録を最善評価値が更新された場合のみ登録するように変更
      - 同一深度において着手位置で評価値の高かった順に並べる方法を追加
      - 初期局面7手読みの探索局面数が約22.6%に減少(3898485⇒880904)
    - 探索深さを3手から4手に変更
    - フィッシャールール時の追加時間が持ち時間となるバグを修正
    - 探索において負け&時間による探索打ち切り時に探索中の中間手を指すバグの修正
- 2021/10/12
  - 爆裂駒捨太郎 V 2.1.0のレーティングを計測(R556)
- 2021/10/10
  - 爆裂駒捨太郎 V 2.1.0にアップデート
    - ムーブオーダリング&反復深化探索の実装
      - 初期局面6手読みの探索局面数が約67.1%に減少(1042068⇒698980)
    - 探索深さを2手全探索から3手全探索に変更
- 2021/10/04
  - 爆裂駒捨太郎 V 2.0.0のレーティングを計測(R478)
- 2021/10/03
  - 爆裂駒捨太郎 V 2.0.0にアップデート
    - [詰将棋問題集500万問](https://yaneuraou.yaneu.com/2020/12/25/christmas-present/)を用いて調整した評価関数を作成
      - 1駒関係評価関数
      - 統計&情報量に基づき、詰将棋初期局面で出現頻度の大きい駒位置は低評価、その逆は高評価となるように設計
    - GUIから設定できるパラメータに評価関数ファイルパスを追加
    - 逆王手で非合法手を出力する問題の解決
    - 千日手は評価値0を出力するように変更
    - 探索深さを3手(王手0.5手延長)全探索から2手全探索に変更
  - 爆裂駒捨太郎の評価パラメータ調整コードを追加
- 2021/09/28
  - 爆裂駒捨太郎 V 1.0.0のレーティングを計測(R279)
  - 爆裂駒捨太郎 V 1.1.0にアップデート
    - エンジン名ミスの修正
    - 自玉詰みで投了するバグの修正
    - 最短で詰まさないバグの修正
    - 持ち時間を考慮して思考するように変更
    - ネガアルファ(fail-soft)探索の実装
    - 探索時間の表示
    - NPSの表示
  - 爆裂駒捨太郎 V 1.1.0のレーティングを計測(R396)
- 2021/09/27
  - イロレーティング計算シミュレーションコードを作成
  - RandamKun V 1.0.0のレーティングを計測(R81)
- 2021/09/25
  - 3手全探索(王手0.5手延長)する将棋エンジン(爆裂駒捨太郎 Version 1.0.0)を追加
- 2021/09/23
  - リポジトリ作成
  - README初期コミット
  - ランダムムーブする将棋エンジン(RandamKun Version 1.0.0)を追加
