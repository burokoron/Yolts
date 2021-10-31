# Yolts

詰将棋に意味はあるのか、はたまた無いのか、永遠に終わらない不毛な議論に終止符を打つ崇高なプロジェクト。  
その名もYolts (You only look Tsume-Shogi) プロジェクト。

## 方針

- やねうらお氏公開の[詰将棋問題集500万問](https://yaneuraou.yaneu.com/2020/12/25/christmas-present/)を用いて将棋エンジンを開発する
- 場合によってはその限りではない

## 使い方

1. [Releases](https://github.com/burokoron/Yolts/releases)から任意のバージョンのYoltsV*.zipをダウンロードする
2. ダウンロードしたzipファイルを展開する
3. フォルダ内にある任意の*.batファイルを将棋所やShogiGUIで将棋エンジンとして登録する
4. 対局する

## イロレーティング

[QRL Software rating](https://www.qhapaq.org/shogi/)基準のレーティング
|エンジン名|レーティング|  
| :---: | ---: |
|RandamKun V 1.0.0|81|
|爆裂駒捨太郎 V 1.0.0|279|
|爆裂駒捨太郎 V 1.1.0|396|
|爆裂駒捨太郎 V 2.0.0|478|
|爆裂駒捨太郎 V 2.1.0|556|
|爆裂駒捨太郎 V 2.2.0|615|
|爆裂駒捨太郎 V 2.2.1|-|

## 更新履歴

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
