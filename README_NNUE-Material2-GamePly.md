# 駒割りと手数のNNUE評価関数
駒割りと初期局面からの手数を特徴量としたNNUE評価関数と実行ファイルです。

## NNUE-Material2-GamePly
- 特徴量：[NNUE-Material2](https://github.com/tttak/YaneuraOu/releases/tag/V4.83_NNUE-Material)の特徴量（駒の種類と枚数。ただし、盤上の駒と持ち駒を区別）と初期局面からの手数（最大255手）
- 特徴量のうち、同時に値が1となるインデックスの数の最大値：21(=20+1)
- 特徴量の次元数：386(=130+256)
- ネットワーク構造：386->256x2-32-32

