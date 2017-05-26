﻿#ifndef _LEARN_H_
#define _LEARN_H_


// =====================
//  学習時の設定
// =====================

// 以下のいずれかを選択すれば、そのあとの細々したものは自動的に選択される。
// いずれも選択しない場合は、そのあとの細々したものをひとつひとつ設定する必要がある。

// デフォルトの学習設定
// #define LEARN_DEFAULT

// elmo方式での学習設定。
// #define LEARN_ELMO_METHOD

// やねうら王2017GOKU用のデフォルトの学習設定
// ※　このオプションは実験中なので使わないように。
#define LEARN_YANEURAOU_2017_GOKU


// ----------------------
//    学習時の設定
// ----------------------

// mini-batchサイズ。
// この数だけの局面をまとめて勾配を計算する。
// 小さくするとupdate_weights()の回数が増えるので収束が速くなる。勾配が不正確になる。
// 大きくするとupdate_weights()の回数が減るので収束が遅くなる。勾配は正確に出るようになる。
// 多くの場合において、この値を変更する必要はないと思う。

#define LEARN_MINI_BATCH_SIZE (1000 * 1000 * 1)

// ファイルから1回に読み込む局面数。これだけ読み込んだあとshuffleする。
// ある程度大きいほうが良いが、この数×34byte×3倍ぐらいのメモリを消費する。10M局面なら340MB*3程度消費する。
// THREAD_BUFFER_SIZE(=10000)の倍数にすること。

#define LEARN_SFEN_READ_SIZE (1000 * 1000 * 10)

// 学習時の評価関数の保存間隔。この局面数だけ学習させるごとに保存。
// 当然ながら、保存間隔を長くしたほうが学習時間は短くなる。
#define LEARN_EVAL_SAVE_INTERVAL (80000000ULL)


// ----------------------
//    目的関数の選択
// ----------------------

// 目的関数が勝率の差の二乗和
// 詳しい説明は、learner.cppを見ること。

//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE

// 目的関数が交差エントロピー
// 詳しい説明は、learner.cppを見ること。
// いわゆる、普通の「雑巾絞り」
//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY

// 目的関数が交差エントロピーだが、勝率の関数を通さない版
// #define LOSS_FUNCTION_IS_CROSS_ENTOROPY_FOR_VALUE

// elmo(WCSC27)の方式
// #define LOSS_FUNCTION_IS_ELMO_METHOD

// ※　他、色々追加するかも。


// ----------------------
// 評価関数ファイルの保存
// ----------------------

// 保存するときのフォルダ番号を、この局面数ごとにインクリメントしていく。
// 例) "0/KK_synthesized.bin" →　"1/KK_synthesized.bin"
// 現状、10億局面ずつ。
#define EVAL_FILE_NAME_CHANGE_INTERVAL (u64)1000000000

// evalファイルの保存は(終了のときの)1度のみにする。
//#define EVAL_SAVE_ONLY_ONCE


// ----------------------
// 学習に関するデバッグ設定
// ----------------------

// 学習時のrmseとタイムスタンプの出力をこの回数に1回に減らす。
// rmseの計算は1スレッドで行なうためそこそこ時間をとられるので出力を減らすと効果がある。
#define LEARN_RMSE_OUTPUT_INTERVAL 1
#define LEARN_TIMESTAMP_OUTPUT_INTERVAL 10


// ----------------------
// ゼロベクトルからの学習
// ----------------------

// 評価関数パラメーターをゼロベクトルから学習を開始する。
// ゼロ初期化して棋譜生成してゼロベクトルから学習させて、
// 棋譜生成→学習を繰り返すとプロの棋譜に依らないパラメーターが得られる。(かも)
// (すごく時間かかる)

//#define RESET_TO_ZERO_VECTOR


// ----------------------
// configureの内容を反映
// ----------------------

#if defined (LEARN_DEFAULT)
#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE
#endif

// ----------------------
//  elmoの方法での学習
// ----------------------

#if defined( LEARN_ELMO_METHOD )
#define LOSS_FUNCTION_IS_ELMO_METHOD
#endif

// ----------------------
//  やねうら王2017GOKUの方法
// ----------------------

#if defined(LEARN_YANEURAOU_2017_GOKU)

// 損失関数、あとでよく考える。比較実験中。
//#define LOSS_FUNCTION_IS_CROSS_ENTOROPY
//#define LOSS_FUNCTION_IS_WINNING_PERCENTAGE
#define LOSS_FUNCTION_IS_ELMO_METHOD
//#define LOSS_FUNCTION_IS_YANE_ELMO_METHOD

// rmseなどの出力を減らして高速化。
#undef LEARN_RMSE_OUTPUT_INTERVAL
#define LEARN_RMSE_OUTPUT_INTERVAL 10

// 実験時は1回だけの保存で良い。
// #define EVAL_SAVE_ONLY_ONCE

#endif


// ----------------------
// Learnerで用いるstructの定義
// ----------------------
#include "../position.h"

namespace Learner
{
	// PackedSfenと評価値が一体化した構造体
	// オプションごとに書き出す内容が異なると教師棋譜を再利用するときに困るので
	// とりあえず、以下のメンバーはオプションによらずすべて書き出しておく。
	struct PackedSfenValue
	{
		// 局面
		PackedSfen sfen;

		// Learner::search()から返ってきた評価値
		s16 score;

		// PVの初手
		u16 move;

		// 初期局面からの局面の手数。
		u16 gamePly;

		// この局面の手番側が、ゲームを最終的に勝っているならtrue。負けているならfalse。
		// 引き分けに至った場合は、局面自体書き出さない。
		bool isWin;

		// 教師局面を書き出したファイルを他の人とやりとりするときに
		// この構造体サイズが不定だと困るため、paddingしてどの環境でも必ず40bytesになるようにしておく。
		u8 padding;

		// 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
	};
}

#endif // ifndef _LEARN_H_