// NNUE評価関数の入力特徴量KingSafetyの定義

#ifndef _NNUE_FEATURES_KING_SAFETY_H_
#define _NNUE_FEATURES_KING_SAFETY_H_

#include "../../../config.h"

#if defined(EVAL_NNUE) && defined(LONG_EFFECT_LIBRARY)

#include "../../../evaluate.h"
#include "features_common.h"

#include "../../../extra/long_effect.h"

namespace Eval {

namespace NNUE {

namespace Features {

// 特徴量KingSafety：玉の安全度（玉の24近傍の駒と利き数）
class KingSafety {
 public:
  // 特徴量名
  static constexpr const char* kName = "KingSafety";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t kHashValue = 0x596B2375u;

  // 壁のPiece値を定義
  static constexpr Piece PIECE_WALL = PIECE_NB;
  static constexpr Piece PIECE_WALL_NB = static_cast<Piece>(PIECE_WALL + 1);

  // 特徴量の次元数
  // ・玉の24近傍
  // ・駒はNO_PIECE、PIECE_WALLを含む
  // ・利き数は先手と後手を同時に評価する
  // ・利き数は0～3に制限する
  static constexpr IndexType kDimensions = Effect24::DIRECT_NB * static_cast<IndexType>(PIECE_WALL_NB) * 4 * 4;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType kMaxActiveDimensions = Effect24::DIRECT_NB;
  // 差分計算の代わりに全計算を行うタイミング
  // ・とりあえず差分計算なし
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);

  // 特徴量のインデックスを求める
  static IndexType MakeIndex(int index24, Piece pc, int effect1, int effect2);
};

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
