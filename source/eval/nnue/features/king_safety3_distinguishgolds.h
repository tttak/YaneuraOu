// NNUE評価関数の入力特徴量KingSafety3_DistinguishGoldsの定義

#ifndef _NNUE_FEATURES_KING_SAFETY3_DISTINGUISH_GOLDS_H_
#define _NNUE_FEATURES_KING_SAFETY3_DISTINGUISH_GOLDS_H_

#include "../../../config.h"

#if defined(EVAL_NNUE) && defined(LONG_EFFECT_LIBRARY)

#include "../../../evaluate.h"
#include "features_common.h"

#include "../../../extra/long_effect.h"

namespace YaneuraOu {
namespace Eval::NNUE::Features {

// 特徴量KingSafety3_DistinguishGolds：玉の安全度（玉の24近傍の駒と利き数）（金と小駒の成り駒（と金、成香、成桂、成銀）を区別する）
template <Side AssociatedKing>
class KingSafety3_DistinguishGolds {
 public:
  // 特徴量名
  static constexpr const char* kName =
      (AssociatedKing == Side::kFriend) ? "KingSafety3_DistinguishGolds(Friend)" : "KingSafety3_DistinguishGolds(Enemy)";
  // 評価関数ファイルに埋め込むハッシュ値
  static constexpr std::uint32_t kHashValue =
      0x596B2375u ^ (AssociatedKing == Side::kFriend);

  // 壁のPiece値を定義
  static constexpr Piece PIECE_WALL = PIECE_NB;
  static constexpr Piece PIECE_WALL_NB = static_cast<Piece>(PIECE_WALL + 1);

  // 特徴量の次元数
  // ・玉の24近傍
  // ・駒はNO_PIECE、PIECE_WALLを含む（※1）
  // ・利き数は先手と後手を同時に評価する
  // ・利き数は0～3に制限する
  // ※1 KingSafety_DistinguishGoldsとは異なりKingSafety3_DistinguishGoldsでは駒にPIECE_WALLを含めないが、互換性のため次元数はKingSafety_DistinguishGoldsと同じにしておく）
  static constexpr IndexType kDimensions = static_cast<IndexType>(Effect24::DIRECT_NB) * static_cast<IndexType>(PIECE_WALL_NB) * 4 * 4;
  // 特徴量のうち、同時に値が1となるインデックスの数の最大値
  static constexpr IndexType kMaxActiveDimensions = Effect24::DIRECT_NB;

  // 差分計算の代わりに全計算を行うタイミング
  static constexpr TriggerEvent kRefreshTrigger =
      (AssociatedKing == Side::kFriend) ?
      TriggerEvent::kFriendKingMoved : TriggerEvent::kEnemyKingMoved;

  // 特徴量のうち、値が1であるインデックスのリストを取得する
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);

  // 特徴量のインデックスを求める
  static IndexType MakeIndex(Color perspective, Effect24::Direct dir, Piece pc, int effect1, int effect2);

  // Pieceの先後反転
  static Piece Inv(Piece pc);

  // Effect24::Directの先後反転
  static Effect24::Direct Inv(Effect24::Direct dir);

  // BonaPieceからSquareとPieceを取得する
  static void GetSquarePieceFromBonaPiece(BonaPiece bp, Square &sq, Piece &pc);

  // Effect24::Directの算出
  static Effect24::Direct CalcDirect(Square sq_king, Square sq);

  // 利き数の取得
  static int GetEffectCount(const Position& pos, Square sq, Color perspective, bool prev_effect);
};

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE) && defined(LONG_EFFECT_LIBRARY)

#endif
