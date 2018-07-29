//空きマス情報を表す特徴量

#include "../../../shogi.h"

#if defined(EVAL_NNUE_EMPTY)

#include "empty.h"
#include "index_list.h"

namespace Eval {

	namespace NNUE {

		namespace Features {

			void Empty::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
				// コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
				if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

				Bitboard empties = pos.empties();

				for (int i = 0; i < SQ_NB; i++) {
					Square sq = static_cast<Square>(i);
					if (empties & sq) {
						if (perspective == WHITE)
							sq = Inv(sq);
						active->push_back(sq);
					}
				}
			}

			void Empty::AppendChangedIndices(const Position & pos, Color perspective, IndexList * removed, IndexList * added)
			{
				const auto& dp = pos.state()->dirtyPiece;

				if (dp.dirty_num > 0) {
					const auto old_p = static_cast<BonaPiece>(dp.changed_piece[0].old_piece.from[perspective]);
					if (old_p >= fe_hand_end)//移動元が駒台ではない＝移動元が空きマスとなった
						added->push_back(squareFromBonapiece(old_p));
					if (dp.dirty_num == 1) {//駒取りではない＝移動先が空きマスだった＝移動先の空きマスが消えた
						const auto new_p = static_cast<BonaPiece>(
							dp.changed_piece[0].new_piece.from[perspective]);
						removed->push_back(squareFromBonapiece(new_p));
					}
				}
			}

		}  // namespace Features

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
