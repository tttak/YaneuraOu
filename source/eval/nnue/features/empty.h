// 空きマスを表すの入力特徴量Emptyの定義

#ifndef _NNUE_FEATURES_Empty_H_
#define _NNUE_FEATURES_Empty_H_

#include "../../../shogi.h"

#if defined(EVAL_NNUE_EMPTY)

#include "../../../evaluate.h"
#include "features_common.h"

namespace Eval {

	namespace NNUE {

		namespace Features {

			// 特徴量Empty：空きマスの位置
			class Empty {
			public:
				// 特徴量名
				static constexpr const char* kName = "Empty";
				// 評価関数ファイルに埋め込むハッシュ値
				static constexpr std::uint32_t kHashValue = 0xA704B200u;
				// 特徴量の次元数
				static constexpr IndexType kDimensions = static_cast<IndexType>(SQ_NB);
				// 特徴量のうち、同時に値が1となるインデックスの数の最大値
				static constexpr IndexType kMaxActiveDimensions = SQ_NB;
				// 差分計算の代わりに全計算を行うタイミング
				static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

				// 特徴量のうち、値が1であるインデックスのリストを取得する
				static void AppendActiveIndices(const Position& pos, Color perspective,
					IndexList* active);

				// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
				static void AppendChangedIndices(const Position& pos, Color perspective,
					IndexList* removed, IndexList* added);

			private:
			};

		}  // namespace Features

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
