//�󂫃}�X����\��������

#include "../../../shogi.h"

#if defined(EVAL_NNUE_EMPTY)

#include "empty.h"
#include "index_list.h"

namespace Eval {

	namespace NNUE {

		namespace Features {

			void Empty::AppendActiveIndices(const Position& pos, Color perspective, IndexList* active) {
				// �R���p�C���̌x����������邽�߁A�z��T�C�Y���������ꍇ�͉������Ȃ�
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
					if (old_p >= fe_hand_end)//�ړ��������ł͂Ȃ����ړ������󂫃}�X�ƂȂ���
						added->push_back(squareFromBonapiece(old_p));
					if (dp.dirty_num == 1) {//����ł͂Ȃ����ړ��悪�󂫃}�X���������ړ���̋󂫃}�X��������
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
