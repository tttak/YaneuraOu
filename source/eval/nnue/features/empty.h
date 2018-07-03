// �󂫃}�X��\���̓��͓�����Empty�̒�`

#ifndef _NNUE_FEATURES_Empty_H_
#define _NNUE_FEATURES_Empty_H_

#include "../../../shogi.h"

#if defined(EVAL_NNUE_EMPTY)

#include "../../../evaluate.h"
#include "features_common.h"

namespace Eval {

	namespace NNUE {

		namespace Features {

			// ������Empty�F�󂫃}�X�̈ʒu
			class Empty {
			public:
				// �����ʖ�
				static constexpr const char* kName = "Empty";
				// �]���֐��t�@�C���ɖ��ߍ��ރn�b�V���l
				static constexpr std::uint32_t kHashValue = 0xA704B200u;
				// �����ʂ̎�����
				static constexpr IndexType kDimensions = static_cast<IndexType>(SQ_NB);
				// �����ʂ̂����A�����ɒl��1�ƂȂ�C���f�b�N�X�̐��̍ő�l
				static constexpr IndexType kMaxActiveDimensions = SQ_NB;
				// �����v�Z�̑���ɑS�v�Z���s���^�C�~���O
				static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

				// �����ʂ̂����A�l��1�ł���C���f�b�N�X�̃��X�g���擾����
				static void AppendActiveIndices(const Position& pos, Color perspective,
					IndexList* active);

				// �����ʂ̂����A���O����l���ω������C���f�b�N�X�̃��X�g���擾����
				static void AppendChangedIndices(const Position& pos, Color perspective,
					IndexList* removed, IndexList* added);

			private:
			};

		}  // namespace Features

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
