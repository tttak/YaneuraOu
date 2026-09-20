#pragma once

#include <cstdint>

// YaneuraOu-local optional side-input extractor.
//
// This file intentionally lives in the engine source tree.  nnue-pytorch has
// its own copy for the training-data loader because the engine and trainer are
// distributed as separate repositories.  Keep schema/version and extraction
// semantics synchronized explicitly; do not include a header outside the
// YaneuraOu repository.
//
// The including translation unit defines NNUE_SIDE_INPUT_KING_SQUARE and the
// namespace wrapper because historical YaneuraOu source layouts expose
// Position in slightly different namespaces.
#if defined(NNUE_SIDE_INPUT_NAMESPACE_BEGIN)
NNUE_SIDE_INPUT_NAMESPACE_BEGIN
#endif
namespace NnueSideInput {

constexpr std::uint32_t kSchemaVersion = 1;
constexpr int kSafeEscapeInputDimensions = 16;

inline std::uint8_t safe_escape_mask_for(const Position& pos, Color us) {
    const Square king = NNUE_SIDE_INPUT_KING_SQUARE(pos, us);
    const Square normalized_king = us == BLACK ? king : Inv(king);
    std::uint8_t mask = 0;
    int direction = 0;
    for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0)
                continue;
            const int bit = 1 << direction++;
            const int nf = int(file_of(normalized_king)) + df;
            const int nr = int(rank_of(normalized_king)) + dr;
            if (nf < 0 || nf >= 9 || nr < 0 || nr >= 9)
                continue;
            Square to = File(nf) | Rank(nr);
            if (us == WHITE)
                to = Inv(to);
            const Piece piece = pos.piece_on(to);
            const bool friendly = piece != NO_PIECE && color_of(piece) == us;
            const bool controlled = bool(pos.board_effect[~us].effect(to));
            if (!friendly && !controlled)
                mask |= static_cast<std::uint8_t>(bit);
        }
    return mask;
}

// Low byte is side-to-move; high byte is the other perspective.
inline std::uint16_t safe_escape_mask16(const Position& pos) {
    const Color stm = pos.side_to_move();
    return static_cast<std::uint16_t>(safe_escape_mask_for(pos, stm))
         | (static_cast<std::uint16_t>(safe_escape_mask_for(pos, ~stm)) << 8);
}

}  // namespace NnueSideInput
#if defined(NNUE_SIDE_INPUT_NAMESPACE_END)
NNUE_SIDE_INPUT_NAMESPACE_END
#endif

