// Experiment 120: local board-only unpromoted pawn/lance pair feature.
#ifndef NNUE_FEATURES_PP3WIDE_SHOGI_H_INCLUDED
#define NNUE_FEATURES_PP3WIDE_SHOGI_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>

#include "../../../bitboard.h"
#include "../../../types.h"
#include "../nnue_common.h"

namespace YaneuraOu::Eval::NNUE::Features::Pp3WideShogi {

constexpr IndexType kDimensions = 15552;
constexpr std::size_t kMaxActive = 256;

struct BoardState {
    Bitboard pieces[COLOR_NB][2]; // [absolute owner][pawn/lance]
};

struct IndexList {
    std::array<IndexType, kMaxActive> values{};
    std::uint16_t count = 0;
    bool overflow = false;
    void push_back(IndexType value) {
        if (count < values.size())
            values[count++] = value;
        else
            overflow = true;
    }
    const IndexType* begin() const { return values.data(); }
    const IndexType* end() const { return values.data() + count; }
};

constexpr int mirror_square(int sq) {
    return (8 - sq / 9) * 9 + sq % 9;
}

constexpr int compact_square_pair(int a, int b) {
    if (b < a) {
        const int t = a; a = b; b = t;
    }
    if (a == b)
        return -1;
    const int fa = a / 9, ra = a % 9;
    const int fb = b / 9, rb = b % 9;
    if (fa == fb)
        return fa * 36 + rb * (rb - 1) / 2 + ra;
    if (fb == fa + 1)
        return 324 + fa * 81 + ra * 9 + rb;
    return -1;
}

struct LocalPiece { int square; int state; };

inline bool operator==(const LocalPiece& a, const LocalPiece& b) {
    return a.square == b.square && a.state == b.state;
}

inline std::size_t collect_local(const BoardState& board, Color perspective,
                                 Square friend_king,
                                 std::array<LocalPiece, 22>& local) {
    int king = static_cast<int>(friend_king);
    if (perspective == WHITE)
        king = 80 - king;
    const bool mirror = king >= static_cast<int>(SQ_61);
    std::size_t count = 0;
    for (int c = 0; c < COLOR_NB; ++c)
        for (int pc = 0; pc < 2; ++pc) {
            Bitboard bb = board.pieces[c][pc];
            while (bb) {
                int sq = static_cast<int>(bb.pop());
                if (perspective == WHITE)
                    sq = 80 - sq;
                if (mirror)
                    sq = mirror_square(sq);
                local[count++] = {sq, (c ^ int(perspective)) * 2 + pc};
            }
        }
    std::sort(local.begin(), local.begin() + count,
              [](const LocalPiece& a, const LocalPiece& b) {
                  return a.square < b.square
                      || (a.square == b.square && a.state < b.state);
              });
    return count;
}

inline IndexType feature_index(const LocalPiece& a, const LocalPiece& b) {
    const int pair = compact_square_pair(a.square, b.square);
    if (pair < 0)
        return kDimensions;
    // collect_local() orders by square, so the state order follows the
    // canonical square order used by the Python/loader contract.
    return static_cast<IndexType>(pair * 16 + a.state * 4 + b.state);
}

inline void append_active(const BoardState& board, Color perspective,
                          Square friend_king, IndexList& output) {
    std::array<LocalPiece, 22> local{};
    const std::size_t count = collect_local(board, perspective, friend_king, local);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = i + 1; j < count; ++j) {
            const IndexType index = feature_index(local[i], local[j]);
            if (index != kDimensions)
                output.push_back(index);
        }
    std::sort(output.values.begin(), output.values.begin() + output.count);
}

// Generate only relations incident to an identity that appeared/disappeared.
// This is equivalent to a full active-set difference while avoiding the
// O(n^2) enumeration of unchanged local pairs on ordinary moves.
inline void make_local_dirty_diff(const BoardState& before,
                                  const BoardState& after,
                                  Color perspective, Square friend_king,
                                  IndexList& removed, IndexList& added) {
    std::array<LocalPiece, 22> old_local{}, new_local{};
    const std::size_t old_count = collect_local(before, perspective, friend_king, old_local);
    const std::size_t new_count = collect_local(after, perspective, friend_king, new_local);

    std::array<bool, 81 * 4> old_present{}, new_present{};
    for (std::size_t i = 0; i < old_count; ++i)
        old_present[old_local[i].square * 4 + old_local[i].state] = true;
    for (std::size_t i = 0; i < new_count; ++i)
        new_present[new_local[i].square * 4 + new_local[i].state] = true;
    auto present = [](const auto& table, const LocalPiece& piece) {
        return table[piece.square * 4 + piece.state];
    };

    for (std::size_t i = 0; i < old_count; ++i)
        for (std::size_t j = i + 1; j < old_count; ++j)
            if (!present(new_present, old_local[i])
                || !present(new_present, old_local[j])) {
                const IndexType index = feature_index(old_local[i], old_local[j]);
                if (index != kDimensions)
                    removed.push_back(index);
            }

    for (std::size_t i = 0; i < new_count; ++i)
        for (std::size_t j = i + 1; j < new_count; ++j)
            if (!present(old_present, new_local[i])
                || !present(old_present, new_local[j])) {
                const IndexType index = feature_index(new_local[i], new_local[j]);
                if (index != kDimensions)
                    added.push_back(index);
            }

    std::sort(removed.values.begin(), removed.values.begin() + removed.count);
    std::sort(added.values.begin(), added.values.begin() + added.count);
}

inline void make_diff(const IndexList& before, const IndexList& after,
                      IndexList& removed, IndexList& added) {
    auto removed_end = std::set_difference(
        before.begin(), before.end(), after.begin(), after.end(),
        removed.values.begin());
    removed.count = static_cast<std::uint16_t>(removed_end - removed.values.begin());
    auto added_end = std::set_difference(
        after.begin(), after.end(), before.begin(), before.end(),
        added.values.begin());
    added.count = static_cast<std::uint16_t>(added_end - added.values.begin());
}

static_assert(kDimensions == (9 * 36 + 8 * 81) * 16);

} // namespace YaneuraOu::Eval::NNUE::Features::Pp3WideShogi

#endif
