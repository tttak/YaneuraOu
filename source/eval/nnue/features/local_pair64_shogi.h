// Experiments 122-126: local piece pairs with a compile-time radius.
#ifndef NNUE_FEATURES_LOCAL_PAIR64_SHOGI_H_INCLUDED
#define NNUE_FEATURES_LOCAL_PAIR64_SHOGI_H_INCLUDED

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>

#include "../../../bitboard.h"
#include "../../../types.h"
#include "../nnue_common.h"

namespace YaneuraOu::Eval::NNUE::Features::LocalPair64Shogi {

#if defined(NNUE_SIMPLE_LOCALPAIR64_R5)
// Experiment 124 R5: silver, gold-like.
constexpr int kClasses = 2;
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R2)
// Experiment 123 R2: knight, silver, gold-like.
constexpr int kClasses = 3;
#else
// Experiment 122 L4: lance, knight, silver, gold-like, bishop, horse, rook,
// dragon.
constexpr int kClasses = 8;
#endif
constexpr int kStates = 2 * kClasses;
#if defined(NNUE_SIMPLE_LOCALPAIR32_R5_D1)
constexpr int kRadius = 1;
constexpr int kSquarePairs = 272;
#else
constexpr int kRadius = 2;
constexpr int kSquarePairs = 720;
#endif
constexpr IndexType kDimensions = kSquarePairs * kStates * kStates;
constexpr std::size_t kMaxPieces = 40;
constexpr std::size_t kMaxActive = 1024;

#if defined(NNUE_SIMPLE_LOCALPAIR64_R5)
inline int piece_class(PieceType pt) {
    switch (pt) {
    case SILVER: return 0;
    case GOLD:
    case PRO_PAWN:
    case PRO_LANCE:
    case PRO_KNIGHT:
    case PRO_SILVER: return 1;
    default: return -1;
    }
}
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R2)
// Use the physical board piece type as the single source of truth.  In
// particular, promoted knight/silver belong only to GOLD_LIKE and must not be
// emitted a second time through their unpromoted class.
inline int piece_class(PieceType pt) {
    switch (pt) {
    case KNIGHT: return 0;
    case SILVER: return 1;
    case GOLD:
    case PRO_PAWN:
    case PRO_LANCE:
    case PRO_KNIGHT:
    case PRO_SILVER: return 2;
    default: return -1;
    }
}
#endif

// [absolute owner][lance, knight, silver, gold-like, bishop, horse, rook,
// dragon].  Promoted pawn/lance/knight/silver are represented only by the
// gold-like class.
// `at` is an absolute-board lookup used by the Experiment 122a direct dirty
// path.  0 means empty/irrelevant, otherwise value-1 is owner*8+class.
// The bitboards remain the full-enumeration/P0 reference representation.
struct BoardState {
    Bitboard pieces[COLOR_NB][kClasses];
    std::array<std::uint8_t, 81> at{};

    BoardState() {
        for (int c = 0; c < COLOR_NB; ++c)
            for (int pc = 0; pc < kClasses; ++pc)
                pieces[c][pc] = Bitboard(0);
    }
};

inline void set_piece(BoardState& board, int owner, int piece_class, Square sq) {
    board.pieces[owner][piece_class] |= Bitboard(sq);
    board.at[static_cast<int>(sq)] = static_cast<std::uint8_t>(
        1 + owner * kClasses + piece_class);
}

struct IndexList {
    std::array<IndexType, kMaxActive> values{};
    std::uint16_t count = 0;
    bool overflow = false;
    void push_back(IndexType value) {
        if (count < values.size()) values[count++] = value;
        else overflow = true;
    }
    const IndexType* begin() const { return values.data(); }
    const IndexType* end() const { return values.data() + count; }
};

constexpr int mirror_square(int sq) {
    return (8 - sq / 9) * 9 + sq % 9;
}

inline int compact_square_pair(int a, int b) {
    if (b < a) std::swap(a, b);
    if (a == b) return -1;
    static const auto table = [] {
        std::array<std::int16_t, 81 * 81> result{};
        result.fill(-1);
        int index = 0;
        for (int x = 0; x < 81; ++x) {
            const int fx = x / 9, rx = x % 9;
            for (int y = x + 1; y < 81; ++y) {
                const int fy = y / 9, ry = y % 9;
                if (fy - fx <= kRadius && std::abs(ry - rx) <= kRadius)
                    result[x * 81 + y] = static_cast<std::int16_t>(index++);
            }
        }
        return result;
    }();
    return table[a * 81 + b];
}

struct LocalPiece { int square; int state; };
inline bool operator==(const LocalPiece& a, const LocalPiece& b) {
    return a.square == b.square && a.state == b.state;
}
inline bool less(const LocalPiece& a, const LocalPiece& b) {
    return a.square < b.square || (a.square == b.square && a.state < b.state);
}

struct Neighborhood {
    std::array<std::uint8_t, 24> squares{};
    std::uint8_t count = 0;
};

inline const std::array<Neighborhood, 81> kNeighborhoods = [] {
    std::array<Neighborhood, 81> result{};
    for (int sq = 0; sq < 81; ++sq) {
        const int file = sq / 9;
        const int rank = sq % 9;
        for (int df = -kRadius; df <= kRadius; ++df)
            for (int dr = -kRadius; dr <= kRadius; ++dr) {
                if ((df == 0 && dr == 0)
                    || file + df < 0 || file + df >= 9
                    || rank + dr < 0 || rank + dr >= 9)
                    continue;
                result[sq].squares[result[sq].count++] =
                    static_cast<std::uint8_t>((file + df) * 9 + rank + dr);
            }
    }
    return result;
}();

inline const std::array<Bitboard, 81> kNeighborhoodMasks = [] {
    std::array<Bitboard, 81> result{};
    for (int sq = 0; sq < 81; ++sq) {
        std::uint64_t lo = 0, hi = 0;
        for (std::uint8_t i = 0; i < kNeighborhoods[sq].count; ++i) {
            const int other = kNeighborhoods[sq].squares[i];
            if (other < 63) lo |= UINT64_C(1) << other;
            else hi |= UINT64_C(1) << (other - 63);
        }
        result[sq] = Bitboard(lo, hi);
    }
    return result;
}();

inline LocalPiece orient_piece(int absolute_square, int absolute_state,
                               Color perspective, Square friend_king) {
    int king = static_cast<int>(friend_king);
    int square = absolute_square;
    if (perspective == WHITE) {
        king = 80 - king;
        square = 80 - square;
    }
    if (king >= static_cast<int>(SQ_61)) square = mirror_square(square);
    const int owner = absolute_state / kClasses;
    const int piece_class = absolute_state % kClasses;
    return {square, (owner ^ int(perspective)) * kClasses + piece_class};
}

struct DirtyStats {
    std::uint64_t calls = 0;
    std::uint64_t changed_identities = 0;
    std::uint64_t board_lookups = 0;
    std::uint64_t relevant_candidates = 0;
    std::uint64_t emitted_before_unique = 0;
    std::uint64_t removed = 0;
    std::uint64_t added = 0;
    std::uint64_t changed_extract_ns = 0;
    std::uint64_t neighborhood_and_index_ns = 0;
    std::uint64_t duplicate_removal_ns = 0;
    std::uint64_t unchanged_calls = 0;
    std::uint64_t changed_calls = 0;
    std::uint64_t unchanged_total_ns = 0;
    std::uint64_t changed_total_ns = 0;
};

// Experiment 127 production path.  The do_move metadata contains at most two
// disappeared endpoints (mover, captured piece) and one appeared endpoint
// (moved/dropped piece).  The old board is reconstructed locally from the
// current Position: erase appeared endpoints, then restore removed endpoints.
// No 81-square comparison, snapshot construction, sorting, or unique pass is
// needed.
struct FastDirtyStats {
    std::uint64_t calls = 0;
    std::uint64_t immediate_returns = 0;
    std::uint64_t removed_endpoints = 0;
    std::uint64_t added_endpoints = 0;
    std::uint64_t board_lookups = 0;
    std::uint64_t relevant_candidates = 0;
    std::uint64_t emitted = 0;
    std::uint64_t classification_ns = 0;
    std::uint64_t neighborhood_ns = 0;
    std::uint64_t total_ns = 0;
};

inline int packed_square(std::uint16_t packed) { return packed & 0x7f; }
inline int packed_state(std::uint16_t packed) {
    return ((packed >> 7) & 1) * kClasses + ((packed >> 8) & 7);
}

template <typename PositionType, typename DeltaType>
inline void make_local_dirty_diff_fast(const PositionType& pos,
                                       const DeltaType& delta,
                                       Color perspective,
                                       Square friend_king,
                                       IndexList& removed,
                                       IndexList& added,
                                       FastDirtyStats* stats = nullptr) {
    const auto start = std::chrono::steady_clock::now();
    if (stats) ++stats->calls;
    if (delta.removed_count == 0 && delta.added_count == 0) {
        if (stats) {
            ++stats->immediate_returns;
            const auto elapsed = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
            stats->classification_ns += elapsed;
            stats->total_ns += elapsed;
        }
        return;
    }
    const auto classified = std::chrono::steady_clock::now();

    const auto after_code = [&](int sq) -> std::uint8_t {
        const Piece piece = pos.piece_on(static_cast<Square>(sq));
        if (piece == NO_PIECE) return 0;
        const int pc = piece_class(type_of(piece));
        return pc < 0 ? 0 : static_cast<std::uint8_t>(
            1 + int(color_of(piece)) * kClasses + pc);
    };
    const auto before_code = [&](int sq) -> std::uint8_t {
        // An appeared endpoint did not occupy this square in the old board.
        for (std::uint8_t i = 0; i < delta.added_count; ++i)
            if (packed_square(delta.added[i]) == sq)
                goto restore_removed;
        {
            const auto code = after_code(sq);
            if (code) return code;
        }
    restore_removed:
        for (std::uint8_t i = 0; i < delta.removed_count; ++i)
            if (packed_square(delta.removed[i]) == sq)
                return static_cast<std::uint8_t>(1 + packed_state(delta.removed[i]));
        return 0;
    };
    const auto in_changed = [](int sq, const std::uint16_t* endpoints,
                               std::uint8_t count) {
        for (std::uint8_t i = 0; i < count; ++i)
            if (packed_square(endpoints[i]) == sq) return true;
        return false;
    };

    Bitboard after_occupied(0);
    for (const Color color : {BLACK, WHITE})
        after_occupied |= pos.pieces(color, SILVER)
                        | pos.pieces(color, GOLDS);
    Bitboard before_occupied = after_occupied;
    for (std::uint8_t i = 0; i < delta.added_count; ++i)
        before_occupied &= ~Bitboard(
            static_cast<Square>(packed_square(delta.added[i])));
    for (std::uint8_t i = 0; i < delta.removed_count; ++i)
        before_occupied |= Bitboard(
            static_cast<Square>(packed_square(delta.removed[i])));

    std::uint64_t lookups = 0, relevant = 0, emitted = 0;
    const auto append = [&](const std::uint16_t* endpoints, std::uint8_t count,
                            bool old_board, IndexList& output) {
        for (std::uint8_t i = 0; i < count; ++i) {
            const int absolute_square = packed_square(endpoints[i]);
            const auto changed = orient_piece(
                absolute_square, packed_state(endpoints[i]),
                perspective, friend_king);
            Bitboard candidates = (old_board ? before_occupied : after_occupied)
                                & kNeighborhoodMasks[absolute_square];
            while (candidates) {
                const int other_square = static_cast<int>(candidates.pop());
                ++lookups;
                const auto code = old_board ? before_code(other_square)
                                            : after_code(other_square);
                ASSERT(code != 0);
                if (in_changed(other_square, endpoints, count)
                    && other_square < absolute_square)
                    continue;
                ++relevant;
                const auto other = orient_piece(
                    other_square, code - 1, perspective, friend_king);
                const auto& a = less(other, changed) ? other : changed;
                const auto& b = less(other, changed) ? changed : other;
                const auto index = feature_index(a, b);
                if (index != kDimensions) {
                    output.push_back(index);
                    ++emitted;
                }
            }
        }
    };
    append(delta.removed, delta.removed_count, true, removed);
    append(delta.added, delta.added_count, false, added);

    if (stats) {
        const auto done = std::chrono::steady_clock::now();
        stats->removed_endpoints += delta.removed_count;
        stats->added_endpoints += delta.added_count;
        stats->board_lookups += lookups;
        stats->relevant_candidates += relevant;
        stats->emitted += emitted;
        stats->classification_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                classified - start).count());
        stats->neighborhood_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                done - classified).count());
        stats->total_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                done - start).count());
    }
}

inline int collect_local_count_for_debug(const BoardState& board) {
    int count = 0;
    for (int c = 0; c < COLOR_NB; ++c)
        for (int pc = 0; pc < kClasses; ++pc)
            count += board.pieces[c][pc].pop_count();
    return count;
}

inline std::size_t collect_local(const BoardState& board, Color perspective,
                                 Square friend_king,
                                 std::array<LocalPiece, kMaxPieces>& local) {
    int king = static_cast<int>(friend_king);
    if (perspective == WHITE) king = 80 - king;
    const bool mirror = king >= static_cast<int>(SQ_61);
    std::size_t count = 0;
    for (int c = 0; c < COLOR_NB; ++c)
        for (int pc = 0; pc < kClasses; ++pc) {
            Bitboard bb = board.pieces[c][pc];
            while (bb) {
                int sq = static_cast<int>(bb.pop());
                if (perspective == WHITE) sq = 80 - sq;
                if (mirror) sq = mirror_square(sq);
                if (count < local.size())
                    local[count++] = {
                        sq, (c ^ int(perspective)) * kClasses + pc};
            }
        }
    std::sort(local.begin(), local.begin() + count, less);
    return count;
}

inline IndexType feature_index(const LocalPiece& a, const LocalPiece& b) {
    const int pair = compact_square_pair(a.square, b.square);
    return pair < 0 ? kDimensions
        : static_cast<IndexType>(pair * kStates * kStates
                                 + a.state * kStates + b.state);
}

inline void append_active(const BoardState& board, Color perspective,
                          Square friend_king, IndexList& output) {
    std::array<LocalPiece, kMaxPieces> local{};
    const auto count = collect_local(board, perspective, friend_king, local);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = i + 1; j < count; ++j) {
            if (local[j].square / 9 - local[i].square / 9 > kRadius) break;
            const auto index = feature_index(local[i], local[j]);
            if (index != kDimensions) output.push_back(index);
        }
    std::sort(output.values.begin(), output.values.begin() + output.count);
}

// Changed-identity algorithm: enumerate only pairs incident to a disappeared
// or appeared identity; never rescan every old/new pair for ordinary moves.
inline void make_local_dirty_diff_scan(const BoardState& before,
                                       const BoardState& after,
                                       Color perspective, Square friend_king,
                                       IndexList& removed, IndexList& added,
                                       DirtyStats* stats = nullptr) {
    const auto start = std::chrono::steady_clock::now();
    std::array<LocalPiece, kMaxPieces> old_local{}, new_local{};
    const auto old_count = collect_local(before, perspective, friend_king, old_local);
    const auto new_count = collect_local(after, perspective, friend_king, new_local);
    std::array<bool, 81 * kStates> old_present{}, new_present{};
    for (std::size_t i = 0; i < old_count; ++i)
        old_present[old_local[i].square * kStates + old_local[i].state] = true;
    for (std::size_t i = 0; i < new_count; ++i)
        new_present[new_local[i].square * kStates + new_local[i].state] = true;
    auto present = [](const auto& table, const LocalPiece& p) {
        return table[p.square * kStates + p.state];
    };
    std::array<LocalPiece, kMaxPieces> disappeared{}, appeared{};
    std::size_t nd = 0, na = 0;
    for (std::size_t i = 0; i < old_count; ++i)
        if (!present(new_present, old_local[i])) disappeared[nd++] = old_local[i];
    for (std::size_t i = 0; i < new_count; ++i)
        if (!present(old_present, new_local[i])) appeared[na++] = new_local[i];
    const auto changed_done = std::chrono::steady_clock::now();
    auto append_incident = [&](const auto& changed, std::size_t changed_count,
                               const auto& all, std::size_t all_count,
                               IndexList& output) {
        for (std::size_t i = 0; i < changed_count; ++i)
            for (std::size_t j = 0; j < all_count; ++j) {
                if (changed[i] == all[j]) continue;
                const bool other_changed = std::binary_search(
                    changed.begin(), changed.begin() + changed_count,
                    all[j], less);
                if (other_changed && less(all[j], changed[i])) continue;
                const auto& a = less(all[j], changed[i]) ? all[j] : changed[i];
                const auto& b = less(all[j], changed[i]) ? changed[i] : all[j];
                if (b.square / 9 - a.square / 9 > kRadius
                    || std::abs(b.square % 9 - a.square % 9) > kRadius) continue;
                const auto index = feature_index(a, b);
                if (index != kDimensions) output.push_back(index);
            }
    };
    append_incident(disappeared, nd, old_local, old_count, removed);
    append_incident(appeared, na, new_local, new_count, added);
    std::sort(removed.values.begin(), removed.values.begin() + removed.count);
    std::sort(added.values.begin(), added.values.begin() + added.count);
    auto unique_list = [](IndexList& list) {
        auto end = std::unique(list.values.begin(), list.values.begin() + list.count);
        list.count = static_cast<std::uint16_t>(end - list.values.begin());
    };
    unique_list(removed);
    unique_list(added);
    if (stats) {
        const auto done = std::chrono::steady_clock::now();
        ++stats->calls;
        stats->changed_identities += nd + na;
        // P0 scans every local piece for every changed identity.
        stats->board_lookups += nd * old_count + na * new_count;
        stats->relevant_candidates += nd * old_count + na * new_count;
        stats->removed += removed.count;
        stats->added += added.count;
        stats->changed_extract_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                changed_done - start).count());
        stats->neighborhood_and_index_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                done - changed_done).count());
    }
}

// Experiment 122a production path.  Identify changed relevant identities by
// comparing the two 81-entry absolute board lookups, then inspect only the
// precomputed compile-time-radius neighborhood of each changed endpoint.
inline void make_local_dirty_diff_direct(const BoardState& before,
                                         const BoardState& after,
                                         Color perspective,
                                         Square friend_king,
                                         IndexList& removed,
                                         IndexList& added,
                                         DirtyStats* stats = nullptr) {
    const auto start = std::chrono::steady_clock::now();
    std::array<std::uint8_t, kMaxPieces> disappeared_square{};
    std::array<std::uint8_t, kMaxPieces> disappeared_state{};
    std::array<std::uint8_t, kMaxPieces> appeared_square{};
    std::array<std::uint8_t, kMaxPieces> appeared_state{};
    std::array<bool, 81> disappeared_at{}, appeared_at{};
    std::size_t nd = 0, na = 0;
    for (int sq = 0; sq < 81; ++sq) {
        const auto old_code = before.at[sq];
        const auto new_code = after.at[sq];
        if (old_code == new_code) continue;
        if (old_code) {
            disappeared_square[nd] = static_cast<std::uint8_t>(sq);
            disappeared_state[nd++] = static_cast<std::uint8_t>(old_code - 1);
            disappeared_at[sq] = true;
        }
        if (new_code) {
            appeared_square[na] = static_cast<std::uint8_t>(sq);
            appeared_state[na++] = static_cast<std::uint8_t>(new_code - 1);
            appeared_at[sq] = true;
        }
    }
    const auto changed_done = std::chrono::steady_clock::now();

    std::uint64_t lookups = 0, relevant = 0, emitted = 0;
    auto append_incident = [&](const BoardState& board,
                               const auto& changed_square,
                               const auto& changed_state,
                               const auto& changed_at,
                               std::size_t changed_count,
                               IndexList& output) {
        for (std::size_t i = 0; i < changed_count; ++i) {
            const int absolute_square = changed_square[i];
            const auto changed = orient_piece(
                absolute_square, changed_state[i], perspective, friend_king);
            const auto& neighborhood = kNeighborhoods[absolute_square];
            for (std::uint8_t j = 0; j < neighborhood.count; ++j) {
                const int other_square = neighborhood.squares[j];
                ++lookups;
                const auto code = board.at[other_square];
                if (!code) continue;
                // If both endpoints changed, emit the unordered pair only
                // from its lower absolute-square endpoint.
                if (changed_at[other_square]
                    && other_square < absolute_square) continue;
                ++relevant;
                const auto other = orient_piece(
                    other_square, code - 1, perspective, friend_king);
                const auto& a = less(other, changed) ? other : changed;
                const auto& b = less(other, changed) ? changed : other;
                const auto index = feature_index(a, b);
                if (index != kDimensions) {
                    output.push_back(index);
                    ++emitted;
                }
            }
        }
    };
    append_incident(before, disappeared_square, disappeared_state,
                    disappeared_at, nd, removed);
    append_incident(after, appeared_square, appeared_state,
                    appeared_at, na, added);
    const auto neighborhood_done = std::chrono::steady_clock::now();

    auto sort_list = [](IndexList& list) {
        std::sort(list.values.begin(), list.values.begin() + list.count);
    };
    sort_list(removed);
    sort_list(added);
    if (stats) {
        const auto done = std::chrono::steady_clock::now();
        ++stats->calls;
        stats->changed_identities += nd + na;
        stats->board_lookups += lookups;
        stats->relevant_candidates += relevant;
        stats->emitted_before_unique += emitted;
        stats->removed += removed.count;
        stats->added += added.count;
        stats->changed_extract_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                changed_done - start).count());
        stats->neighborhood_and_index_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                neighborhood_done - changed_done).count());
        stats->duplicate_removal_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                done - neighborhood_done).count());
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                done - start).count());
        if (nd + na == 0) {
            ++stats->unchanged_calls;
            stats->unchanged_total_ns += elapsed;
        } else {
            ++stats->changed_calls;
            stats->changed_total_ns += elapsed;
        }
    }
}

inline void make_local_dirty_diff(const BoardState& before,
                                  const BoardState& after,
                                  Color perspective, Square friend_king,
                                  IndexList& removed, IndexList& added) {
#if defined(NNUE_LOCALPAIR64_DIRTY_SCAN_REFERENCE)
    make_local_dirty_diff_scan(
        before, after, perspective, friend_king, removed, added);
#else
    make_local_dirty_diff_direct(
        before, after, perspective, friend_king, removed, added);
#endif
}

inline void make_diff(const IndexList& before, const IndexList& after,
                      IndexList& removed, IndexList& added) {
    auto re = std::set_difference(before.begin(), before.end(), after.begin(),
                                  after.end(), removed.values.begin());
    removed.count = static_cast<std::uint16_t>(re - removed.values.begin());
    auto ae = std::set_difference(after.begin(), after.end(), before.begin(),
                                  before.end(), added.values.begin());
    added.count = static_cast<std::uint16_t>(ae - added.values.begin());
}

#if defined(NNUE_SIMPLE_LOCALPAIR32_R5_D1)
static_assert(kSquarePairs == 272);
static_assert(kDimensions == 4352);
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R5)
static_assert(kDimensions == 11520);
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R2)
static_assert(kDimensions == 25920);
#else
static_assert(kDimensions == 184320);
#endif

} // namespace YaneuraOu::Eval::NNUE::Features::LocalPair64Shogi

#endif
