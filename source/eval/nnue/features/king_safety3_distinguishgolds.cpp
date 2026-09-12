// NNUE評価関数の入力特徴量KingSafety3_DistinguishGoldsの定義

#include "../../../config.h"

#if defined(EVAL_NNUE) && defined(LONG_EFFECT_LIBRARY) && defined(USE_BOARD_EFFECT_PREV) && defined(DISTINGUISH_GOLDS)

#include "king_safety3_distinguishgolds.h"
#include "index_list.h"

#if defined(ENABLE_NNUE_BENCH) \
    || defined(USE_NNUE_KSDG3_EFFECT_TOUCHED_MASK)
#include <array>
#include <cstdint>
#endif
#if defined(ENABLE_NNUE_BENCH)
#include <chrono>
#endif

namespace YaneuraOu {
namespace Eval::NNUE::Features {

#if defined(ENABLE_NNUE_BENCH)
namespace {

constexpr std::uint8_t kInvalidKsdg3Direction = 0xff;

struct Ksdg3Neighbor {
  std::uint8_t square;
  std::uint8_t direction;
};

struct Ksdg3Neighborhood {
  std::array<Ksdg3Neighbor, Effect24::DIRECT_NB> neighbors{};
  std::uint8_t count = 0;
};

constexpr int Ksdg3Abs(const int value) { return value < 0 ? -value : value; }

constexpr std::uint8_t Ksdg3Direction(const int king, const int square) {
  const int king_file = king / 9;
  const int king_rank = king % 9;
  const int file_diff = square / 9 - king_file;
  const int rank_diff = square % 9 - king_rank;
  if ((file_diff == 0 && rank_diff == 0)
      || Ksdg3Abs(file_diff) > 2 || Ksdg3Abs(rank_diff) > 2)
    return kInvalidKsdg3Direction;
  const int uncompressed = file_diff * 5 + rank_diff + 12;
  return static_cast<std::uint8_t>(
      uncompressed - (uncompressed >= 12));
}

constexpr auto BuildKsdg3DirectionTable() {
  std::array<std::array<std::uint8_t, SQ_NB>, SQ_NB> table{};
  for (int king = 0; king < SQ_NB; ++king)
    for (int square = 0; square < SQ_NB; ++square)
      table[king][square] = Ksdg3Direction(king, square);
  return table;
}

constexpr auto BuildKsdg3Neighborhoods() {
  std::array<Ksdg3Neighborhood, SQ_NB> table{};
  for (int king = 0; king < SQ_NB; ++king) {
    // Insert by direction so iteration remains bit-for-bit/order compatible
    // with Effect24::Direct(), while keeping constexpr work below Clang's
    // default evaluation step limit.
    for (int square = 0; square < SQ_NB; ++square) {
      const std::uint8_t direction = Ksdg3Direction(king, square);
      if (direction == kInvalidKsdg3Direction)
        continue;
      std::uint8_t insertion = table[king].count;
      while (insertion != 0
             && table[king].neighbors[insertion - 1].direction > direction) {
        table[king].neighbors[insertion] =
            table[king].neighbors[insertion - 1];
        --insertion;
      }
      table[king].neighbors[insertion] = {
          static_cast<std::uint8_t>(square), direction};
      ++table[king].count;
    }
  }
  return table;
}

constexpr auto kKsdg3DirectionTable = BuildKsdg3DirectionTable();
constexpr auto kKsdg3Neighborhoods = BuildKsdg3Neighborhoods();

constexpr auto BuildKsdg3SquareByDirection() {
  std::array<std::array<std::uint8_t, Effect24::DIRECT_NB>, SQ_NB> table{};
  for (int king = 0; king < SQ_NB; ++king) {
    for (int direction = 0; direction < Effect24::DIRECT_NB; ++direction)
      table[king][direction] = 0xff;
    for (int square = 0; square < SQ_NB; ++square) {
      const auto direction = Ksdg3Direction(king, square);
      if (direction != kInvalidKsdg3Direction)
        table[king][direction] = static_cast<std::uint8_t>(square);
    }
  }
  return table;
}

auto BuildKsdg3NeighborhoodBitboards() {
  std::array<Bitboard, SQ_NB> table;
  for (int king = 0; king < SQ_NB; ++king) {
    table[king] = Bitboard(ZERO);
    for (std::uint8_t i = 0; i < kKsdg3Neighborhoods[king].count; ++i)
      table[king] |= static_cast<Square>(
          kKsdg3Neighborhoods[king].neighbors[i].square);
  }
  return table;
}

constexpr auto kKsdg3SquareByDirection = BuildKsdg3SquareByDirection();

const auto& Ksdg3NeighborhoodBitboards() {
  // SquareBB is initialized at engine startup, so this table must not be
  // constructed during namespace static initialization.
  static const auto table = BuildKsdg3NeighborhoodBitboards();
  return table;
}

Ksdg3BenchmarkVariant kKsdg3BenchmarkVariant =
    Ksdg3BenchmarkVariant::kBaseline;
thread_local Ksdg3BenchmarkStageTiming* kKsdg3BenchmarkStageTiming = nullptr;

inline int CappedKsdg3Effect(const LongEffect::ByteBoard& effects,
                             const Square square) {
  return std::min(int(effects.effect(square)), 3);
}

template <Side AssociatedKing, bool HoistEffects, bool UseTables>
void AppendKsdg3ActiveCandidate(const Position& pos, Color perspective,
                                IndexList* const active) {
  using Feature = KingSafety3_DistinguishGolds<AssociatedKing>;
  if constexpr (AssociatedKing == Side::kEnemy)
    perspective = ~perspective;
  const Color opponent = ~perspective;
  const Square king = pos.square<KING>(perspective);
  const auto& now_us = pos.board_effect[perspective];
  const auto& now_them = pos.board_effect[opponent];

  if constexpr (UseTables) {
    const auto& neighborhood = kKsdg3Neighborhoods[king];
    for (std::uint8_t i = 0; i < neighborhood.count; ++i) {
      const auto neighbor = neighborhood.neighbors[i];
      const Square square = static_cast<Square>(neighbor.square);
      const auto direction = static_cast<Effect24::Direct>(neighbor.direction);
      const int us = HoistEffects
          ? CappedKsdg3Effect(now_us, square)
          : Feature::GetEffectCount(pos, square, perspective, false);
      const int them = HoistEffects
          ? CappedKsdg3Effect(now_them, square)
          : Feature::GetEffectCount(pos, square, opponent, false);
      active->push_back(Feature::MakeIndex(
          perspective, direction, pos.piece_on(square), us, them));
    }
  } else {
    const SquareWithWall king_with_wall = to_sqww(king);
    for (Effect24::Direct direction : Effect24::Direct()) {
      const SquareWithWall square_with_wall =
          king_with_wall + DirectToDeltaWW(direction);
      if (!is_ok(square_with_wall))
        continue;
      const Square square = sqww_to_sq(square_with_wall);
      const int us = HoistEffects
          ? CappedKsdg3Effect(now_us, square)
          : Feature::GetEffectCount(pos, square, perspective, false);
      const int them = HoistEffects
          ? CappedKsdg3Effect(now_them, square)
          : Feature::GetEffectCount(pos, square, opponent, false);
      active->push_back(Feature::MakeIndex(
          perspective, direction, pos.piece_on(square), us, them));
    }
  }
}

template <Side AssociatedKing, bool HoistEffects, bool UseTables,
          bool UseTouchedMask = false>
void AppendKsdg3ChangedCandidate(const Position& pos, Color perspective,
                                 IndexList* const removed,
                                 IndexList* const added) {
  using Feature = KingSafety3_DistinguishGolds<AssociatedKing>;
  if constexpr (AssociatedKing == Side::kEnemy)
    perspective = ~perspective;

  const Color opponent = ~perspective;
  const Square king = pos.square<KING>(perspective);
  const SquareWithWall king_with_wall = to_sqww(king);
  const auto& dirty_piece = pos.state()->dirtyPiece;
  const auto& prev_us = pos.board_effect_prev[perspective];
  const auto& prev_them = pos.board_effect_prev[opponent];
  const auto& now_us = pos.board_effect[perspective];
  const auto& now_them = pos.board_effect[opponent];
  Bitboard dirty_squares(ZERO);
  std::uint32_t dirty_directions = 0;

  const auto prev_effect_us = [&](const Square square) {
    if constexpr (HoistEffects)
      return CappedKsdg3Effect(prev_us, square);
    return Feature::GetEffectCount(pos, square, perspective, true);
  };
  const auto prev_effect_them = [&](const Square square) {
    if constexpr (HoistEffects)
      return CappedKsdg3Effect(prev_them, square);
    return Feature::GetEffectCount(pos, square, opponent, true);
  };
  const auto now_effect_us = [&](const Square square) {
    if constexpr (HoistEffects)
      return CappedKsdg3Effect(now_us, square);
    return Feature::GetEffectCount(pos, square, perspective, false);
  };
  const auto now_effect_them = [&](const Square square) {
    if constexpr (HoistEffects)
      return CappedKsdg3Effect(now_them, square);
    return Feature::GetEffectCount(pos, square, opponent, false);
  };

  const auto direction_of = [&](const Square square) {
    if constexpr (UseTables) {
      const std::uint8_t direction = kKsdg3DirectionTable[king][square];
      return direction == kInvalidKsdg3Direction
          ? Effect24::DIRECT_NB
          : static_cast<Effect24::Direct>(direction);
    }
    return dist(king, square) <= 2
        ? Feature::CalcDirect(king, square)
        : Effect24::DIRECT_NB;
  };

  const auto mark_dirty = [&](const Square square,
                              const Effect24::Direct direction) {
    if constexpr (UseTables)
      dirty_directions |= UINT32_C(1) << static_cast<unsigned>(direction);
    else
      dirty_squares |= square;
  };

  std::chrono::steady_clock::time_point dirty_begin;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    dirty_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < dirty_piece.dirty_num; ++i) {
    const auto old_piece = static_cast<BonaPiece>(
        dirty_piece.changed_piece[i].old_piece.from[BLACK]);
    Square old_square;
    Piece old_board_piece;
    Feature::GetSquarePieceFromBonaPiece(
        old_piece, old_square, old_board_piece);
    if (old_square != SQ_NB) {
      const auto direction = direction_of(old_square);
      if (direction != Effect24::DIRECT_NB) {
        mark_dirty(old_square, direction);
        removed->push_back(Feature::MakeIndex(
            perspective, direction, old_board_piece,
            prev_effect_us(old_square), prev_effect_them(old_square)));
        if (i == 0)
          added->push_back(Feature::MakeIndex(
              perspective, direction, NO_PIECE,
              now_effect_us(old_square), now_effect_them(old_square)));
      }
    }

    const auto new_piece = static_cast<BonaPiece>(
        dirty_piece.changed_piece[i].new_piece.from[BLACK]);
    Square new_square;
    Piece new_board_piece;
    Feature::GetSquarePieceFromBonaPiece(
        new_piece, new_square, new_board_piece);
    if (new_square != SQ_NB) {
      const auto direction = direction_of(new_square);
      if (direction != Effect24::DIRECT_NB) {
        mark_dirty(new_square, direction);
        if ((dirty_piece.dirty_num == 1 && i == 0)
            || (dirty_piece.dirty_num == 2 && i == 1))
          removed->push_back(Feature::MakeIndex(
              perspective, direction, NO_PIECE,
              prev_effect_us(new_square), prev_effect_them(new_square)));
        added->push_back(Feature::MakeIndex(
            perspective, direction, new_board_piece,
            now_effect_us(new_square), now_effect_them(new_square)));
      }
    }
  }
  std::chrono::steady_clock::time_point dirty_end;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    dirty_end = std::chrono::steady_clock::now();

  const auto process_neighbor = [&](const Square square,
                                    const Effect24::Direct direction) {
    const bool dirty = UseTables
        ? (dirty_directions &
           (UINT32_C(1) << static_cast<unsigned>(direction))) != 0
        : static_cast<bool>(dirty_squares & square);
    if (dirty)
      return;
    const int previous_us = prev_effect_us(square);
    const int previous_them = prev_effect_them(square);
    const int current_us = now_effect_us(square);
    const int current_them = now_effect_them(square);
    if (previous_us != current_us || previous_them != current_them) {
      if (kKsdg3BenchmarkStageTiming != nullptr)
        ++kKsdg3BenchmarkStageTiming->capped_changed_squares;
      const Piece piece = pos.piece_on(square);
      removed->push_back(Feature::MakeIndex(
          perspective, direction, piece, previous_us, previous_them));
      added->push_back(Feature::MakeIndex(
          perspective, direction, piece, current_us, current_them));
    }
  };

  std::chrono::steady_clock::time_point neighbor_begin;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    neighbor_begin = std::chrono::steady_clock::now();
  if constexpr (UseTouchedMask) {
    Bitboard touched = pos.state()->effect_touched_any
                     & Ksdg3NeighborhoodBitboards()[king];
    if (kKsdg3BenchmarkStageTiming != nullptr) {
      kKsdg3BenchmarkStageTiming->valid_neighbors +=
          kKsdg3Neighborhoods[king].count;
      kKsdg3BenchmarkStageTiming->touched_near_king += touched.pop_count();
    }
    std::uint32_t touched_directions = 0;
    while (touched) {
      const Square square = touched.pop();
      const auto direction = kKsdg3DirectionTable[king][square];
      if (direction != kInvalidKsdg3Direction)
        touched_directions |= UINT32_C(1) << direction;
    }
    touched_directions &= ~dirty_directions;
    while (touched_directions) {
      const unsigned direction = LSB32(touched_directions);
      touched_directions &= touched_directions - 1;
      const auto square = kKsdg3SquareByDirection[king][direction];
      process_neighbor(static_cast<Square>(square),
                       static_cast<Effect24::Direct>(direction));
    }
  } else if constexpr (UseTables) {
    const auto& neighborhood = kKsdg3Neighborhoods[king];
    if (kKsdg3BenchmarkStageTiming != nullptr)
      kKsdg3BenchmarkStageTiming->valid_neighbors += neighborhood.count;
    for (std::uint8_t i = 0; i < neighborhood.count; ++i) {
      const auto neighbor = neighborhood.neighbors[i];
      process_neighbor(static_cast<Square>(neighbor.square),
                       static_cast<Effect24::Direct>(neighbor.direction));
    }
  } else {
    for (Effect24::Direct direction : Effect24::Direct()) {
      const SquareWithWall square_with_wall =
          king_with_wall + DirectToDeltaWW(direction);
      if (is_ok(square_with_wall)) {
        if (kKsdg3BenchmarkStageTiming != nullptr)
          ++kKsdg3BenchmarkStageTiming->valid_neighbors;
        process_neighbor(sqww_to_sq(square_with_wall), direction);
      }
    }
  }
  std::chrono::steady_clock::time_point neighbor_end;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    neighbor_end = std::chrono::steady_clock::now();
  if (kKsdg3BenchmarkStageTiming != nullptr) {
    ++kKsdg3BenchmarkStageTiming->calls;
    kKsdg3BenchmarkStageTiming->dirty_nanoseconds +=
        std::chrono::duration<double, std::nano>(dirty_end - dirty_begin).count();
    kKsdg3BenchmarkStageTiming->neighbor_nanoseconds +=
        std::chrono::duration<double, std::nano>(neighbor_end - neighbor_begin).count();
  }
}

}  // namespace

void SetKsdg3BenchmarkVariant(const Ksdg3BenchmarkVariant variant) {
  kKsdg3BenchmarkVariant = variant;
}

Ksdg3BenchmarkVariant GetKsdg3BenchmarkVariant() {
  return kKsdg3BenchmarkVariant;
}

void SetKsdg3BenchmarkStageTiming(Ksdg3BenchmarkStageTiming* const timing) {
  kKsdg3BenchmarkStageTiming = timing;
}

std::uint64_t ValidateKsdg3BenchmarkTables() {
  std::uint64_t mismatches = 0;
  for (int king = 0; king < SQ_NB; ++king) {
    std::uint8_t expected_count = 0;
    for (int direction = 0; direction < Effect24::DIRECT_NB; ++direction) {
      bool found = false;
      for (int square = 0; square < SQ_NB; ++square) {
        if (Ksdg3Direction(king, square) == direction) {
          found = true;
          const auto& neighbor =
              kKsdg3Neighborhoods[king].neighbors[expected_count];
          mismatches += neighbor.square != square;
          mismatches += neighbor.direction != direction;
          ++expected_count;
          break;
        }
      }
      (void)found;
    }
    mismatches += kKsdg3Neighborhoods[king].count != expected_count;
    for (int square = 0; square < SQ_NB; ++square) {
      const bool nearby = dist(static_cast<Square>(king),
                               static_cast<Square>(square)) <= 2
                       && king != square;
      const auto table_direction = kKsdg3DirectionTable[king][square];
      mismatches += nearby != (table_direction != kInvalidKsdg3Direction);
      if (nearby)
        mismatches += table_direction != static_cast<std::uint8_t>(
            KingSafety3_DistinguishGolds<Side::kFriend>::CalcDirect(
                static_cast<Square>(king), static_cast<Square>(square)));
      if (nearby)
        mismatches += kKsdg3SquareByDirection[king][table_direction]
                    != square;
    }
    mismatches += Ksdg3NeighborhoodBitboards()[king].pop_count()
                != expected_count;
  }
  return mismatches;
}
#endif

#if defined(USE_NNUE_KSDG3_EFFECT_TOUCHED_MASK)
namespace {

constexpr std::uint8_t kProductionInvalidDirection = 0xff;

constexpr int ProductionAbs(const int value) {
  return value < 0 ? -value : value;
}

constexpr std::uint8_t ProductionDirection(const int king, const int square) {
  const int file_diff = square / 9 - king / 9;
  const int rank_diff = square % 9 - king % 9;
  if ((file_diff == 0 && rank_diff == 0)
      || ProductionAbs(file_diff) > 2 || ProductionAbs(rank_diff) > 2)
    return kProductionInvalidDirection;
  const int uncompressed = file_diff * 5 + rank_diff + 12;
  return static_cast<std::uint8_t>(
      uncompressed - (uncompressed >= 12));
}

constexpr auto BuildProductionDirectionTable() {
  std::array<std::array<std::uint8_t, SQ_NB>, SQ_NB> table{};
  for (int king = 0; king < SQ_NB; ++king)
    for (int square = 0; square < SQ_NB; ++square)
      table[king][square] = ProductionDirection(king, square);
  return table;
}

constexpr auto BuildProductionSquareByDirection() {
  std::array<std::array<std::uint8_t, Effect24::DIRECT_NB>, SQ_NB> table{};
  for (int king = 0; king < SQ_NB; ++king) {
    for (int direction = 0; direction < Effect24::DIRECT_NB; ++direction)
      table[king][direction] = 0xff;
    for (int square = 0; square < SQ_NB; ++square) {
      const auto direction = ProductionDirection(king, square);
      if (direction != kProductionInvalidDirection)
        table[king][direction] = static_cast<std::uint8_t>(square);
    }
  }
  return table;
}

constexpr auto kProductionDirectionTable = BuildProductionDirectionTable();
constexpr auto kProductionSquareByDirection =
    BuildProductionSquareByDirection();

auto BuildProductionNeighborhoodBitboards() {
  std::array<Bitboard, SQ_NB> table;
  for (int king = 0; king < SQ_NB; ++king) {
    table[king] = Bitboard(ZERO);
    for (int square = 0; square < SQ_NB; ++square)
      if (kProductionDirectionTable[king][square]
          != kProductionInvalidDirection)
        table[king] |= static_cast<Square>(square);
  }
  return table;
}

const auto& ProductionNeighborhoodBitboards() {
  // SquareBB is initialized during engine startup, so initialize lazily.
  static const auto table = BuildProductionNeighborhoodBitboards();
  return table;
}

inline int ProductionCappedEffect(const LongEffect::ByteBoard& effects,
                                  const Square square) {
  return std::min(int(effects.effect(square)), 3);
}

template <Side AssociatedKing>
void AppendKsdg3ChangedProduction(const Position& pos, Color perspective,
                                  IndexList* const removed,
                                  IndexList* const added) {
  using Feature = KingSafety3_DistinguishGolds<AssociatedKing>;
  if constexpr (AssociatedKing == Side::kEnemy)
    perspective = ~perspective;

  const Color opponent = ~perspective;
  const Square king = pos.square<KING>(perspective);
  const auto& dirty_piece = pos.state()->dirtyPiece;
  const auto& prev_us = pos.board_effect_prev[perspective];
  const auto& prev_them = pos.board_effect_prev[opponent];
  const auto& now_us = pos.board_effect[perspective];
  const auto& now_them = pos.board_effect[opponent];
  std::uint32_t dirty_directions = 0;

  const auto previous_us = [&](const Square square) {
    return ProductionCappedEffect(prev_us, square);
  };
  const auto previous_them = [&](const Square square) {
    return ProductionCappedEffect(prev_them, square);
  };
  const auto current_us = [&](const Square square) {
    return ProductionCappedEffect(now_us, square);
  };
  const auto current_them = [&](const Square square) {
    return ProductionCappedEffect(now_them, square);
  };

  for (int i = 0; i < dirty_piece.dirty_num; ++i) {
    const auto old_piece = static_cast<BonaPiece>(
        dirty_piece.changed_piece[i].old_piece.from[BLACK]);
    Square old_square;
    Piece old_board_piece;
    Feature::GetSquarePieceFromBonaPiece(
        old_piece, old_square, old_board_piece);
    if (old_square != SQ_NB) {
      const auto direction = kProductionDirectionTable[king][old_square];
      if (direction != kProductionInvalidDirection) {
        dirty_directions |= UINT32_C(1) << direction;
        const auto direct = static_cast<Effect24::Direct>(direction);
        removed->push_back(Feature::MakeIndex(
            perspective, direct, old_board_piece,
            previous_us(old_square), previous_them(old_square)));
        if (i == 0)
          added->push_back(Feature::MakeIndex(
              perspective, direct, NO_PIECE,
              current_us(old_square), current_them(old_square)));
      }
    }

    const auto new_piece = static_cast<BonaPiece>(
        dirty_piece.changed_piece[i].new_piece.from[BLACK]);
    Square new_square;
    Piece new_board_piece;
    Feature::GetSquarePieceFromBonaPiece(
        new_piece, new_square, new_board_piece);
    if (new_square != SQ_NB) {
      const auto direction = kProductionDirectionTable[king][new_square];
      if (direction != kProductionInvalidDirection) {
        dirty_directions |= UINT32_C(1) << direction;
        const auto direct = static_cast<Effect24::Direct>(direction);
        if ((dirty_piece.dirty_num == 1 && i == 0)
            || (dirty_piece.dirty_num == 2 && i == 1))
          removed->push_back(Feature::MakeIndex(
              perspective, direct, NO_PIECE,
              previous_us(new_square), previous_them(new_square)));
        added->push_back(Feature::MakeIndex(
            perspective, direct, new_board_piece,
            current_us(new_square), current_them(new_square)));
      }
    }
  }

  Bitboard touched = pos.state()->effect_touched_any
                   & ProductionNeighborhoodBitboards()[king];
  std::uint32_t touched_directions = 0;
  while (touched) {
    const Square square = touched.pop();
    const auto direction = kProductionDirectionTable[king][square];
    if (direction != kProductionInvalidDirection)
      touched_directions |= UINT32_C(1) << direction;
  }
  touched_directions &= ~dirty_directions;

  // Enumerate the direction mask in the original Effect24::Direct order.
  while (touched_directions) {
    const unsigned direction = LSB32(touched_directions);
    touched_directions &= touched_directions - 1;
    const Square square = static_cast<Square>(
        kProductionSquareByDirection[king][direction]);
    const int prev_us_value = previous_us(square);
    const int prev_them_value = previous_them(square);
    const int now_us_value = current_us(square);
    const int now_them_value = current_them(square);
    if (prev_us_value == now_us_value && prev_them_value == now_them_value)
      continue;
    const auto direct = static_cast<Effect24::Direct>(direction);
    const Piece piece = pos.piece_on(square);
    removed->push_back(Feature::MakeIndex(
        perspective, direct, piece, prev_us_value, prev_them_value));
    added->push_back(Feature::MakeIndex(
        perspective, direct, piece, now_us_value, now_them_value));
  }
}

}  // namespace
#endif

// 盤上の駒のBonaPieceからPieceへの変換配列
Piece sqBonaPieceToPiece3[] = {B_PAWN, W_PAWN, B_LANCE, W_LANCE, B_KNIGHT, W_KNIGHT, B_SILVER, W_SILVER, B_GOLD, W_GOLD
    , B_BISHOP, W_BISHOP, B_HORSE, W_HORSE, B_ROOK, W_ROOK, B_DRAGON, W_DRAGON
    , B_PRO_PAWN, W_PRO_PAWN, B_PRO_LANCE, W_PRO_LANCE, B_PRO_KNIGHT, W_PRO_KNIGHT, B_PRO_SILVER, W_PRO_SILVER
    , B_KING, W_KING };

// BonaPieceからSquareとPieceを取得する
// ・持ち駒の場合は「SQ_NB、NO_PIECE」を返す。本来は持ち駒のPieceを正確に算出することもできるが、この評価関数では不要なので。
template <Side AssociatedKing>
inline void KingSafety3_DistinguishGolds<AssociatedKing>::GetSquarePieceFromBonaPiece(BonaPiece bp, Square &sq, Piece &pc) {
  // 持ち駒の場合
  if (bp < fe_hand_end) {
    sq = SQ_NB;
    pc = NO_PIECE;
  }
  // 盤上の駒の場合
  else {
    int offset = bp - fe_hand_end;
    sq = static_cast<Square>(offset % SQ_NB);
    pc = sqBonaPieceToPiece3[offset / SQ_NB];
  }
}

// Effect24::Directの算出
template <Side AssociatedKing>
inline Effect24::Direct KingSafety3_DistinguishGolds<AssociatedKing>::CalcDirect(Square sq_king, Square sq) {
  int file_diff = file_of(sq) - file_of(sq_king);
  int rank_diff = rank_of(sq) - rank_of(sq_king);
  int calc = file_diff * 5 + rank_diff + 12;

  //return Effect24::Direct(calc < 12 ? calc : calc - 1);
  return Effect24::Direct(calc - (calc >= 12));
}

// 利き数の取得
template <Side AssociatedKing>
inline int KingSafety3_DistinguishGolds<AssociatedKing>::GetEffectCount(const Position& pos, Square sq, Color perspective, bool prev_effect) {
  if (sq == SQ_NB) {
    return 0;
  }
  else {
    const auto& eff = prev_effect ? pos.board_effect_prev[perspective] : pos.board_effect[perspective];
    return std::min(int(eff.effect(sq)), 3);
  }
}

// Pieceの先後反転
template <Side AssociatedKing>
inline Piece KingSafety3_DistinguishGolds<AssociatedKing>::Inv(Piece pc) {
  if (pc == NO_PIECE) {
    return NO_PIECE;
  }
  else if (pc == PIECE_WALL) {
    return PIECE_WALL;
  }
  else {
    return make_piece(~color_of(pc), type_of(pc));
  }
}

// Effect24::Directの先後反転
template <Side AssociatedKing>
inline Effect24::Direct KingSafety3_DistinguishGolds<AssociatedKing>::Inv(Effect24::Direct dir) {
  return Effect24::DIRECT_NB - static_cast<Effect24::Direct>(1) - dir;
}

// 特徴量のインデックスを求める
template <Side AssociatedKing>
inline IndexType KingSafety3_DistinguishGolds<AssociatedKing>::MakeIndex(Color perspective, Effect24::Direct dir, Piece pc, int effect1, int effect2) {
  if (perspective == WHITE) {
    pc = Inv(pc);
    dir = Inv(dir);
  }

  return ((static_cast<IndexType>(dir)
      * static_cast<IndexType>(PIECE_WALL_NB) + static_cast<IndexType>(pc))
      * 4 + effect1)
      * 4 + effect2;
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
template <Side AssociatedKing>
void KingSafety3_DistinguishGolds<AssociatedKing>::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
#if defined(ENABLE_NNUE_BENCH)
  switch (GetKsdg3BenchmarkVariant()) {
    case Ksdg3BenchmarkVariant::kEffectHoist:
      // Active enumeration has no previous/current board selection to hoist.
      break;
    case Ksdg3BenchmarkVariant::kNeighborTables:
      AppendKsdg3ActiveCandidate<AssociatedKing, false, true>(
          pos, perspective, active);
      return;
    case Ksdg3BenchmarkVariant::kCombined:
      AppendKsdg3ActiveCandidate<AssociatedKing, true, true>(
          pos, perspective, active);
      return;
    case Ksdg3BenchmarkVariant::kTouchedMask:
      AppendKsdg3ActiveCandidate<AssociatedKing, true, true>(
          pos, perspective, active);
      return;
    case Ksdg3BenchmarkVariant::kMaskOnly:
    case Ksdg3BenchmarkVariant::kBaseline:
      break;
  }
#endif
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  if (AssociatedKing == Side::kEnemy) {
    perspective = ~perspective;
  }

  // perspective側の玉のマス（先手目線）
  SquareWithWall sqww_king = to_sqww(pos.square<KING>(perspective));

  // 24近傍をループ
  for (Effect24::Direct dir : Effect24::Direct()) {
    SquareWithWall sqww = sqww_king + DirectToDeltaWW(dir);

    // 盤内の場合
    if (is_ok(sqww)) {
      Square sq = sqww_to_sq(sqww);

#if defined(ENABLE_NNUE_BENCH)
      if (kKsdg3BenchmarkStageTiming != nullptr)
        ++kKsdg3BenchmarkStageTiming->valid_neighbors;
#endif
      active->push_back(MakeIndex(perspective, dir, pos.piece_on(sq)
          , GetEffectCount(pos, sq, perspective, false)
          , GetEffectCount(pos, sq, ~perspective, false)
        ));
    }

    // 盤外の場合、何もしない

  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
template <Side AssociatedKing>
void KingSafety3_DistinguishGolds<AssociatedKing>::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {

#if defined(USE_NNUE_KSDG3_EFFECT_TOUCHED_MASK)
  AppendKsdg3ChangedProduction<AssociatedKing>(
      pos, perspective, removed, added);
  return;
#elif defined(ENABLE_NNUE_BENCH)
  switch (GetKsdg3BenchmarkVariant()) {
    case Ksdg3BenchmarkVariant::kEffectHoist:
      AppendKsdg3ChangedCandidate<AssociatedKing, true, false>(
          pos, perspective, removed, added);
      return;
    case Ksdg3BenchmarkVariant::kNeighborTables:
      AppendKsdg3ChangedCandidate<AssociatedKing, false, true>(
          pos, perspective, removed, added);
      return;
    case Ksdg3BenchmarkVariant::kCombined:
      AppendKsdg3ChangedCandidate<AssociatedKing, true, true>(
          pos, perspective, removed, added);
      return;
    case Ksdg3BenchmarkVariant::kTouchedMask:
      AppendKsdg3ChangedCandidate<AssociatedKing, true, true, true>(
          pos, perspective, removed, added);
      return;
    case Ksdg3BenchmarkVariant::kMaskOnly:
    case Ksdg3BenchmarkVariant::kBaseline:
      break;
  }
#endif

  if (AssociatedKing == Side::kEnemy) {
    perspective = ~perspective;
  }

  // perspective側の玉のマス（先手目線）
  Square sq_king = pos.square<KING>(perspective);
  SquareWithWall sqww_king = to_sqww(sq_king);

  // pieces（先手目線）
  BonaPiece* pieces = pos.eval_list()->piece_list_fb();
  const auto& dp = pos.state()->dirtyPiece;

  // 玉の24近傍でdirtyなマス
  Bitboard dirty_bb(ZERO);

#if defined(ENABLE_NNUE_BENCH)
  std::chrono::steady_clock::time_point dirty_begin;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    dirty_begin = std::chrono::steady_clock::now();
#endif
  for (int i = 0; i < dp.dirty_num; ++i) {
    // old_piece（先手目線）
    const auto old_piece = static_cast<BonaPiece>(dp.changed_piece[i].old_piece.from[BLACK]);
    Square old_sq;
    Piece old_pc;
    GetSquarePieceFromBonaPiece(old_piece, old_sq, old_pc);

    // old_pieceのマスが玉の24近傍の場合
    if (old_sq != SQ_NB && dist(sq_king, old_sq) <= 2) {
      dirty_bb |= old_sq;

      Effect24::Direct dir = CalcDirect(sq_king, old_sq);

      removed->push_back(MakeIndex(perspective, dir, old_pc
          , GetEffectCount(pos, old_sq, perspective, true)
          , GetEffectCount(pos, old_sq, ~perspective, true)
        ));

      // iが0以外の場合（iが1の場合）は取られた駒なので除外
      if (i == 0) {
        added->push_back(MakeIndex(perspective, dir, NO_PIECE
            , GetEffectCount(pos, old_sq, perspective, false)
            , GetEffectCount(pos, old_sq, ~perspective, false)
          ));
      }
    }

    // new_piece（先手目線）
    const auto new_piece = static_cast<BonaPiece>(dp.changed_piece[i].new_piece.from[BLACK]);
    Square new_sq;
    Piece new_pc;
    GetSquarePieceFromBonaPiece(new_piece, new_sq, new_pc);

    // new_pieceのマスが玉の24近傍の場合
    if (new_sq != SQ_NB && dist(sq_king, new_sq) <= 2) {
      dirty_bb |= new_sq;

      Effect24::Direct dir = CalcDirect(sq_king, new_sq);

      // 「dp.dirty_num == 2 && i == 0)」の場合は除外
      if (   (dp.dirty_num == 1 && i == 0)
          || (dp.dirty_num == 2 && i == 1)) {
        removed->push_back(MakeIndex(perspective, dir, NO_PIECE
            , GetEffectCount(pos, new_sq, perspective, true)
            , GetEffectCount(pos, new_sq, ~perspective, true)
          ));
      }

      added->push_back(MakeIndex(perspective, dir, new_pc
          , GetEffectCount(pos, new_sq, perspective, false)
          , GetEffectCount(pos, new_sq, ~perspective, false)
        ));
    }
  }
#if defined(ENABLE_NNUE_BENCH)
  std::chrono::steady_clock::time_point dirty_end;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    dirty_end = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point neighbor_begin;
  if (kKsdg3BenchmarkStageTiming != nullptr)
    neighbor_begin = std::chrono::steady_clock::now();
#endif

  // 24近傍をループ
  for (Effect24::Direct dir : Effect24::Direct()) {
    SquareWithWall sqww = sqww_king + DirectToDeltaWW(dir);

    // 盤内の場合
    if (is_ok(sqww)) {
      Square sq = sqww_to_sq(sqww);

#if defined(ENABLE_NNUE_BENCH)
      if (kKsdg3BenchmarkStageTiming != nullptr)
        ++kKsdg3BenchmarkStageTiming->valid_neighbors;
#endif

      // dirtyな場合は既に処理済み
      if (dirty_bb & sq) {
        continue;
      }

      int effectCount_prev_1 = GetEffectCount(pos, sq, perspective, true);
      int effectCount_prev_2 = GetEffectCount(pos, sq, ~perspective, true);
      int effectCount_now_1 = GetEffectCount(pos, sq, perspective, false);
      int effectCount_now_2 = GetEffectCount(pos, sq, ~perspective, false);

      // 利き数に変化があった場合
      if (   effectCount_prev_1 != effectCount_now_1
          || effectCount_prev_2 != effectCount_now_2) {
#if defined(ENABLE_NNUE_BENCH)
        if (kKsdg3BenchmarkStageTiming != nullptr)
          ++kKsdg3BenchmarkStageTiming->capped_changed_squares;
#endif
        Piece pc = pos.piece_on(sq);
        removed->push_back(MakeIndex(perspective, dir, pc, effectCount_prev_1, effectCount_prev_2));
        added->push_back(MakeIndex(perspective, dir, pc, effectCount_now_1, effectCount_now_2));
      }
    }

    // 盤外の場合、何もしない

  }
#if defined(ENABLE_NNUE_BENCH)
  if (kKsdg3BenchmarkStageTiming != nullptr) {
    const auto neighbor_end = std::chrono::steady_clock::now();
    ++kKsdg3BenchmarkStageTiming->calls;
    kKsdg3BenchmarkStageTiming->dirty_nanoseconds +=
        std::chrono::duration<double, std::nano>(dirty_end - dirty_begin).count();
    kKsdg3BenchmarkStageTiming->neighbor_nanoseconds +=
        std::chrono::duration<double, std::nano>(neighbor_end - neighbor_begin).count();
  }
#endif
}

template class KingSafety3_DistinguishGolds<Side::kFriend>;
template class KingSafety3_DistinguishGolds<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
}  // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
