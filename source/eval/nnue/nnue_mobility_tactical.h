#pragma once

// Experiment 84, phase 1: cheap aggregated mobility/tactical availability.
//
// This is diagnostic-only.  It deliberately does not generate legal moves and
// does not consider pins, recaptures, defenders, SEE, or exchange value.

#include <array>
#include <cstdint>

namespace YaneuraOu::NnueMobilityTactical {

enum Feature : std::size_t {
  UsRookMobility,
  ThemRookMobility,
  UsBishopMobility,
  ThemBishopMobility,
  UsCaptureCount,
  ThemCaptureCount,
  UsMaxCapturableBucket,
  ThemMaxCapturableBucket,
  FeatureCount,
};

struct RawFeatures {
  std::array<std::uint16_t, FeatureCount> value{};
};

inline std::array<float, FeatureCount> normalize(const RawFeatures& raw) {
  constexpr std::array<float, FeatureCount> scale =
      {{20.0f, 20.0f, 16.0f, 16.0f, 8.0f, 8.0f, 6.0f, 6.0f}};
  std::array<float, FeatureCount> normalized{};
  for (std::size_t i = 0; i < normalized.size(); ++i)
    normalized[i] = std::min(float(raw.value[i]) / scale[i], 1.0f);
  return normalized;
}

struct CaptureAvailability {
  std::uint16_t unique_targets = 0;
  std::uint16_t relations = 0;  // diagnostic alternative only
  std::uint8_t max_bucket = 0;
};

using CapturePair = std::array<CaptureAvailability, COLOR_NB>;

inline std::uint8_t capturable_value_bucket(const PieceType pt) {
  switch (pt) {
  case PAWN: return 1;
  case LANCE:
  case KNIGHT: return 2;
  case SILVER:
  case GOLD:
  case PRO_PAWN:
  case PRO_LANCE:
  case PRO_KNIGHT:
  case PRO_SILVER: return 3;
  case BISHOP: return 4;
  case ROOK: return 5;
  case HORSE:
  case DRAGON: return 6;
  default: return 0;  // King and invalid pieces are not capture targets.
  }
}

inline std::uint16_t rook_mobility(const Position& pos, const Color side,
                                   const bool include_dragon) {
  const Bitboard occupied = pos.pieces();
  const Bitboard own = pos.pieces(side);
  Bitboard sliders = include_dragon ? pos.pieces(side, ROOK_DRAGON)
                                    : pos.pieces(side, ROOK);
  std::uint16_t count = 0;
  while (sliders) {
    const Square square = sliders.pop();
    count += static_cast<std::uint16_t>(
        (rookEffect(square, occupied) & ~own).pop_count());
  }
  return count;
}

inline std::uint16_t bishop_mobility(const Position& pos, const Color side,
                                     const bool include_horse) {
  const Bitboard occupied = pos.pieces();
  const Bitboard own = pos.pieces(side);
  Bitboard sliders = include_horse ? pos.pieces(side, BISHOP_HORSE)
                                   : pos.pieces(side, BISHOP);
  std::uint16_t count = 0;
  while (sliders) {
    const Square square = sliders.pop();
    count += static_cast<std::uint16_t>(
        (bishopEffect(square, occupied) & ~own).pop_count());
  }
  return count;
}

inline CaptureAvailability capture_availability(
    const Position& pos, const Color side,
    const bool count_relations = false) {
  CaptureAvailability result{};
  Bitboard targets = pos.pieces(~side);
  while (targets) {
    const Square target = targets.pop();
    const PieceType pt = type_of(pos.piece_on(target));
    if (pt == KING)
      continue;

    // effected_to() uses LongEffect's already-maintained effect count in the
    // production engine build.  The fallback implementation uses attackers_to.
    if (!pos.effected_to(side, target))
      continue;

    ++result.unique_targets;
    if (count_relations) {
      const unsigned relations =
          static_cast<unsigned>(pos.attackers_to(side, target).pop_count());
      result.relations += static_cast<std::uint16_t>(relations);
    }
    const auto bucket = capturable_value_bucket(pt);
    if (bucket > result.max_bucket)
      result.max_bucket = bucket;
  }
  return result;
}

// Compute both colors in one board-piece pass.  This is the v1 production
// candidate used by extract(); capture_availability() remains useful for the
// relation-count diagnostic and focused tests.
inline CapturePair capture_pair(const Position& pos) {
  CapturePair result{};
  Bitboard targets = pos.pieces();
  while (targets) {
    const Square target = targets.pop();
    const Piece piece = pos.piece_on(target);
    const PieceType pt = type_of(piece);
    if (pt == KING)
      continue;
    const Color attacker = ~color_of(piece);
    if (!pos.effected_to(attacker, target))
      continue;
    auto& availability = result[attacker];
    ++availability.unique_targets;
    const auto bucket = capturable_value_bucket(pt);
    if (bucket > availability.max_bucket)
      availability.max_bucket = bucket;
  }
  return result;
}

inline RawFeatures extract(const Position& pos) {
  RawFeatures result{};
  const Color us = pos.side_to_move();
  const Color them = ~us;

  result.value[UsRookMobility] = rook_mobility(pos, us, true);
  result.value[ThemRookMobility] = rook_mobility(pos, them, true);
  result.value[UsBishopMobility] = bishop_mobility(pos, us, true);
  result.value[ThemBishopMobility] = bishop_mobility(pos, them, true);
  // Two color-specific target scans benchmark faster than the branchier
  // all-piece capture_pair() on the current AVX2 build.
  const auto us_capture = capture_availability(pos, us);
  const auto them_capture = capture_availability(pos, them);
  result.value[UsCaptureCount] = us_capture.unique_targets;
  result.value[ThemCaptureCount] = them_capture.unique_targets;
  result.value[UsMaxCapturableBucket] = us_capture.max_bucket;
  result.value[ThemMaxCapturableBucket] = them_capture.max_bucket;
  return result;
}

}  // namespace YaneuraOu::NnueMobilityTactical
