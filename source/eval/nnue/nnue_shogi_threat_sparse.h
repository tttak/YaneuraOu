#pragma once

// Experiment 83: compact shogi threat sparse features.
//
// This header intentionally contains only the reference feature definition and
// an oracle multiset-diff accumulator prototype.  It is not part of the
// production NNUE architecture.  A later implementation can feed the same
// added/removed indices from Position::do_move() without changing the feature
// index contract established here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace YaneuraOu::NnueShogiThreatSparse {

constexpr std::size_t PieceTypeCount = 14;
constexpr std::size_t OwnerDirectionCount = 4;
constexpr std::size_t GeometryCount = 6;
constexpr std::size_t FeatureCount =
    PieceTypeCount * PieceTypeCount * OwnerDirectionCount * GeometryCount;
constexpr std::size_t PrototypeWidth = 32;
constexpr std::size_t MaxRelations = 40 * 39;

enum class Geometry : std::uint8_t {
  Adjacent = 0,
  KnightLike = 1,
  SameFile = 2,
  SameRank = 3,
  SameDiagonal = 4,
  Other = 5,
};

struct Edge {
  Square from;
  Square to;
  Piece attacker;
  Piece target;
  std::uint16_t index;
};

inline std::uint32_t logical_edge_key(const Edge& edge) {
  return std::uint32_t(edge.from)
       | (std::uint32_t(edge.to) << 7)
       | (std::uint32_t(edge.attacker) << 14)
       | (std::uint32_t(edge.target) << 19);
}

inline int compact_piece_type(const PieceType pt) {
  switch (pt) {
  case PAWN:       return 0;
  case LANCE:      return 1;
  case KNIGHT:     return 2;
  case SILVER:     return 3;
  case GOLD:       return 4;
  case BISHOP:     return 5;
  case ROOK:       return 6;
  case PRO_PAWN:   return 7;
  case PRO_LANCE:  return 8;
  case PRO_KNIGHT: return 9;
  case PRO_SILVER: return 10;
  case HORSE:      return 11;
  case DRAGON:     return 12;
  case KING:       return 13;
  default:         return -1;
  }
}

inline Geometry geometry(const Square from, const Square to,
                         const Color perspective) {
  int dx = int(file_of(to)) - int(file_of(from));
  int dy = int(rank_of(to)) - int(rank_of(from));
  // Directional candidates added later must normalize White.  The six-bucket
  // v1 contract is symmetric, but normalize here so the convention is fixed.
  if (perspective == WHITE) {
    dx = -dx;
    dy = -dy;
  }
  const int ax = std::abs(dx);
  const int ay = std::abs(dy);
  if (std::max(ax, ay) == 1)
    return Geometry::Adjacent;
  if ((ax == 1 && ay == 2) || (ax == 2 && ay == 1))
    return Geometry::KnightLike;
  if (dx == 0)
    return Geometry::SameFile;
  if (dy == 0)
    return Geometry::SameRank;
  if (ax == ay)
    return Geometry::SameDiagonal;
  return Geometry::Other;
}

inline std::uint16_t feature_index(const Piece attacker, const Piece target,
                                   const Square from, const Square to,
                                   const Color perspective) {
  const int attacker_type = compact_piece_type(type_of(attacker));
  const int target_type = compact_piece_type(type_of(target));
  const Color attacker_color = color_of(attacker);
  const Color target_color = color_of(target);
  const int owner_direction =
      attacker_color == perspective
          ? (target_color == perspective ? 0 : 1)
          : (target_color == perspective ? 3 : 2);
  const int geometry_index = static_cast<int>(geometry(from, to, perspective));
  return static_cast<std::uint16_t>(
      attacker_type + PieceTypeCount *
          (target_type + PieceTypeCount *
              (owner_direction + OwnerDirectionCount * geometry_index)));
}

// Geometry6 is invariant under the BLACK/WHITE 180-degree normalization.
// Produce both fixed-perspective indices with one piece/geometry decode; the
// WHITE owner-direction is exactly BLACK xor 2.
inline void feature_indices(const Piece attacker, const Piece target,
                            const Square from, const Square to,
                            std::uint16_t& black, std::uint16_t& white) {
  const int attacker_type = compact_piece_type(type_of(attacker));
  const int target_type = compact_piece_type(type_of(target));
  const int black_owner = color_of(attacker) == BLACK
      ? (color_of(target) == BLACK ? 0 : 1)
      : (color_of(target) == BLACK ? 3 : 2);
  const int geometry_index = static_cast<int>(geometry(from, to, BLACK));
  const int common = attacker_type + PieceTypeCount *
      (target_type + PieceTypeCount * OwnerDirectionCount * geometry_index);
  black = static_cast<std::uint16_t>(
      common + PieceTypeCount * PieceTypeCount * black_owner);
  white = static_cast<std::uint16_t>(
      common + PieceTypeCount * PieceTypeCount * (black_owner ^ 2));
}

inline std::size_t generate_edges(const Position& pos, const Color perspective,
                                  Edge* const output,
                                  const std::size_t capacity) {
  const Bitboard occupied = pos.pieces();
  Bitboard attackers = occupied;
  std::size_t count = 0;
  while (attackers) {
    const Square from = attackers.pop();
    const Piece attacker = pos.piece_on(from);
    Bitboard targets = effects_from(attacker, from, occupied) & occupied;
    while (targets) {
      const Square to = targets.pop();
      if (count < capacity)
        output[count] = {from, to, attacker, pos.piece_on(to), feature_index(
            attacker, pos.piece_on(to), from, to, perspective)};
      ++count;
    }
  }
  return count;
}

inline std::vector<std::uint16_t> sorted_indices(
    const Edge* const edges, const std::size_t count) {
  std::vector<std::uint16_t> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    result.push_back(edges[i].index);
  std::sort(result.begin(), result.end());
  return result;
}

// Multiset difference is required: the compact 4704-dimensional contract can
// map more than one board edge to the same sparse row.
inline void multiset_delta(const std::vector<std::uint16_t>& before,
                           const std::vector<std::uint16_t>& after,
                           std::vector<std::uint16_t>& removed,
                           std::vector<std::uint16_t>& added) {
  removed.clear();
  added.clear();
  std::set_difference(before.begin(), before.end(), after.begin(), after.end(),
                      std::back_inserter(removed));
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(added));
}

// Deterministic pseudo weights make the accumulator prototype independent of
// an nn.bin and expose ordering/count bugs (including duplicate indices).
inline float prototype_weight(const std::uint16_t index,
                              const std::size_t channel) {
  std::uint32_t x = std::uint32_t(index) * 0x9e3779b9u
                  ^ std::uint32_t(channel + 1) * 0x85ebca6bu;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return (float(int(x & 0xffffu) - 32768) / 32768.0f) * 0.03125f;
}

using PrototypeAccumulator = std::array<float, PrototypeWidth>;

inline void refresh_accumulator(const std::vector<std::uint16_t>& active,
                                PrototypeAccumulator& accumulator) {
  accumulator.fill(0.0f);
  for (const auto index : active)
    for (std::size_t channel = 0; channel < PrototypeWidth; ++channel)
      accumulator[channel] += prototype_weight(index, channel);
}

inline void update_accumulator(const std::vector<std::uint16_t>& removed,
                               const std::vector<std::uint16_t>& added,
                               PrototypeAccumulator& accumulator) {
  for (const auto index : removed)
    for (std::size_t channel = 0; channel < PrototypeWidth; ++channel)
      accumulator[channel] -= prototype_weight(index, channel);
  for (const auto index : added)
    for (std::size_t channel = 0; channel < PrototypeWidth; ++channel)
      accumulator[channel] += prototype_weight(index, channel);
}

}  // namespace YaneuraOu::NnueShogiThreatSparse
