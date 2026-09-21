#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace YaneuraOu::NnuePairRelation {

constexpr std::size_t PieceTypeCount = 14;
constexpr std::size_t OwnerDirectionCount = 4;
constexpr std::size_t RelationTypeCount =
    PieceTypeCount * PieceTypeCount * OwnerDirectionCount;
constexpr std::size_t MaxRelations = 40 * 39;

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

inline std::uint16_t relation_index(const Piece attacker, const Piece target,
                                    const Color us) {
    const int a = compact_piece_type(type_of(attacker));
    const int t = compact_piece_type(type_of(target));
    const Color ac = color_of(attacker);
    const Color tc = color_of(target);
    const int direction = ac == us ? (tc == us ? 0 : 1)
                                   : (tc == us ? 3 : 2);
    return static_cast<std::uint16_t>(
        a + PieceTypeCount * (t + PieceTypeCount * direction));
}

inline std::size_t generate(const Position& pos,
                            std::uint16_t* const output,
                            const std::size_t capacity) {
    const Color us = pos.side_to_move();
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
                output[count] = relation_index(
                    attacker, pos.piece_on(to), us);
            ++count;
        }
    }
    return count;
}

}  // namespace YaneuraOu::NnuePairRelation
