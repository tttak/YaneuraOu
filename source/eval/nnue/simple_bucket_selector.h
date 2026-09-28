#ifndef NNUE_SIMPLE_BUCKET_SELECTOR_H_INCLUDED
#define NNUE_SIMPLE_BUCKET_SELECTOR_H_INCLUDED

#include "../../position.h"

namespace YaneuraOu::Eval::NNUE::SimpleBucket {

inline int promoted_count(const Position& pos) {
    return (pos.pieces(PRO_PAWN) | pos.pieces(PRO_LANCE)
            | pos.pieces(PRO_KNIGHT) | pos.pieces(PRO_SILVER)
            | pos.pieces(HORSE) | pos.pieces(DRAGON)).pop_count();
}

inline int hand_piece_count(const Position& pos) {
    int n = 0;
    for (Color c : {BLACK, WHITE})
        for (PieceType pt : {PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK})
            n += hand_count(pos.hand_of(c), pt);
    return n;
}

inline int major_hand_count(const Position& pos) {
    int n = 0;
    for (Color c : {BLACK, WHITE})
        n += hand_count(pos.hand_of(c), BISHOP)
           + hand_count(pos.hand_of(c), ROOK);
    return n;
}

inline int phase_score(const Position& pos) {
    constexpr int values[] = {0, 100, 300, 300, 500, 800, 1000, 600};
    int score = 200 * promoted_count(pos);
    for (Color c : {BLACK, WHITE})
        for (PieceType pt : {PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK})
            score += values[static_cast<int>(pt)]
                   * hand_count(pos.hand_of(c), pt);
    return score;
}

inline int k3k3(const Position& pos) {
    constexpr int f[] = {0,0,0,3,3,3,6,6,6};
    constexpr int e[] = {0,0,0,1,1,1,2,2,2};
    const Color stm = pos.side_to_move();
    const Square fk = pos.square<KING>(stm);
    const Square ek = pos.square<KING>(~stm);
    const int fr = stm == BLACK ? rank_of(fk) : rank_of(Inv(fk));
    const int er = stm == BLACK ? rank_of(Inv(ek)) : rank_of(ek);
    return f[fr] + e[er];
}

inline int phase9(const Position& pos) {
    constexpr int t[] = {200,800,1600,2000,2500,3000,3700,4400};
    const int score = phase_score(pos);
    int b = 0;
    while (b < 8 && score >= t[b]) ++b;
    return b;
}

inline int kingfree_tree(const Position& pos) {
    const int promoted = promoted_count(pos);
    const int hand = hand_piece_count(pos);
    if (promoted <= 0) {
        if (hand <= 2) {
            if (hand <= 0) return 0;
            return major_hand_count(pos) <= 0 ? 1 : 4;
        }
        if (hand <= 6) return hand <= 4 ? 2 : 3;
        return 6;
    }
    if (promoted <= 1) return hand <= 7 ? 5 : 7;
    return 8;
}

inline int selected(const Position& pos) {
#if defined(NNUE_SIMPLE_BUCKET_PHASE9)
    return phase9(pos);
#elif defined(NNUE_SIMPLE_BUCKET_KINGFREE_TREE)
    return kingfree_tree(pos);
#else
    return k3k3(pos);
#endif
}

} // namespace YaneuraOu::Eval::NNUE::SimpleBucket
#endif
