// Experiment 53のfrozen context-only LMR probeをproduction相当buildで評価する。
// 探索側から明示的に呼ばない限り、評価値や探索挙動には影響しない。
#ifndef YANEURAOU_NNUE_DECISION_RISK_PREDICTOR_H_INCLUDED
#define YANEURAOU_NNUE_DECISION_RISK_PREDICTOR_H_INCLUDED

#include "../../config.h"

#if defined(USE_NNUE_DECISION_RISK_LMR)

#include "nnue_decision_risk_weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace YaneuraOu::Search::NnueDecisionRiskPredictor {

#ifndef NNUE_DECISION_RISK_LMR_Q8_THRESHOLD
#error "USE_NNUE_DECISION_RISK_LMR requires NNUE_DECISION_RISK_LMR_Q8_THRESHOLD"
#endif
#if NNUE_DECISION_RISK_LMR_Q8_THRESHOLD < 0 || NNUE_DECISION_RISK_LMR_Q8_THRESHOLD > 255
#error "NNUE_DECISION_RISK_LMR_Q8_THRESHOLD must be in [0,255]"
#endif

struct Input {
    int depth = 0;
    int ply = 0;
    int static_eval = 0;
    int alpha = 0;
    int beta = 0;
    int material = 0;
    int move_count = 0;
    int reduction = 0;
    int history = 0;
    int parent_eval = 0;
    int grandparent_eval = 0;
    int great_grandparent_eval = 0;
    int node_kind = 0;
    int bucket = -1;
    bool improving = false;
    bool in_check = false;
    bool capture = false;
    bool check = false;
    bool tt_move = false;
};

inline bool ValidEval(const int value) {
    return value != static_cast<int>(VALUE_NONE) && std::abs(value) < 30000;
}

inline std::uint8_t LmrQ8(const Input& input) {
    using namespace NnueDecisionRiskWeights;
    std::array<float, Dimensions> x{};
    const auto clipped = [](const int value, const int lo, const int hi) {
        return std::clamp(value, lo, hi);
    };
    x[0] = clipped(input.depth, -2, 32) / 16.0f;
    x[1] = clipped(input.ply, 0, 256) / 128.0f;
    x[2] = clipped(input.static_eval, -4000, 4000) / 2000.0f;
    x[3] = clipped(input.alpha, -4000, 4000) / 2000.0f;
    x[4] = clipped(input.beta, -4000, 4000) / 2000.0f;
    x[5] = clipped(input.static_eval - input.alpha, -4000, 4000) / 2000.0f;
    x[6] = clipped(input.static_eval - input.beta, -4000, 4000) / 2000.0f;
    x[7] = clipped(std::abs(input.material), 0, 8000) / 4000.0f;
    x[8] = input.improving ? 1.0f : 0.0f;
    x[9] = input.in_check ? 1.0f : 0.0f;
    x[10] = clipped(input.move_count, 0, 64) / 32.0f;
    x[11] = clipped(input.reduction, -4, 16) / 8.0f;
    x[12] = clipped(input.history, -20000, 20000) / 10000.0f;

    const bool parent_valid = ValidEval(input.parent_eval);
    const bool grandparent_valid = ValidEval(input.grandparent_eval);
    const bool great_grandparent_valid = ValidEval(input.great_grandparent_eval);
    const int parent = -input.parent_eval;
    const int grandparent = input.grandparent_eval;
    const int great_grandparent = -input.great_grandparent_eval;
    x[13] = clipped(parent_valid ? std::abs(input.static_eval - parent) : 0, 0, 4000) / 1000.0f;
    x[14] = clipped(grandparent_valid ? std::abs(input.static_eval - grandparent) : 0, 0, 4000) / 1000.0f;
    x[15] = clipped(great_grandparent_valid
                      ? std::abs(input.static_eval - great_grandparent) : 0,
                    0, 4000) / 1000.0f;
    x[16] = input.capture ? 1.0f : 0.0f;
    x[17] = input.check ? 1.0f : 0.0f;
    x[18] = input.tt_move ? 1.0f : 0.0f;
    const int d01 = input.static_eval - parent;
    const int d12 = parent - grandparent;
    const int d23 = grandparent - great_grandparent;
    const bool oscillation2 = parent_valid && grandparent_valid && d01 != 0 && d12 != 0
                           && std::int64_t(d01) * d12 < 0;
    const bool oscillation3 = oscillation2 && great_grandparent_valid && d23 != 0
                           && std::int64_t(d12) * d23 < 0;
    x[19] = oscillation2 ? 1.0f : 0.0f;
    x[20] = oscillation3 ? 1.0f : 0.0f;
    x[21 + std::clamp(input.node_kind, 0, 4)] = 1.0f;
    if (input.bucket >= 0 && input.bucket < 12)
        x[26 + input.bucket] = 1.0f;

    float logit = LMRBias;
    for (std::size_t index = 0; index < x.size(); ++index)
        logit += x[index] * LMRWeight[index];
    const float probability = 1.0f / (1.0f + std::exp(-logit));
    return std::uint8_t(std::clamp(std::lround(probability * 255.0f), 0L, 255L));
}

inline bool HighRisk(const Input& input) {
    return LmrQ8(input) >= NNUE_DECISION_RISK_LMR_Q8_THRESHOLD;
}

}  // namespace YaneuraOu::Search::NnueDecisionRiskPredictor

#endif
#endif
