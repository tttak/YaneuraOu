// Diagnostic-only frozen policy probe shared by NNUE propagation and search.
#ifndef YANEURAOU_NNUE_POLICY_PROBE_H_INCLUDED
#define YANEURAOU_NNUE_POLICY_PROBE_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_POLICY_SHADOW)

#include "nnue_policy_probe_weights.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace YaneuraOu::Eval::NNUE::PolicyProbe {

template<int Width>
inline void ProjectQuery(const std::uint8_t* l2_input, float* output) {
    const auto& weights = FrozenWeights<Width>::query_weight;
    const auto& bias = FrozenWeights<Width>::query_bias;
    constexpr float kInverseL2Scale = 1.0f / 127.0f;
    for (int row = 0; row < Width; ++row) {
        float value = bias[row];
        const float* weight = weights.data() + row * 128;
        for (int column = 0; column < 128; ++column)
            value += weight[column]
                   * (static_cast<float>(l2_input[column]) * kInverseL2Scale);
        output[row] = value;
    }
}

template<int Width>
inline float Score(const float* query, const std::array<int, 20>& feature) {
    const auto& offsets = FrozenWeights<Width>::offsets;
    const auto& embedding = FrozenWeights<Width>::embedding;
    float result = 0.0f;
    for (int dimension = 0; dimension < Width; ++dimension) {
        float encoded = 0.0f;
        for (int kind = 0; kind < 20; ++kind)
            encoded += embedding[(offsets[kind] + feature[kind]) * Width + dimension];
        result += query[dimension] * encoded;
    }
    return result * (1.0f / std::sqrt(static_cast<float>(Width)));
}

}  // namespace YaneuraOu::Eval::NNUE::PolicyProbe

#endif
#endif
