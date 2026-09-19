// Experiment 64 diagnostic-only frozen qsearch-correction probes.
#ifndef CLASSIC_NNUE_QSEARCH_CORRECTION_SHADOW_H_INCLUDED
#define CLASSIC_NNUE_QSEARCH_CORRECTION_SHADOW_H_INCLUDED

#include "qsearch_correction_shadow_weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace YaneuraOu::Eval::NNUE::QsearchCorrectionShadow {

struct Prediction {
    float linear_probability[2]{};
    float mlp32_probability[2]{};
    std::uint8_t linear_q8[2]{};
    std::uint8_t mlp32_q8[2]{};
};
using Input = std::array<float, QsearchCorrectionShadowWeights::InputSize>;

inline float Sigmoid(const float value) {
    return 1.0f / (1.0f + std::exp(-std::clamp(value, -30.0f, 30.0f)));
}

inline std::uint8_t ToQ8(const float probability) {
    return static_cast<std::uint8_t>(std::clamp(
      static_cast<int>(std::lround(probability * 255.0f)), 0, 255));
}

inline Input MakeInput(const std::uint8_t* l2_input, const int static_eval,
                       const int game_ply, const int material_stm,
                       const int in_check, const int bucket) {
    Input input{};
    input[0] = static_cast<float>(static_eval);
    input[1] = static_cast<float>(std::abs(static_eval));
    input[2] = static_cast<float>(game_ply);
    input[3] = static_cast<float>(material_stm);
    input[4] = static_cast<float>(in_check);
    input[5 + std::clamp(bucket, 0, 11)] = 1.0f;
    for (std::size_t i = 0; i < 128; ++i)
        input[17 + i] = static_cast<float>(l2_input[i]) / 127.0f;
    return input;
}

inline void PredictLinear(const Input& input, Prediction& result) {
    namespace W = QsearchCorrectionShadowWeights;
    for (std::size_t output = 0; output < W::OutputSize; ++output) {
        float logit = W::LinearBias[output];
        for (std::size_t i = 0; i < W::InputSize; ++i)
            logit += W::LinearWeight[output * W::InputSize + i] * input[i];
        result.linear_probability[output] = Sigmoid(logit);
        result.linear_q8[output] = ToQ8(result.linear_probability[output]);
    }
}

inline void PredictMlp32(const Input& input, Prediction& result) {
    namespace W = QsearchCorrectionShadowWeights;
    std::array<float, W::HiddenSize> hidden{};
    for (std::size_t h = 0; h < W::HiddenSize; ++h) {
        float value = W::MlpInputBias[h];
        for (std::size_t i = 0; i < W::InputSize; ++i)
            value += W::MlpInputWeight[h * W::InputSize + i] * input[i];
        hidden[h] = std::max(value, 0.0f);
    }
    for (std::size_t output = 0; output < W::OutputSize; ++output) {
        float logit = W::MlpOutputBias[output];
        for (std::size_t h = 0; h < W::HiddenSize; ++h)
            logit += W::MlpOutputWeight[output * W::HiddenSize + h] * hidden[h];
        result.mlp32_probability[output] = Sigmoid(logit);
        result.mlp32_q8[output] = ToQ8(result.mlp32_probability[output]);
    }
}

inline Prediction Predict(const std::uint8_t* l2_input, const int static_eval,
                          const int game_ply, const int material_stm,
                          const int in_check, const int bucket) {
    const auto input = MakeInput(l2_input, static_eval, game_ply, material_stm, in_check, bucket);
    Prediction result{};
    PredictLinear(input, result);
    PredictMlp32(input, result);
    return result;
}

}  // namespace YaneuraOu::Eval::NNUE::QsearchCorrectionShadow
#endif
