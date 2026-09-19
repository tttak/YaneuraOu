// Diagnostic-only NNUE signals used to measure search predictiveness.
// This header deliberately contributes no declarations to production builds.

#ifndef CLASSIC_NNUE_SIGNAL_H_INCLUDED
#define CLASSIC_NNUE_SIGNAL_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_ROUTER_LMR)

#include <cstddef>
#include <cstdint>

namespace YaneuraOu::Eval::NNUE {

#if defined(USE_NNUE_ROUTER_LMR)
// Production experiment payload: only the Router margin needed by LMR.
// Keeping this separate from NnueSignalSnapshot avoids carrying the diagnostic
// phase/deep/bypass fields, counters, and report machinery in match binaries.
struct NnueRouterLmrSignal {
    std::int32_t router_margin = 0;
#if defined(USE_NNUE_DECISION_RISK_LMR)
    // Experiment 53 context probeのbucket one-hot用。評価値には影響しない。
    std::int8_t selected_bucket = -1;
#endif
#if defined(USE_NNUE_LCA_LMR)
    // Exact sum of the absolute byte-domain LCA correction over 32 channels.
    // The 295/epoch20 top-1% experiment compares this integer directly.
    std::int32_t lca_abs_delta_sum = 0;
#endif
#if defined(USE_NNUE_CROSS_LMR)
    // Maximum of the 32 uint8 Cross activations, used by Cross-LMR.
    std::uint8_t cross_abs_max = 0;
#endif
#if defined(USE_NNUE_PHASE_FM_LMR)
    float fm_reliance = 0.0f;
#endif
    bool valid = false;
};

inline thread_local NnueRouterLmrSignal g_last_nnue_router_lmr_signal{};

inline void SetLastNnueRouterLmrSignal(const NnueRouterLmrSignal* signal = nullptr) {
    g_last_nnue_router_lmr_signal = signal ? *signal : NnueRouterLmrSignal{};
}

inline const NnueRouterLmrSignal& LastNnueRouterLmrSignal() {
    return g_last_nnue_router_lmr_signal;
}
#endif

#if defined(ENABLE_NNUE_SIGNAL_LOG)
struct NnueSignalSnapshot {
    std::int32_t deep_output = 0;
    std::int32_t bypass_output = 0;
    std::int32_t signed_deep_bypass = 0;
    std::int32_t deep_bypass_disagreement = 0;

    std::int32_t selected_bucket = 0;
    std::int32_t router_top1_logit = 0;
    std::int32_t router_top2_logit = 0;
    std::int32_t router_margin = 0;

    float phase_scale[6]{};
    float main_reliance = 0.0f;
    float fm_reliance = 0.0f;
    float cross_reliance = 0.0f;
    // Existing Diff RMSNorm reduction result. This is the sum of squares of
    // the 32 value channels before sqrt/division; diagnostics reuse it without
    // another pass over diff_fc_out.
    float diff_rms_energy_sum = 0.0f;
    // Cross activation is uint8_t, so abs(value) == value. Both summaries are
    // collected from the completed 32-byte output without changing its values.
    std::uint16_t cross_abs_sum = 0;
    std::uint8_t cross_abs_max = 0;
    // Main gate is an integer Q64 sigmoid in [0,63].  Diagnostics aggregate
    // these values in the existing 32-channel generation loop.
    std::uint16_t main_gate_sum = 0;
    std::uint8_t main_gate_min = 0;
    std::uint8_t main_gate_max = 0;
    std::uint8_t main_gate_saturated_low_count = 0;   // gate <= 1
    std::uint8_t main_gate_saturated_high_count = 0;  // gate >= 63
    // FM activations are uint8 [0,127]. Diff is a signed-like signal stored
    // around neutral 64, so its activity is abs(q-64); Abs activity is q.
    std::uint16_t fm_diff_activity_sum = 0;
    std::uint8_t fm_diff_activity_max = 0;
    std::uint8_t fm_diff_saturated_count = 0;
    std::uint16_t fm_abs_activity_sum = 0;
    std::uint8_t fm_abs_activity_max = 0;
    std::uint8_t fm_abs_saturated_count = 0;
    float lca_mean_abs_delta = 0.0f;
    std::int32_t lca_max_abs_delta = 0;
    std::int32_t lca_abs_delta_sum = 0;
#if defined(ENABLE_NNUE_POLICY_SHADOW)
    // Frozen Experiment-51 L2->query projections. Move scoring remains in the
    // shadow logger and never changes MovePicker ordering or search decisions.
    float policy_query16[16]{};
    float policy_query32[32]{};
    std::uint32_t policy_query16_projection_ns = 0;
    std::uint32_t policy_query32_projection_ns = 0;
#endif
#if defined(ENABLE_NNUE_UNCERTAINTY_SIGNAL)
    // Frozen DLS/Fuka disagreement probe.  These fields are diagnostic only;
    // uncertainty_q8 is the scalar intended for later search logging.
    float uncertainty_logit = 0.0f;
    float uncertainty_probability = 0.0f;
    std::uint8_t uncertainty_q8 = 0;
#endif
#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL)
    // Diagnostic-only predictions of Hao static-vs-depth9 abs-delta-probability.
    // Context uses this engine's static score in place of the Hao HalfKP score;
    // that substitution is intentionally tracked as a distribution shift.
    enum HaoRiskKind : std::size_t {
        HaoFc1Only = 0,
        HaoContextOnly,
        HaoContextFc1,
        HaoResidualPositive,
        HaoRiskCount
    };
    float hao_risk_logit[HaoRiskCount]{};
    float hao_risk_probability[HaoRiskCount]{};
    std::uint8_t hao_risk_q8[HaoRiskCount]{};
    std::uint32_t hao_risk_compute_ns = 0;
    std::int16_t hao_context_static_eval = 0;
    std::int16_t hao_context_material = 0;
    std::uint16_t hao_context_game_ply = 0;
#endif
    bool valid = false;
};

enum class NnueSignalEvalSource : std::uint8_t {
    FreshNetwork,
    AccumulatorCached,
    EvalHashHit,
    TtEvalReuse,
    InCheckSkipped,
    Unavailable,
    Count
};

struct NnueSignalEvalAccess {
    NnueSignalEvalSource source = NnueSignalEvalSource::Unavailable;
    NnueSignalSnapshot signal{};
};

// Search and evaluation run on the same worker thread. Keeping only the last
// access avoids synchronization in the evaluation hot path; the search logger
// copies it immediately after Eval::evaluate() returns.
inline thread_local NnueSignalEvalAccess g_last_nnue_signal_access{};

inline void SetLastNnueSignalAccess(const NnueSignalEvalSource source,
                                    const NnueSignalSnapshot* signal = nullptr) {
    g_last_nnue_signal_access.source = source;
    if (signal)
        g_last_nnue_signal_access.signal = *signal;
    else
        g_last_nnue_signal_access.signal = NnueSignalSnapshot{};
}

inline const NnueSignalEvalAccess& LastNnueSignalAccess() {
    return g_last_nnue_signal_access;
}
#endif

}  // namespace YaneuraOu::Eval::NNUE

#endif  // defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_ROUTER_LMR)
#endif  // CLASSIC_NNUE_SIGNAL_H_INCLUDED
