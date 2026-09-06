// Diagnostic-only NNUE signals used to measure search predictiveness.
// This header deliberately contributes no declarations to production builds.

#ifndef CLASSIC_NNUE_SIGNAL_H_INCLUDED
#define CLASSIC_NNUE_SIGNAL_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_ROUTER_LMR)

#include <cstdint>

namespace YaneuraOu::Eval::NNUE {

#if defined(USE_NNUE_ROUTER_LMR)
// Production experiment payload: only the Router margin needed by LMR.
// Keeping this separate from NnueSignalSnapshot avoids carrying the diagnostic
// phase/deep/bypass fields, counters, and report machinery in match binaries.
struct NnueRouterLmrSignal {
    std::int32_t router_margin = 0;
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
