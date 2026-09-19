// Experiment 56: diagnostic-only counters for the restricted decision-risk LMR rule.
#ifndef YANEURAOU_NNUE_DECISION_RISK_LMR_COUNTERS_H_INCLUDED
#define YANEURAOU_NNUE_DECISION_RISK_LMR_COUNTERS_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_DECISION_RISK_LMR_COUNTERS)

#if !defined(USE_NNUE_DECISION_RISK_LMR)
#error "ENABLE_NNUE_DECISION_RISK_LMR_COUNTERS requires USE_NNUE_DECISION_RISK_LMR"
#endif

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <ostream>

namespace YaneuraOu::Search::NnueDecisionRiskLmrCounters {

struct Snapshot {
    std::uint64_t total_lmr_events = 0;
    std::uint64_t positive_reduction_events = 0;
    std::uint64_t q8_ge_56_events = 0;
    std::uint64_t cohort_events = 0;
    std::uint64_t overlap_skip_events = 0;
    std::uint64_t router_overlap_skips = 0;
    std::uint64_t lca_overlap_skips = 0;
    std::uint64_t cross_overlap_skips = 0;
    std::uint64_t applied_events = 0;
};

inline std::atomic<std::uint64_t> TotalLmrEvents{0};
inline std::atomic<std::uint64_t> PositiveReductionEvents{0};
inline std::atomic<std::uint64_t> Q8Ge56Events{0};
inline std::atomic<std::uint64_t> CohortEvents{0};
inline std::atomic<std::uint64_t> OverlapSkipEvents{0};
inline std::atomic<std::uint64_t> RouterOverlapSkips{0};
inline std::atomic<std::uint64_t> LcaOverlapSkips{0};
inline std::atomic<std::uint64_t> CrossOverlapSkips{0};
inline std::atomic<std::uint64_t> AppliedEvents{0};

inline void Reset() {
    TotalLmrEvents = 0;
    PositiveReductionEvents = 0;
    Q8Ge56Events = 0;
    CohortEvents = 0;
    OverlapSkipEvents = 0;
    RouterOverlapSkips = 0;
    LcaOverlapSkips = 0;
    CrossOverlapSkips = 0;
    AppliedEvents = 0;
}

inline Snapshot GetSnapshot() {
    return {
      TotalLmrEvents.load(), PositiveReductionEvents.load(), Q8Ge56Events.load(),
      CohortEvents.load(), OverlapSkipEvents.load(), RouterOverlapSkips.load(),
      LcaOverlapSkips.load(), CrossOverlapSkips.load(), AppliedEvents.load()
    };
}

inline double Percent(const std::uint64_t numerator, const std::uint64_t denominator) {
    return denominator ? 100.0 * static_cast<double>(numerator)
                              / static_cast<double>(denominator)
                       : 0.0;
}

inline void Report(std::ostream& output) {
    const auto s = GetSnapshot();
    output << "NNUE decision-risk restricted LMR counters\n"
           << "total_lmr_events=" << s.total_lmr_events << '\n'
           << "positive_reduction_events=" << s.positive_reduction_events << '\n'
           << "q8_ge_56_events=" << s.q8_ge_56_events << '\n'
           << "cohort_events=" << s.cohort_events << '\n'
           << "overlap_skip_events=" << s.overlap_skip_events << '\n'
           << "router_overlap_skips=" << s.router_overlap_skips << '\n'
           << "lca_overlap_skips=" << s.lca_overlap_skips << '\n'
           << "cross_overlap_skips=" << s.cross_overlap_skips << '\n'
           << "applied_events=" << s.applied_events << '\n'
           << std::fixed << std::setprecision(6)
           << "cohort_over_q8_ge_56_pct="
           << Percent(s.cohort_events, s.q8_ge_56_events) << '\n'
           << "applied_over_q8_ge_56_pct="
           << Percent(s.applied_events, s.q8_ge_56_events) << '\n'
           << "applied_over_total_lmr_pct="
           << Percent(s.applied_events, s.total_lmr_events) << '\n';
}

}  // namespace YaneuraOu::Search::NnueDecisionRiskLmrCounters

#endif
#endif
