// Search-only diagnostics for NNUE internal signals.
// Everything in this file disappears unless ENABLE_NNUE_SIGNAL_LOG is set.

#ifndef YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_SIGNAL_LOG)

#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
#ifndef NNUE_ROUTER_LMR_VARIANT
#define NNUE_ROUTER_LMR_VARIANT 2
#endif
#ifndef NNUE_ROUTER_LMR_MARGIN_THRESHOLD
#define NNUE_ROUTER_LMR_MARGIN_THRESHOLD 1024
#endif
#ifndef NNUE_ROUTER_LMR_REDUCTION_DELTA
#define NNUE_ROUTER_LMR_REDUCTION_DELTA 256
#endif
#if NNUE_ROUTER_LMR_VARIANT < 0 || NNUE_ROUTER_LMR_VARIANT > 3
#error "NNUE_ROUTER_LMR_VARIANT must be 0 (current), 1/2 (fixed-point delta), or 3 (explicit +1 ply)"
#endif
#endif

#include "../../eval/nnue/nnue_signal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace YaneuraOu::Search::NnueSignalLog {

constexpr std::size_t kSourceCount =
  static_cast<std::size_t>(Eval::NNUE::NnueSignalEvalSource::Count);
constexpr std::size_t kDepthBins = 20;  // 1..16, 17-24, 25-32, 33+, qsearch
constexpr std::size_t kSignalKinds = 5;
constexpr std::size_t kSignalBins = 24;
constexpr std::size_t kErrorBins = 73;  // exact 0..63, followed by powers of two
constexpr std::size_t kConditionBins = 5;
constexpr std::size_t kBucketCount = 12;

struct CoverageCell {
    std::uint64_t nodes = 0;
    std::uint64_t static_eval_needed = 0;
    std::uint64_t signal_valid = 0;
    std::uint64_t static_eval_signal_valid = 0;
};

struct PredictionBucket {
    std::uint64_t count = 0;
    std::uint64_t error_sum = 0;
    std::array<std::uint64_t, kErrorBins> error_hist{};
    std::uint64_t fail_high = 0;
    std::uint64_t fail_low = 0;
    std::uint64_t futility_pruned = 0;
    std::uint64_t lmr = 0;
    std::uint64_t lmr_research = 0;
    std::uint64_t depth_sum = 0;
    std::uint64_t move_count_sum = 0;
};

struct BasicSummary {
    std::uint64_t count = 0;
    double sum = 0;
    // Finite sentinels avoid -ffast-math's disabled-infinity warning.
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
};

struct CorrelationSums {
    std::uint64_t count = 0;
    double sx = 0;
    double sy = 0;
    double sxx = 0;
    double syy = 0;
    double sxy = 0;
};

struct SignedDirectionStats {
    std::uint64_t count = 0;
    std::uint64_t search_error_positive = 0;
    std::uint64_t search_error_negative = 0;
    std::uint64_t search_error_zero = 0;
    double signed_error_sum = 0;
    std::uint64_t abs_error_sum = 0;
    std::uint64_t fail_high = 0;
    std::uint64_t fail_low = 0;
};

struct FutilityConditionalBucket {
    std::uint64_t count = 0;
    double distance_sum = 0;
    std::uint64_t condition_true = 0;
    std::uint64_t returned_above_static = 0;
    std::uint64_t returned_below_static = 0;
    std::uint64_t returned_equal_static = 0;
    std::uint64_t fail_high = 0;
    std::uint64_t fail_low = 0;
};

struct LmrConditionalBucket {
    std::uint64_t events = 0;
    std::uint64_t researches = 0;
};

#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
struct RouterLmrCohortStats {
    std::uint64_t moves = 0;
    std::uint64_t reduced_fail_highs = 0;
    std::uint64_t researches = 0;
    std::uint64_t research_score_samples = 0;
    std::int64_t research_score_delta_sum = 0;
    std::uint64_t research_score_abs_delta_sum = 0;
    std::uint64_t research_score_abs_delta_max = 0;
    std::uint64_t beta_exceeded = 0;
    std::uint64_t final_cutoffs = 0;
    std::int64_t depth_sum = 0;
    std::int64_t move_count_sum = 0;
    std::uint64_t router_margin_sum = 0;
    std::array<std::uint64_t, kBucketCount> selected_bucket{};
    // Conservative Router-LMR predicate limits used by variant 2.
    std::array<std::uint64_t, 6> depth_3_to_8{};
    std::array<std::uint64_t, 7> move_count_2_to_8{};
};

struct RouterLmrExperimentStats {
    std::uint64_t lmr_moves = 0;
    std::uint64_t signal_valid_moves = 0;
    std::uint64_t high_risk_moves = 0;
    std::uint64_t adjusted_moves = 0;
    std::uint64_t adjusted_nodes = 0;
    std::uint64_t all_reduced_fail_highs = 0;
    std::uint64_t all_researches = 0;
    std::uint64_t reduced_fail_highs = 0;
    std::uint64_t researches = 0;
    std::uint64_t fixed_reduction_removed = 0;
    std::uint64_t search_depth_increase = 0;
    std::uint64_t search_depth_increase_moves = 0;
    std::array<std::uint64_t, kBucketCount> adjusted_by_bucket{};
    std::array<std::uint64_t, kSignalBins> adjusted_by_margin{};
    // Index 0: applying delta would leave integer d unchanged.
    // Index 1: applying delta would increase integer d by one ply.
    std::array<RouterLmrCohortStats, 2> eligible_depth_cohort{};
    // Floor-normalized original r modulo 1024, in eight 128-unit bins.
    std::array<RouterLmrCohortStats, 8> original_remainder{};
    // Counterfactual +1-ply cohorts for delta={256,512,768,1024}.
    std::array<RouterLmrCohortStats, 4> counterfactual_delta{};
    std::array<std::array<std::uint64_t, 8>, 4> counterfactual_delta_by_remainder{};
    // Nested margin cohorts for threshold={128,256,512,1024}.  These are
    // observational only: variant 0 never changes r or the searched depth.
    std::array<RouterLmrCohortStats, 4> margin_threshold_cohort{};
};
#endif

// Used to measure a phase/error relationship after jointly stratifying on
// depth, |staticEval|, and |materialValue|. Sums are centered per stratum at
// report time, avoiding a large per-sample log.
using PhaseStratum = CorrelationSums;

struct ThreadStats {
    CoverageCell total{};
    std::array<CoverageCell, 2> pv{};
    std::array<CoverageCell, 2> in_check{};
    std::array<CoverageCell, 2> qsearch{};
    std::array<CoverageCell, kDepthBins> depth{};
    std::array<std::uint64_t, kSourceCount> source{};
    std::array<std::array<PredictionBucket, kSignalBins>, kSignalKinds> prediction{};
    std::array<CorrelationSums, kSignalKinds> correlation{};
    std::array<BasicSummary, 17> raw_signal{};
    std::array<std::uint64_t, kBucketCount> selected_bucket{};
    std::array<SignedDirectionStats, 3> signed_direction{};  // negative, zero, positive
    CorrelationSums signed_deep_vs_signed_error{};
    std::array<std::array<PredictionBucket, kSignalBins>, kConditionBins>
      disagreement_by_depth{};
    std::array<std::array<PredictionBucket, kSignalBins>, kConditionBins>
      disagreement_by_static_eval{};
    std::array<std::array<PredictionBucket, kSignalBins>, kBucketCount>
      disagreement_by_bucket{};
    std::array<FutilityConditionalBucket, kSignalBins> futility_eligible{};
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kConditionBins>
      router_lmr_by_depth{};
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kConditionBins>
      router_lmr_by_move_count{};
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kBucketCount>
      router_lmr_by_bucket{};
    std::array<CorrelationSums, 4> control_correlation{};  // depth, |eval|, |material|, ply
    std::array<std::array<std::array<std::array<PhaseStratum, kConditionBins>,
                                     kConditionBins>,
                          kConditionBins>,
               3>
      phase_conditioned{};
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
    RouterLmrExperimentStats router_lmr_experiment{};
#endif
    std::uint64_t fresh_with_result = 0;
    std::uint64_t fresh_without_result = 0;
    std::uint64_t fresh_decisive_excluded = 0;
};

inline std::mutex g_registry_mutex;
inline std::vector<ThreadStats*> g_thread_stats;

inline ThreadStats& LocalStats() {
    thread_local ThreadStats* stats = [] {
        auto* value = new ThreadStats();
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        g_thread_stats.push_back(value);
        return value;
    }();
    return *stats;
}

inline std::size_t DepthBin(const int depth) {
    if (depth <= 16)
        return static_cast<std::size_t>(std::max(depth, 1) - 1);
    if (depth <= 24)
        return 16;
    if (depth <= 32)
        return 17;
    return 18;
}

inline std::size_t ConditionalDepthBin(const int depth) {
    if (depth <= 2) return 0;
    if (depth <= 4) return 1;
    if (depth <= 6) return 2;
    if (depth <= 8) return 3;
    return 4;
}

inline std::size_t MagnitudeBin(const std::int64_t value) {
    const auto magnitude = static_cast<std::uint64_t>(std::llabs(value));
    if (magnitude < 100) return 0;
    if (magnitude < 300) return 1;
    if (magnitude < 600) return 2;
    if (magnitude < 1200) return 3;
    return 4;
}

inline std::size_t MoveCountBin(const int move_count) {
    if (move_count <= 2) return 0;
    if (move_count <= 4) return 1;
    if (move_count <= 8) return 2;
    if (move_count <= 16) return 3;
    return 4;
}

inline std::size_t ErrorBin(const std::uint32_t error) {
    if (error < 64)
        return error;
    std::size_t bin = 64;
    std::uint32_t upper = 127;
    while (bin + 1 < kErrorBins && error > upper) {
        upper = upper * 2 + 1;
        ++bin;
    }
    return bin;
}

inline std::uint32_t ErrorBinUpper(const std::size_t bin) {
    if (bin < 64)
        return static_cast<std::uint32_t>(bin);
    return (std::uint32_t(1) << std::min<std::size_t>(bin - 57, 31)) - 1;
}

inline std::size_t LogSignalBin(const std::int64_t value) {
    const std::uint64_t magnitude = static_cast<std::uint64_t>(std::max<std::int64_t>(value, 0));
    if (!magnitude)
        return 0;
    std::size_t bin = 1;
    std::uint64_t upper = 1;
    while (bin + 1 < kSignalBins && magnitude > upper) {
        upper <<= 1;
        ++bin;
    }
    return bin;
}

inline std::size_t PhaseSignalBin(const float value) {
    // 1/8-wide bins cover [0, 3); all larger values share the last bin.
    return std::min<std::size_t>(static_cast<std::size_t>(std::max(value, 0.0f) * 8.0f),
                                 kSignalBins - 1);
}

inline double SignalNumericValue(const Eval::NNUE::NnueSignalSnapshot& signal,
                                 const std::size_t kind) {
    switch (kind) {
    case 0: return signal.deep_bypass_disagreement;
    case 1: return signal.router_margin;
    case 2: return signal.main_reliance;
    case 3: return signal.fm_reliance;
    default: return signal.cross_reliance;
    }
}

inline std::size_t SignalBin(const Eval::NNUE::NnueSignalSnapshot& signal,
                             const std::size_t kind) {
    if (kind == 0)
        return LogSignalBin(signal.deep_bypass_disagreement);
    if (kind == 1)
        return LogSignalBin(signal.router_margin);
    return PhaseSignalBin(static_cast<float>(SignalNumericValue(signal, kind)));
}

inline void AddCorrelationSample(CorrelationSums& sums, const double x, const double y) {
    ++sums.count;
    sums.sx += x;
    sums.sy += y;
    sums.sxx += x * x;
    sums.syy += y * y;
    sums.sxy += x * y;
}

inline void AddPredictionSample(PredictionBucket& bucket, const std::uint32_t error,
                                const int result, const int alpha, const int beta,
                                const bool futility_pruned, const bool lmr,
                                const bool lmr_research, const int depth,
                                const int move_count) {
    ++bucket.count;
    bucket.error_sum += error;
    ++bucket.error_hist[ErrorBin(error)];
    bucket.fail_high += result >= beta;
    bucket.fail_low += result <= alpha;
    bucket.futility_pruned += futility_pruned;
    bucket.lmr += lmr;
    bucket.lmr_research += lmr_research;
    bucket.depth_sum += static_cast<std::uint64_t>(std::max(depth, 0));
    bucket.move_count_sum += static_cast<std::uint64_t>(std::max(move_count, 0));
}

class NodeObservation {
   public:
    NodeObservation(const bool pv, const bool in_check, const int depth,
                    const int alpha, const int beta, const int material,
                    const int ply, const bool qsearch = false)
        : pv_(pv), in_check_(in_check), qsearch_(qsearch), depth_(depth), alpha_(alpha), beta_(beta),
          material_(material), ply_(ply) {}

    ~NodeObservation() {
        auto& stats = LocalStats();
        const bool valid = access_.signal.valid;
        auto add_coverage = [&](CoverageCell& cell) {
            ++cell.nodes;
            cell.static_eval_needed += static_eval_needed_;
            cell.signal_valid += valid;
            cell.static_eval_signal_valid += static_eval_needed_ && valid;
        };
        add_coverage(stats.total);
        add_coverage(stats.pv[pv_]);
        add_coverage(stats.in_check[in_check_]);
        add_coverage(stats.qsearch[qsearch_]);
        add_coverage(stats.depth[qsearch_ ? 19 : DepthBin(depth_)]);
        ++stats.source[static_cast<std::size_t>(access_.source)];
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
        stats.router_lmr_experiment.adjusted_nodes += router_lmr_node_adjusted_;
#endif

        if (access_.source == Eval::NNUE::NnueSignalEvalSource::FreshNetwork && valid) {
            const double values[17] = {
              double(access_.signal.deep_output), double(access_.signal.bypass_output),
              double(access_.signal.signed_deep_bypass),
              double(access_.signal.deep_bypass_disagreement),
              double(access_.signal.selected_bucket), double(access_.signal.router_top1_logit),
              double(access_.signal.router_top2_logit), double(access_.signal.router_margin),
              access_.signal.phase_scale[0], access_.signal.phase_scale[1],
              access_.signal.phase_scale[2], access_.signal.phase_scale[3],
              access_.signal.phase_scale[4], access_.signal.phase_scale[5],
              access_.signal.main_reliance, access_.signal.fm_reliance,
              access_.signal.cross_reliance};
            for (std::size_t i = 0; i < 17; ++i) {
                auto& summary = stats.raw_signal[i];
                ++summary.count;
                summary.sum += values[i];
                summary.minimum = std::min(summary.minimum, values[i]);
                summary.maximum = std::max(summary.maximum, values[i]);
            }
            if (access_.signal.selected_bucket >= 0 && access_.signal.selected_bucket < 12)
                ++stats.selected_bucket[static_cast<std::size_t>(access_.signal.selected_bucket)];
        }

        if (access_.source != Eval::NNUE::NnueSignalEvalSource::FreshNetwork || !valid || qsearch_)
            return;
        if (!has_static_eval_ || !has_result_) {
            ++stats.fresh_without_result;
            return;
        }

        ++stats.fresh_with_result;
        // Mate/TB-like returns measure proof distance rather than static-eval
        // error and would dominate the mean/correlation. Keep their coverage
        // count but exclude them from the predictiveness tables.
        if (std::abs(result_) >= 30000) {
            ++stats.fresh_decisive_excluded;
            return;
        }
        const std::uint32_t error = static_cast<std::uint32_t>(
          std::min<std::int64_t>(std::llabs(std::int64_t(result_) - static_eval_),
                                 std::numeric_limits<std::uint32_t>::max()));
        const std::int64_t signed_error = std::int64_t(result_) - static_eval_;
        for (std::size_t kind = 0; kind < kSignalKinds; ++kind) {
            const auto bin = SignalBin(access_.signal, kind);
            AddPredictionSample(stats.prediction[kind][bin], error, result_, alpha_, beta_,
                                futility_pruned_, lmr_, lmr_research_, depth_, move_count_);

            const double x = SignalNumericValue(access_.signal, kind);
            const double y = error;
            AddCorrelationSample(stats.correlation[kind], x, y);
        }

        const int direction = access_.signal.signed_deep_bypass < 0 ? 0
                            : access_.signal.signed_deep_bypass > 0 ? 2 : 1;
        auto& signed_stats = stats.signed_direction[static_cast<std::size_t>(direction)];
        ++signed_stats.count;
        signed_stats.search_error_positive += signed_error > 0;
        signed_stats.search_error_negative += signed_error < 0;
        signed_stats.search_error_zero += signed_error == 0;
        signed_stats.signed_error_sum += static_cast<double>(signed_error);
        signed_stats.abs_error_sum += error;
        signed_stats.fail_high += result_ >= beta_;
        signed_stats.fail_low += result_ <= alpha_;
        AddCorrelationSample(stats.signed_deep_vs_signed_error,
                             access_.signal.signed_deep_bypass,
                             static_cast<double>(signed_error));

        const auto disagreement_bin = SignalBin(access_.signal, 0);
        const auto depth_bin = ConditionalDepthBin(depth_);
        const auto static_bin = MagnitudeBin(static_eval_);
        const auto bucket_index = static_cast<std::size_t>(std::clamp(
          access_.signal.selected_bucket, 0, static_cast<int>(kBucketCount - 1)));
        AddPredictionSample(stats.disagreement_by_depth[depth_bin][disagreement_bin],
                            error, result_, alpha_, beta_, futility_pruned_, lmr_,
                            lmr_research_, depth_, move_count_);
        AddPredictionSample(stats.disagreement_by_static_eval[static_bin][disagreement_bin],
                            error, result_, alpha_, beta_, futility_pruned_, lmr_,
                            lmr_research_, depth_, move_count_);
        AddPredictionSample(stats.disagreement_by_bucket[bucket_index][disagreement_bin],
                            error, result_, alpha_, beta_, futility_pruned_, lmr_,
                            lmr_research_, depth_, move_count_);

        if (reverse_futility_eligible_) {
            auto& bucket = stats.futility_eligible[disagreement_bin];
            ++bucket.count;
            bucket.distance_sum += reverse_futility_distance_;
            bucket.condition_true += reverse_futility_taken_;
            bucket.returned_above_static += signed_error > 0;
            bucket.returned_below_static += signed_error < 0;
            bucket.returned_equal_static += signed_error == 0;
            bucket.fail_high += result_ >= beta_;
            bucket.fail_low += result_ <= alpha_;
        }

        const auto margin_bin = SignalBin(access_.signal, 1);
        for (std::size_t move_bin = 0; move_bin < kConditionBins; ++move_bin) {
            const auto events = lmr_events_[move_bin];
            const auto researches = lmr_researches_[move_bin];
            stats.router_lmr_by_move_count[move_bin][margin_bin].events += events;
            stats.router_lmr_by_move_count[move_bin][margin_bin].researches += researches;
            stats.router_lmr_by_depth[depth_bin][margin_bin].events += events;
            stats.router_lmr_by_depth[depth_bin][margin_bin].researches += researches;
            stats.router_lmr_by_bucket[bucket_index][margin_bin].events += events;
            stats.router_lmr_by_bucket[bucket_index][margin_bin].researches += researches;
        }

        const double controls[4] = {double(depth_), double(std::llabs(std::int64_t(static_eval_))),
                                    double(std::llabs(std::int64_t(material_))), double(ply_)};
        for (std::size_t i = 0; i < 4; ++i)
            AddCorrelationSample(stats.control_correlation[i], controls[i], error);

        const auto material_bin = MagnitudeBin(material_);
        for (std::size_t phase = 0; phase < 3; ++phase)
            AddCorrelationSample(
              stats.phase_conditioned[phase][depth_bin][static_bin][material_bin],
              SignalNumericValue(access_.signal, phase + 2), error);
    }

    void StaticEvalNeeded() { static_eval_needed_ = true; }

    void SetSource(const Eval::NNUE::NnueSignalEvalSource source) {
        access_ = Eval::NNUE::NnueSignalEvalAccess{};
        access_.source = source;
    }

    void SetEvalAccess(const Eval::NNUE::NnueSignalEvalAccess& access) { access_ = access; }

    const Eval::NNUE::NnueSignalSnapshot* Signal() const {
        return access_.signal.valid ? &access_.signal : nullptr;
    }

    void SetStaticEval(const int value) {
        static_eval_ = value;
        has_static_eval_ = true;
    }

    void MarkFutilityPruned() { futility_pruned_ = true; }
    void MarkLmr() { lmr_ = true; }
    void MarkLmrResearch() { lmr_research_ = true; }
    void MarkReverseFutilityEligible(const int distance, const bool taken) {
        reverse_futility_eligible_ = true;
        reverse_futility_distance_ = distance;
        reverse_futility_taken_ = taken;
    }
    void RecordLmrEvent(const int move_count, const bool researched) {
        const auto bin = MoveCountBin(move_count);
        ++lmr_events_[bin];
        lmr_researches_[bin] += researched;
    }
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
    void RecordRouterLmrExperiment(const bool high_risk, const bool adjusted,
                                   const int fixed_reduction_removed,
                                   const int search_depth_increase,
                                   const bool reduced_fail_high, const bool researched) {
        auto& stats = LocalStats().router_lmr_experiment;
        ++stats.lmr_moves;
        stats.signal_valid_moves += access_.signal.valid;
        stats.high_risk_moves += high_risk;
        stats.all_reduced_fail_highs += reduced_fail_high;
        stats.all_researches += researched;
        if (!adjusted)
            return;

        ++stats.adjusted_moves;
        router_lmr_node_adjusted_ = true;
        stats.reduced_fail_highs += reduced_fail_high;
        stats.researches += researched;
        stats.fixed_reduction_removed += static_cast<std::uint64_t>(fixed_reduction_removed);
        if (search_depth_increase > 0) {
            ++stats.search_depth_increase_moves;
            stats.search_depth_increase += static_cast<std::uint64_t>(search_depth_increase);
        }
        const auto bucket = static_cast<std::size_t>(std::clamp(
          access_.signal.selected_bucket, 0, static_cast<int>(kBucketCount - 1)));
        ++stats.adjusted_by_bucket[bucket];
        ++stats.adjusted_by_margin[SignalBin(access_.signal, 1)];
    }

    void RecordRouterLmrCohortOutcome(const int candidate_depth_increase,
                                      const int original_remainder,
                                      const unsigned counterfactual_delta_mask,
                                      const int depth, const int move_count,
                                      const bool reduced_fail_high, const bool researched,
                                      const int reduced_value, const int final_value,
                                      const bool beta_exceeded, const bool final_cutoff) {
        auto add = [&](RouterLmrCohortStats& cohort) {
            ++cohort.moves;
            cohort.reduced_fail_highs += reduced_fail_high;
            cohort.researches += researched;
            cohort.beta_exceeded += beta_exceeded;
            cohort.final_cutoffs += final_cutoff;
            cohort.depth_sum += depth;
            cohort.move_count_sum += move_count;
            cohort.router_margin_sum +=
              static_cast<std::uint64_t>(std::max(access_.signal.router_margin, 0));
            const auto bucket = static_cast<std::size_t>(std::clamp(
              access_.signal.selected_bucket, 0, static_cast<int>(kBucketCount - 1)));
            ++cohort.selected_bucket[bucket];
            if (depth >= 3 && depth <= 8)
                ++cohort.depth_3_to_8[static_cast<std::size_t>(depth - 3)];
            if (move_count >= 2 && move_count <= 8)
                ++cohort.move_count_2_to_8[static_cast<std::size_t>(move_count - 2)];
            if (!researched)
                return;
            const auto delta = static_cast<std::int64_t>(final_value)
                             - static_cast<std::int64_t>(reduced_value);
            const auto abs_delta = static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
            ++cohort.research_score_samples;
            cohort.research_score_delta_sum += delta;
            cohort.research_score_abs_delta_sum += abs_delta;
            cohort.research_score_abs_delta_max =
              std::max(cohort.research_score_abs_delta_max, abs_delta);
        };

        const auto cohort_index = candidate_depth_increase > 0 ? 1U : 0U;
        auto& experiment = LocalStats().router_lmr_experiment;
        add(experiment.eligible_depth_cohort[cohort_index]);
        add(experiment.original_remainder[static_cast<std::size_t>(original_remainder / 128)]);
        for (std::size_t delta_index = 0; delta_index < 4; ++delta_index)
            if (counterfactual_delta_mask & (1U << delta_index)) {
                add(experiment.counterfactual_delta[delta_index]);
                ++experiment.counterfactual_delta_by_remainder[delta_index]
                                                           [original_remainder / 128];
            }
        const int margin_thresholds[4] = {128, 256, 512, 1024};
        for (std::size_t threshold_index = 0; threshold_index < 4; ++threshold_index)
            if (access_.signal.router_margin <= margin_thresholds[threshold_index])
                add(experiment.margin_threshold_cohort[threshold_index]);
    }
#endif
    void SetMoveCount(const int move_count) { move_count_ = move_count; }

    template<typename ValueType>
    ValueType Return(const ValueType value) {
        result_ = static_cast<int>(value);
        has_result_ = true;
        return value;
    }

   private:
    bool pv_ = false;
    bool in_check_ = false;
    bool qsearch_ = false;
    int depth_ = 0;
    int alpha_ = 0;
    int beta_ = 0;
    bool static_eval_needed_ = false;
    bool has_static_eval_ = false;
    bool has_result_ = false;
    bool futility_pruned_ = false;
    bool lmr_ = false;
    bool lmr_research_ = false;
    int static_eval_ = 0;
    int result_ = 0;
    int move_count_ = 0;
    int material_ = 0;
    int ply_ = 0;
    bool reverse_futility_eligible_ = false;
    bool reverse_futility_taken_ = false;
    int reverse_futility_distance_ = 0;
    std::array<std::uint32_t, kConditionBins> lmr_events_{};
    std::array<std::uint32_t, kConditionBins> lmr_researches_{};
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
    bool router_lmr_node_adjusted_ = false;
#endif
    Eval::NNUE::NnueSignalEvalAccess access_{};
};

inline void Reset() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (auto* stats : g_thread_stats)
        *stats = ThreadStats{};
}

inline void AddCoverage(CoverageCell& dst, const CoverageCell& src) {
    dst.nodes += src.nodes;
    dst.static_eval_needed += src.static_eval_needed;
    dst.signal_valid += src.signal_valid;
    dst.static_eval_signal_valid += src.static_eval_signal_valid;
}

inline void AddPrediction(PredictionBucket& dst, const PredictionBucket& src) {
    dst.count += src.count;
    dst.error_sum += src.error_sum;
    for (std::size_t i = 0; i < kErrorBins; ++i)
        dst.error_hist[i] += src.error_hist[i];
    dst.fail_high += src.fail_high;
    dst.fail_low += src.fail_low;
    dst.futility_pruned += src.futility_pruned;
    dst.lmr += src.lmr;
    dst.lmr_research += src.lmr_research;
    dst.depth_sum += src.depth_sum;
    dst.move_count_sum += src.move_count_sum;
}

inline void AddCorrelation(CorrelationSums& dst, const CorrelationSums& src) {
    dst.count += src.count;
    dst.sx += src.sx;
    dst.sy += src.sy;
    dst.sxx += src.sxx;
    dst.syy += src.syy;
    dst.sxy += src.sxy;
}

inline ThreadStats Snapshot() {
    ThreadStats result{};
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (const auto* stats : g_thread_stats) {
        AddCoverage(result.total, stats->total);
        for (std::size_t i = 0; i < result.pv.size(); ++i)
            AddCoverage(result.pv[i], stats->pv[i]);
        for (std::size_t i = 0; i < result.in_check.size(); ++i)
            AddCoverage(result.in_check[i], stats->in_check[i]);
        for (std::size_t i = 0; i < result.qsearch.size(); ++i)
            AddCoverage(result.qsearch[i], stats->qsearch[i]);
        for (std::size_t i = 0; i < result.depth.size(); ++i)
            AddCoverage(result.depth[i], stats->depth[i]);
        for (std::size_t i = 0; i < kSourceCount; ++i)
            result.source[i] += stats->source[i];
        for (std::size_t kind = 0; kind < kSignalKinds; ++kind) {
            for (std::size_t bin = 0; bin < kSignalBins; ++bin)
                AddPrediction(result.prediction[kind][bin], stats->prediction[kind][bin]);
            AddCorrelation(result.correlation[kind], stats->correlation[kind]);
        }
        result.fresh_with_result += stats->fresh_with_result;
        result.fresh_without_result += stats->fresh_without_result;
        result.fresh_decisive_excluded += stats->fresh_decisive_excluded;
        for (std::size_t i = 0; i < result.raw_signal.size(); ++i) {
            auto& dst = result.raw_signal[i];
            const auto& src = stats->raw_signal[i];
            if (!src.count)
                continue;
            dst.count += src.count;
            dst.sum += src.sum;
            dst.minimum = std::min(dst.minimum, src.minimum);
            dst.maximum = std::max(dst.maximum, src.maximum);
        }
        for (std::size_t i = 0; i < result.selected_bucket.size(); ++i)
            result.selected_bucket[i] += stats->selected_bucket[i];
        for (std::size_t direction = 0; direction < 3; ++direction) {
            auto& dst = result.signed_direction[direction];
            const auto& src = stats->signed_direction[direction];
            dst.count += src.count;
            dst.search_error_positive += src.search_error_positive;
            dst.search_error_negative += src.search_error_negative;
            dst.search_error_zero += src.search_error_zero;
            dst.signed_error_sum += src.signed_error_sum;
            dst.abs_error_sum += src.abs_error_sum;
            dst.fail_high += src.fail_high;
            dst.fail_low += src.fail_low;
        }
        AddCorrelation(result.signed_deep_vs_signed_error, stats->signed_deep_vs_signed_error);
        for (std::size_t group = 0; group < kConditionBins; ++group)
            for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
                AddPrediction(result.disagreement_by_depth[group][bin],
                              stats->disagreement_by_depth[group][bin]);
                AddPrediction(result.disagreement_by_static_eval[group][bin],
                              stats->disagreement_by_static_eval[group][bin]);
                result.router_lmr_by_depth[group][bin].events +=
                  stats->router_lmr_by_depth[group][bin].events;
                result.router_lmr_by_depth[group][bin].researches +=
                  stats->router_lmr_by_depth[group][bin].researches;
                result.router_lmr_by_move_count[group][bin].events +=
                  stats->router_lmr_by_move_count[group][bin].events;
                result.router_lmr_by_move_count[group][bin].researches +=
                  stats->router_lmr_by_move_count[group][bin].researches;
            }
        for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
            for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
                AddPrediction(result.disagreement_by_bucket[bucket][bin],
                              stats->disagreement_by_bucket[bucket][bin]);
                result.router_lmr_by_bucket[bucket][bin].events +=
                  stats->router_lmr_by_bucket[bucket][bin].events;
                result.router_lmr_by_bucket[bucket][bin].researches +=
                  stats->router_lmr_by_bucket[bucket][bin].researches;
            }
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            auto& dst = result.futility_eligible[bin];
            const auto& src = stats->futility_eligible[bin];
            dst.count += src.count;
            dst.distance_sum += src.distance_sum;
            dst.condition_true += src.condition_true;
            dst.returned_above_static += src.returned_above_static;
            dst.returned_below_static += src.returned_below_static;
            dst.returned_equal_static += src.returned_equal_static;
            dst.fail_high += src.fail_high;
            dst.fail_low += src.fail_low;
        }
        for (std::size_t i = 0; i < 4; ++i)
            AddCorrelation(result.control_correlation[i], stats->control_correlation[i]);
        for (std::size_t phase = 0; phase < 3; ++phase)
            for (std::size_t depth = 0; depth < kConditionBins; ++depth)
                for (std::size_t eval = 0; eval < kConditionBins; ++eval)
                    for (std::size_t material = 0; material < kConditionBins; ++material)
                        AddCorrelation(result.phase_conditioned[phase][depth][eval][material],
                          stats->phase_conditioned[phase][depth][eval][material]);
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
        auto& experiment = result.router_lmr_experiment;
        const auto& source_experiment = stats->router_lmr_experiment;
        experiment.lmr_moves += source_experiment.lmr_moves;
        experiment.signal_valid_moves += source_experiment.signal_valid_moves;
        experiment.high_risk_moves += source_experiment.high_risk_moves;
        experiment.adjusted_moves += source_experiment.adjusted_moves;
        experiment.adjusted_nodes += source_experiment.adjusted_nodes;
        experiment.all_reduced_fail_highs += source_experiment.all_reduced_fail_highs;
        experiment.all_researches += source_experiment.all_researches;
        experiment.reduced_fail_highs += source_experiment.reduced_fail_highs;
        experiment.researches += source_experiment.researches;
        experiment.fixed_reduction_removed += source_experiment.fixed_reduction_removed;
        experiment.search_depth_increase += source_experiment.search_depth_increase;
        experiment.search_depth_increase_moves += source_experiment.search_depth_increase_moves;
        for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
            experiment.adjusted_by_bucket[bucket] += source_experiment.adjusted_by_bucket[bucket];
        for (std::size_t bin = 0; bin < kSignalBins; ++bin)
            experiment.adjusted_by_margin[bin] += source_experiment.adjusted_by_margin[bin];
        const auto add_cohort = [](RouterLmrCohortStats& cohort,
                                   const RouterLmrCohortStats& source) {
            cohort.moves += source.moves;
            cohort.reduced_fail_highs += source.reduced_fail_highs;
            cohort.researches += source.researches;
            cohort.research_score_samples += source.research_score_samples;
            cohort.research_score_delta_sum += source.research_score_delta_sum;
            cohort.research_score_abs_delta_sum += source.research_score_abs_delta_sum;
            cohort.research_score_abs_delta_max =
              std::max(cohort.research_score_abs_delta_max, source.research_score_abs_delta_max);
            cohort.beta_exceeded += source.beta_exceeded;
            cohort.final_cutoffs += source.final_cutoffs;
            cohort.depth_sum += source.depth_sum;
            cohort.move_count_sum += source.move_count_sum;
            cohort.router_margin_sum += source.router_margin_sum;
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                cohort.selected_bucket[bucket] += source.selected_bucket[bucket];
            for (std::size_t depth = 0; depth < cohort.depth_3_to_8.size(); ++depth)
                cohort.depth_3_to_8[depth] += source.depth_3_to_8[depth];
            for (std::size_t move = 0; move < cohort.move_count_2_to_8.size(); ++move)
                cohort.move_count_2_to_8[move] += source.move_count_2_to_8[move];
        };
        for (std::size_t cohort_index = 0; cohort_index < 2; ++cohort_index) {
            add_cohort(experiment.eligible_depth_cohort[cohort_index],
                       source_experiment.eligible_depth_cohort[cohort_index]);
        }
        for (std::size_t bin = 0; bin < 8; ++bin)
            add_cohort(experiment.original_remainder[bin],
                       source_experiment.original_remainder[bin]);
        for (std::size_t delta_index = 0; delta_index < 4; ++delta_index)
            add_cohort(experiment.counterfactual_delta[delta_index],
                       source_experiment.counterfactual_delta[delta_index]);
        for (std::size_t threshold_index = 0; threshold_index < 4; ++threshold_index)
            add_cohort(experiment.margin_threshold_cohort[threshold_index],
                       source_experiment.margin_threshold_cohort[threshold_index]);
        for (std::size_t delta_index = 0; delta_index < 4; ++delta_index)
            for (std::size_t bin = 0; bin < 8; ++bin)
                experiment.counterfactual_delta_by_remainder[delta_index][bin] +=
                  source_experiment.counterfactual_delta_by_remainder[delta_index][bin];
#endif
    }
    return result;
}

inline double Percent(const std::uint64_t numerator, const std::uint64_t denominator) {
    return denominator ? 100.0 * double(numerator) / double(denominator) : 0.0;
}

inline std::uint32_t Quantile(const PredictionBucket& bucket, const double q) {
    if (!bucket.count)
        return 0;
    const auto target = static_cast<std::uint64_t>(std::ceil(q * bucket.count));
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < kErrorBins; ++i) {
        cumulative += bucket.error_hist[i];
        if (cumulative >= target)
            return ErrorBinUpper(i);
    }
    return ErrorBinUpper(kErrorBins - 1);
}

inline double Pearson(const CorrelationSums& c) {
    if (c.count < 2)
        return 0.0L;
    const double n = static_cast<double>(c.count);
    const double numerator = n * c.sxy - c.sx * c.sy;
    const double dx = n * c.sxx - c.sx * c.sx;
    const double dy = n * c.syy - c.sy * c.sy;
    return dx > 0 && dy > 0 ? numerator / std::sqrt(dx * dy) : 0.0;
}

inline double PhaseWithinStrataPearson(const ThreadStats& stats, const std::size_t phase,
                                       std::uint64_t& sample_count) {
    double covariance = 0;
    double variance_x = 0;
    double variance_y = 0;
    sample_count = 0;
    for (std::size_t depth = 0; depth < kConditionBins; ++depth)
        for (std::size_t eval = 0; eval < kConditionBins; ++eval)
            for (std::size_t material = 0; material < kConditionBins; ++material) {
                const auto& cell = stats.phase_conditioned[phase][depth][eval][material];
                if (cell.count < 2)
                    continue;
                const double n = static_cast<double>(cell.count);
                covariance += cell.sxy - cell.sx * cell.sy / n;
                variance_x += cell.sxx - cell.sx * cell.sx / n;
                variance_y += cell.syy - cell.sy * cell.sy / n;
                sample_count += cell.count;
            }
    return variance_x > 0 && variance_y > 0
         ? covariance / std::sqrt(variance_x * variance_y) : 0.0;
}

inline const char* SourceName(const std::size_t source) {
    static constexpr const char* names[] = {
      "fresh Network evaluation", "accumulator cached score", "eval hash hit",
      "TT eval reuse", "in-check evaluation skipped", "signal unavailable"};
    return names[source];
}

inline const char* SignalName(const std::size_t kind) {
    static constexpr const char* names[] = {
      "deep_bypass_disagreement", "router_margin", "phase.main_reliance",
      "phase.fm_reliance", "phase.cross_reliance"};
    return names[kind];
}

inline void PrintCoverage(std::ostream& out, const char* label, const CoverageCell& cell) {
    out << "  " << std::left << std::setw(18) << label << std::right
        << " nodes=" << std::setw(12) << cell.nodes
        << " static-needed=" << std::setw(12) << cell.static_eval_needed
        << " valid=" << std::setw(12) << cell.signal_valid
        << " (all " << std::fixed << std::setprecision(3)
        << Percent(cell.signal_valid, cell.nodes) << "%, static-needed "
        << Percent(cell.static_eval_signal_valid, cell.static_eval_needed) << "% = "
        << cell.static_eval_signal_valid << '/' << cell.static_eval_needed << ")\n";
}

inline void PrintSignalRange(std::ostream& out, const std::size_t kind,
                             const std::size_t bin) {
    if (kind < 2) {
        if (bin == 0)
            out << "0";
        else {
            const std::uint64_t low = bin == 1 ? 1 : (std::uint64_t(1) << (bin - 2)) + 1;
            const std::uint64_t high = std::uint64_t(1) << (bin - 1);
            out << low << ".." << high;
        }
    } else {
        out << std::fixed << std::setprecision(3) << double(bin) / 8.0 << ".."
            << double(bin + 1) / 8.0;
    }
}

template<std::size_t Groups>
inline void PrintConditionalPrediction(
  std::ostream& out, const char* title, const std::array<const char*, Groups>& labels,
  const std::array<std::array<PredictionBucket, kSignalBins>, Groups>& table) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.count)
                continue;
            out << "    disagreement=";
            PrintSignalRange(out, 0, bin);
            out << " count=" << value.count
                << " mean=" << std::fixed << std::setprecision(3)
                << double(value.error_sum) / value.count
                << " median<=" << Quantile(value, 0.50)
                << " p90<=" << Quantile(value, 0.90)
                << " p99<=" << Quantile(value, 0.99)
                << " FH=" << Percent(value.fail_high, value.count) << '%'
                << " FL=" << Percent(value.fail_low, value.count) << "%\n";
        }
    }
}

template<std::size_t Groups>
inline void PrintConditionalLmr(
  std::ostream& out, const char* title, const std::array<const char*, Groups>& labels,
  const std::array<std::array<LmrConditionalBucket, kSignalBins>, Groups>& table) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.events)
                continue;
            out << "    router_margin=";
            PrintSignalRange(out, 1, bin);
            out << " events=" << value.events << " researches=" << value.researches
                << " rate=" << std::fixed << std::setprecision(3)
                << Percent(value.researches, value.events) << "%\n";
        }
    }
}

inline void Report(std::ostream& out) {
    const auto stats = Snapshot();
    out << "[NNUE search signal diagnostics]\n"
        << "  coverage scope: main search and qsearch nodes\n"
        << "  prediction population: main-search fresh evaluations with an observed non-decisive return value\n"
        << "  static eval: correction-history-adjusted ss->staticEval\n"
        << "  futility/LMR flags: node-level (one or more such move events)\n";
    PrintCoverage(out, "all", stats.total);
    PrintCoverage(out, "non-PV", stats.pv[0]);
    PrintCoverage(out, "PV", stats.pv[1]);
    PrintCoverage(out, "non-check", stats.in_check[0]);
    PrintCoverage(out, "in-check", stats.in_check[1]);
    PrintCoverage(out, "main search", stats.qsearch[0]);
    PrintCoverage(out, "qsearch", stats.qsearch[1]);

    out << "[evaluation access path]\n";
    for (std::size_t i = 0; i < kSourceCount; ++i)
        out << "  " << std::left << std::setw(29) << SourceName(i) << std::right
            << std::setw(12) << stats.source[i] << "  ("
            << std::fixed << std::setprecision(3) << Percent(stats.source[i], stats.total.nodes)
            << "%)\n";

    out << "[depth coverage]\n";
    for (std::size_t i = 0; i < 20; ++i) {
        const char* label = nullptr;
        std::string generated;
        if (i < 16)
            generated = std::to_string(i + 1);
        else if (i == 16)
            generated = "17-24";
        else if (i == 17)
            generated = "25-32";
        else if (i == 18)
            generated = "33+";
        else
            generated = "qsearch";
        label = generated.c_str();
        PrintCoverage(out, label, stats.depth[i]);
    }

    out << "[prediction coverage]\n"
        << "  fresh with returned value    : " << stats.fresh_with_result << '\n'
        << "  fresh without returned value : " << stats.fresh_without_result << '\n'
        << "  decisive returns excluded    : " << stats.fresh_decisive_excluded << '\n'
        << "  note: error quantiles above 63 cp use power-of-two histogram bounds.\n";

    static constexpr const char* raw_names[17] = {
      "deep_output", "bypass_output", "signed_deep_bypass",
      "deep_bypass_disagreement", "selected_bucket",
      "router_top1_logit", "router_top2_logit", "router_margin",
      "phase.main_sqr_scale", "phase.main_raw_scale", "phase.diff_scale",
      "phase.abs_raw_scale", "phase.abs_sqr_scale", "phase.cross_scale",
      "phase.main_reliance", "phase.fm_reliance", "phase.cross_reliance"};
    out << "[fresh NNUE signal ranges]\n";
    for (std::size_t i = 0; i < 17; ++i) {
        const auto& summary = stats.raw_signal[i];
        out << "  " << std::left << std::setw(28) << raw_names[i] << std::right
            << " count=" << summary.count;
        if (summary.count)
            out << " min=" << summary.minimum
                << " mean=" << static_cast<double>(summary.sum / summary.count)
                << " max=" << summary.maximum;
        out << '\n';
    }
    out << "  selected_bucket counts       :";
    for (std::size_t i = 0; i < stats.selected_bucket.size(); ++i)
        out << ' ' << i << '=' << stats.selected_bucket[i];
    out << '\n';

    for (std::size_t kind = 0; kind < kSignalKinds; ++kind) {
        out << '[' << SignalName(kind) << "]\n"
            << "  Pearson(signal, abs_error): " << std::fixed << std::setprecision(6)
            << static_cast<double>(Pearson(stats.correlation[kind])) << '\n'
            << "  signal-bin              count   mean-err median p90 p99 fail-high fail-low futility LMR-research\n";
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            const auto& value = stats.prediction[kind][bin];
            if (!value.count)
                continue;
            out << "  ";
            PrintSignalRange(out, kind, bin);
            out << " count=" << value.count
                << " mean=" << std::fixed << std::setprecision(3)
                << double(value.error_sum) / value.count
                << " median<=" << Quantile(value, 0.50)
                << " p90<=" << Quantile(value, 0.90)
                << " p99<=" << Quantile(value, 0.99)
                << " FH=" << Percent(value.fail_high, value.count) << '%'
                << " FL=" << Percent(value.fail_low, value.count) << '%'
                << " fut=" << Percent(value.futility_pruned, value.count) << '%'
                << " LMR-R=" << Percent(value.lmr_research, value.lmr) << "% ("
                << value.lmr_research << '/' << value.lmr << ')'
                << " depth=" << double(value.depth_sum) / value.count
                << " moves=" << double(value.move_count_sum) / value.count << "\n";
        }
    }

    out << "[signed deep-bypass vs signed search error]\n"
        << "  signed_search_error = returned_value - staticEval\n"
        << "  Pearson(signed_deep_bypass, signed_search_error): "
        << std::fixed << std::setprecision(6)
        << Pearson(stats.signed_deep_vs_signed_error) << '\n';
    static constexpr const char* direction_names[3] = {
      "deep < bypass", "deep == bypass", "deep > bypass"};
    for (std::size_t direction = 0; direction < 3; ++direction) {
        const auto& value = stats.signed_direction[direction];
        out << "  " << std::left << std::setw(15) << direction_names[direction] << std::right
            << " count=" << value.count;
        if (value.count)
            out << " error(+/0/-)=" << Percent(value.search_error_positive, value.count)
                << "%/" << Percent(value.search_error_zero, value.count)
                << "%/" << Percent(value.search_error_negative, value.count)
                << "% mean-signed=" << value.signed_error_sum / value.count
                << " mean-abs=" << double(value.abs_error_sum) / value.count
                << " FH=" << Percent(value.fail_high, value.count) << '%'
                << " FL=" << Percent(value.fail_low, value.count) << '%';
        out << '\n';
    }

    static constexpr std::array<const char*, kConditionBins> depth_labels = {
      "depth 1-2", "depth 3-4", "depth 5-6", "depth 7-8", "depth 9+"};
    static constexpr std::array<const char*, kConditionBins> magnitude_labels = {
      "|staticEval| <100", "|staticEval| 100-299", "|staticEval| 300-599",
      "|staticEval| 600-1199", "|staticEval| 1200+"};
    static constexpr std::array<const char*, kBucketCount> bucket_labels = {
      "B00", "B01", "B02", "B03", "B04", "B05",
      "B06", "B07", "B08", "B09", "B10", "B11"};
    static constexpr std::array<const char*, kConditionBins> move_labels = {
      "move 1-2", "move 3-4", "move 5-8", "move 9-16", "move 17+"};

    PrintConditionalPrediction(out, "disagreement conditioned by depth", depth_labels,
                               stats.disagreement_by_depth);
    PrintConditionalPrediction(out, "disagreement conditioned by |staticEval|",
                               magnitude_labels, stats.disagreement_by_static_eval);
    PrintConditionalPrediction(out, "disagreement conditioned by selected bucket",
                               bucket_labels, stats.disagreement_by_bucket);

    out << "[reverse-futility-eligible nodes by disagreement]\n"
        << "  distance = eval - futility_margin - beta; condition rate also requires eval >= beta\n";
    for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
        const auto& value = stats.futility_eligible[bin];
        if (!value.count)
            continue;
        out << "  disagreement=";
        PrintSignalRange(out, 0, bin);
        out << " count=" << value.count
            << " mean-distance=" << value.distance_sum / value.count
            << " condition=" << Percent(value.condition_true, value.count) << '%'
            << " return(+/0/- vs static)="
            << Percent(value.returned_above_static, value.count) << "%/"
            << Percent(value.returned_equal_static, value.count) << "%/"
            << Percent(value.returned_below_static, value.count) << '%'
            << " FH=" << Percent(value.fail_high, value.count) << '%'
            << " FL=" << Percent(value.fail_low, value.count) << "%\n";
    }

    PrintConditionalLmr(out, "router-margin LMR events conditioned by depth", depth_labels,
                        stats.router_lmr_by_depth);
    PrintConditionalLmr(out, "router-margin LMR events conditioned by move count", move_labels,
                        stats.router_lmr_by_move_count);
    PrintConditionalLmr(out, "router-margin LMR events conditioned by selected bucket",
                        bucket_labels, stats.router_lmr_by_bucket);

    static constexpr const char* control_names[4] = {
      "depth", "|staticEval|", "|materialValue|", "ply"};
    out << "[control-signal Pearson correlations with abs_error]\n";
    for (std::size_t i = 0; i < 4; ++i)
        out << "  " << std::left << std::setw(18) << control_names[i] << std::right
            << std::fixed << std::setprecision(6) << Pearson(stats.control_correlation[i])
            << " (n=" << stats.control_correlation[i].count << ")\n";

    out << "[phase correlation within depth x |staticEval| x |materialValue| strata]\n"
        << "  strata: depth={1-2,3-4,5-6,7-8,9+}; magnitude bins={0-99,100-299,300-599,600-1199,1200+}\n";
    for (std::size_t phase = 0; phase < 3; ++phase) {
        std::uint64_t samples = 0;
        const double correlation = PhaseWithinStrataPearson(stats, phase, samples);
        out << "  " << std::left << std::setw(24) << SignalName(phase + 2) << std::right
            << std::fixed << std::setprecision(6) << correlation
            << " (within-strata n=" << samples << ")\n";
    }

#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
    const auto& experiment = stats.router_lmr_experiment;
    out << "[Router-margin LMR experiment]\n"
        << "  variant                  : " << NNUE_ROUTER_LMR_VARIANT
        << (NNUE_ROUTER_LMR_VARIANT == 0 ? " (current LMR)"
            : NNUE_ROUTER_LMR_VARIANT == 1 ? " (broad fixed-point delta)"
            : NNUE_ROUTER_LMR_VARIANT == 2 ? " (limited fixed-point delta)"
                                           : " (limited explicit +1 ply)") << '\n'
        << "  margin threshold         : " << NNUE_ROUTER_LMR_MARGIN_THRESHOLD << '\n'
        << "  fixed reduction change   : -" << NNUE_ROUTER_LMR_REDUCTION_DELTA
        << "/1024 ply\n"
        << "  all LMR moves            : " << experiment.lmr_moves << '\n'
        << "  signal-valid LMR moves   : " << experiment.signal_valid_moves << " ("
        << Percent(experiment.signal_valid_moves, experiment.lmr_moves) << "%)\n"
        << "  router high-risk moves   : " << experiment.high_risk_moves << " ("
        << Percent(experiment.high_risk_moves, experiment.lmr_moves)
        << "% of all LMR moves)\n"
        << "  adjusted nodes / moves   : " << experiment.adjusted_nodes << " / "
        << experiment.adjusted_moves << " ("
        << Percent(experiment.adjusted_moves, experiment.lmr_moves)
        << "% of all LMR moves)\n"
        << "  all LMR fail-high rate   : "
        << Percent(experiment.all_reduced_fail_highs, experiment.lmr_moves) << "%\n"
        << "  all LMR re-search rate   : "
        << Percent(experiment.all_researches, experiment.lmr_moves) << "%\n"
        << "  reduced fail-high rate   : "
        << Percent(experiment.reduced_fail_highs, experiment.adjusted_moves) << "%\n"
        << "  re-search rate           : "
        << Percent(experiment.researches, experiment.adjusted_moves) << "%\n"
        << "  fixed units removed      : " << experiment.fixed_reduction_removed << '\n'
        << "  actual depth +1 moves    : " << experiment.search_depth_increase_moves
        << " (" << Percent(experiment.search_depth_increase_moves, experiment.adjusted_moves)
        << "% of adjusted moves)\n"
        << "  total search-depth gain  : " << experiment.search_depth_increase << " ply\n"
        << "  adjusted by bucket       :";
    for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
        out << " B" << std::setfill('0') << std::setw(2) << bucket << std::setfill(' ')
            << '=' << experiment.adjusted_by_bucket[bucket];
    out << "\n  adjusted by margin bin   :\n";
    for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
        if (!experiment.adjusted_by_margin[bin])
            continue;
        out << "    ";
        PrintSignalRange(out, 1, bin);
        out << " : " << experiment.adjusted_by_margin[bin] << '\n';
    }
    out << "  eligible depth-effect cohorts"
        << (NNUE_ROUTER_LMR_VARIANT == 0 ? " (counterfactual C predicate)\n" : "\n");
    const char* cohort_labels[2] = {
      "A: delta applied, integer depth unchanged",
      "B: delta applied, actual/counterfactual +1 ply"};
    // Print the actual/+1 cohort first: if a long diagnostics stream is interrupted,
    // the primary counterfactual comparison remains available in the console log.
    for (std::size_t order = 0; order < 2; ++order) {
        const std::size_t cohort_index = 1 - order;
        const auto& cohort = experiment.eligible_depth_cohort[cohort_index];
        const double mean_delta = cohort.research_score_samples
          ? double(cohort.research_score_delta_sum) / double(cohort.research_score_samples) : 0.0;
        const double mean_abs_delta = cohort.research_score_samples
          ? double(cohort.research_score_abs_delta_sum) / double(cohort.research_score_samples) : 0.0;
        out << "    " << cohort_labels[cohort_index] << '\n'
            << "      count                    : " << cohort.moves << '\n'
            << "      reduced-search fail-high : "
            << Percent(cohort.reduced_fail_highs, cohort.moves) << "%\n"
            << "      re-search rate           : "
            << Percent(cohort.researches, cohort.moves) << "%\n"
            << "      re-search score delta    : mean_signed=" << mean_delta
            << " mean_abs=" << mean_abs_delta
            << " max_abs=" << cohort.research_score_abs_delta_max
            << " samples=" << cohort.research_score_samples << '\n'
            << "      final beta exceeded      : "
            << Percent(cohort.beta_exceeded, cohort.moves) << "%\n"
            << "      final cutoff             : "
            << Percent(cohort.final_cutoffs, cohort.moves) << "%\n";
    }
    out << "  original r modulo 1024 (Router predicate eligible)\n";
    for (std::size_t bin = 0; bin < 8; ++bin) {
        const auto& cohort = experiment.original_remainder[bin];
        const double mean_abs_delta = cohort.research_score_samples
          ? double(cohort.research_score_abs_delta_sum) / double(cohort.research_score_samples) : 0.0;
        const double mean_depth = cohort.moves ? double(cohort.depth_sum) / double(cohort.moves) : 0.0;
        const double mean_move_count = cohort.moves
          ? double(cohort.move_count_sum) / double(cohort.moves) : 0.0;
        const double mean_margin = cohort.moves
          ? double(cohort.router_margin_sum) / double(cohort.moves) : 0.0;
        out << "    " << (bin * 128) << '-' << (bin * 128 + 127)
            << " count=" << cohort.moves
            << " FH=" << Percent(cohort.reduced_fail_highs, cohort.moves) << '%'
            << " re-search=" << Percent(cohort.researches, cohort.moves) << '%'
            << " mean_abs_score_delta=" << mean_abs_delta
            << " beta=" << Percent(cohort.beta_exceeded, cohort.moves) << '%'
            << " cutoff=" << Percent(cohort.final_cutoffs, cohort.moves) << '%'
            << " depth=" << mean_depth
            << " moveCount=" << mean_move_count
            << " margin=" << mean_margin
            << " buckets=";
        for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
            out << (bucket ? "," : "") << "B" << std::setfill('0') << std::setw(2)
                << bucket << std::setfill(' ') << ':' << cohort.selected_bucket[bucket];
        out << '\n';
    }
    const std::uint64_t eligible_moves = [&]() {
        std::uint64_t count = 0;
        for (const auto& cohort : experiment.original_remainder)
            count += cohort.moves;
        return count;
    }();
    out << "  counterfactual reduction deltas (variant=0 search unchanged)\n";
    const int delta_values[4] = {256, 512, 768, 1024};
    for (std::size_t delta_index = 0; delta_index < 4; ++delta_index) {
        const auto& cohort = experiment.counterfactual_delta[delta_index];
        out << "    delta=" << delta_values[delta_index]
            << " +1_count=" << cohort.moves
            << " eligible_rate=" << Percent(cohort.moves, eligible_moves) << '%'
            << " all_LMR_rate=" << Percent(cohort.moves, experiment.lmr_moves) << '%'
            << " FH=" << Percent(cohort.reduced_fail_highs, cohort.moves) << '%'
            << " re-search=" << Percent(cohort.researches, cohort.moves) << '%'
            << " remainder_counts=";
        for (std::size_t bin = 0; bin < 8; ++bin)
            out << (bin ? "," : "") << (bin * 128) << '-' << (bin * 128 + 127)
                << ':' << experiment.counterfactual_delta_by_remainder[delta_index][bin];
        out << '\n';
    }
    out << "  Router-margin threshold cohorts (variant=0, C predicate fixed)\n";
    const int margin_thresholds[4] = {128, 256, 512, 1024};
    for (std::size_t threshold_index = 0; threshold_index < 4; ++threshold_index) {
        const auto& cohort = experiment.margin_threshold_cohort[threshold_index];
        const double mean_abs_delta = cohort.research_score_samples
          ? double(cohort.research_score_abs_delta_sum) / double(cohort.research_score_samples) : 0.0;
        out << "    threshold<=" << margin_thresholds[threshold_index]
            << " count=" << cohort.moves
            << " all_LMR_rate=" << Percent(cohort.moves, experiment.lmr_moves) << '%'
            << " FH=" << Percent(cohort.reduced_fail_highs, cohort.moves) << '%'
            << " re-search=" << Percent(cohort.researches, cohort.moves) << '%'
            << " mean_abs_score_delta=" << mean_abs_delta
            << " beta=" << Percent(cohort.beta_exceeded, cohort.moves) << '%'
            << " cutoff=" << Percent(cohort.final_cutoffs, cohort.moves) << "%\n"
            << "      buckets=";
        for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
            out << (bucket ? "," : "") << "B" << std::setfill('0') << std::setw(2)
                << bucket << std::setfill(' ') << ':' << cohort.selected_bucket[bucket];
        out << "\n      depth=";
        for (std::size_t depth = 0; depth < cohort.depth_3_to_8.size(); ++depth)
            out << (depth ? "," : "") << (depth + 3) << ':' << cohort.depth_3_to_8[depth];
        out << "\n      moveCount=";
        for (std::size_t move = 0; move < cohort.move_count_2_to_8.size(); ++move)
            out << (move ? "," : "") << (move + 2) << ':' << cohort.move_count_2_to_8[move];
        out << '\n';
    }
    out << "  note: searched nodes, NPS, and completed depth are reported by the normal bench output.\n";
#endif
}

}  // namespace YaneuraOu::Search::NnueSignalLog

#endif  // defined(ENABLE_NNUE_SIGNAL_LOG)
#endif  // YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED
