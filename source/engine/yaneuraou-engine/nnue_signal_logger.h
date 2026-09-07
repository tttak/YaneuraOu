// Search-only diagnostics for NNUE internal signals.
// Everything in this file disappears unless ENABLE_NNUE_SIGNAL_LOG is set.

#ifndef YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_SIGNAL_LOG)

#include <memory>

#ifndef NNUE_COMBINED_LCA_SUM_THRESHOLD
// Per-build calibration for the combined Router/LCA diagnostic.
// 295/epoch20 top 1%: 1897. 297/epoch0 top 1%: 1230.
#define NNUE_COMBINED_LCA_SUM_THRESHOLD 1897
#endif
#if NNUE_COMBINED_LCA_SUM_THRESHOLD < 0 || NNUE_COMBINED_LCA_SUM_THRESHOLD > 4064
#error "NNUE_COMBINED_LCA_SUM_THRESHOLD must be in [0, 4064]"
#endif

#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
#ifndef NNUE_ROUTER_LMR_VARIANT
#define NNUE_ROUTER_LMR_VARIANT 3
#endif
#ifndef NNUE_ROUTER_LMR_MARGIN_THRESHOLD
#define NNUE_ROUTER_LMR_MARGIN_THRESHOLD 256
#endif
#ifndef NNUE_ROUTER_LMR_REDUCTION_DELTA
#define NNUE_ROUTER_LMR_REDUCTION_DELTA 256
#endif
#if NNUE_ROUTER_LMR_VARIANT < 0 || NNUE_ROUTER_LMR_VARIANT > 3
#error "NNUE_ROUTER_LMR_VARIANT must be 0 (current), 1/2 (fixed-point delta), or 3 (explicit +1 ply)"
#endif
#endif

#if defined(ENABLE_NNUE_LCA_LMR_EXPERIMENT)
#ifndef NNUE_LCA_LMR_VARIANT
#define NNUE_LCA_LMR_VARIANT 0
#endif

#ifndef NNUE_LCA_LMR_SUM_THRESHOLD
// 295/epoch20 top-1% cutoff: mean 59.28125 == exact delta sum 1897 / 32.
#define NNUE_LCA_LMR_SUM_THRESHOLD 1897
#endif
#if NNUE_LCA_LMR_VARIANT < 0 || NNUE_LCA_LMR_VARIANT > 1
#error "NNUE_LCA_LMR_VARIANT must be 0 (shadow current) or 1 (explicit +1 ply)"
#endif
#if NNUE_LCA_LMR_SUM_THRESHOLD < 0 || NNUE_LCA_LMR_SUM_THRESHOLD > 4064
#error "NNUE_LCA_LMR_SUM_THRESHOLD must be in [0, 4064]"
#endif
#endif

#if defined(ENABLE_NNUE_CROSS_LMR_EXPERIMENT)
#ifndef NNUE_CROSS_LMR_VARIANT
#define NNUE_CROSS_LMR_VARIANT 0
#endif
#ifndef NNUE_CROSS_LMR_MAX_THRESHOLD
#define NNUE_CROSS_LMR_MAX_THRESHOLD 127
#endif
#if NNUE_CROSS_LMR_VARIANT < 0 || NNUE_CROSS_LMR_VARIANT > 1
#error "NNUE_CROSS_LMR_VARIANT must be 0 (shadow) or 1 (explicit +1 ply)"
#endif
#endif

#if defined(ENABLE_NNUE_RFP_SHADOW)
#ifndef NNUE_RFP_SHADOW_SAMPLE_SHIFT
// Shadow one out of every 256 signal-valid reverse-futility cut candidates.
#define NNUE_RFP_SHADOW_SAMPLE_SHIFT 8
#endif
#if NNUE_RFP_SHADOW_SAMPLE_SHIFT < 0 || NNUE_RFP_SHADOW_SAMPLE_SHIFT > 20
#error "NNUE_RFP_SHADOW_SAMPLE_SHIFT must be in [0, 20]"
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
#include <new>
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
constexpr std::size_t kFmRelianceGroups = 3;
constexpr std::size_t kFmRelianceFineGroups = 5;
constexpr std::size_t kRouterMarginGroups = 2;
constexpr std::size_t kLcaMeanDeltaBins = 10;
constexpr std::size_t kLcaMaxDeltaBins = 9;
constexpr std::size_t kLcaMagnitudeGroups = 4;
constexpr std::size_t kLcaThresholdCount = 5;
constexpr std::array<float, kLcaThresholdCount> kLcaThresholds = {
  16.0f, 24.0f, 32.0f, 48.0f, 64.0f};
constexpr std::size_t kLcaDeltaSumMax = 32 * 127;
constexpr std::size_t kLcaPercentileCount = 5;
constexpr std::size_t kCombinedSignalGroups = 4;
constexpr std::size_t kDiffRmsEnergyBins = 10;
constexpr std::array<float, kDiffRmsEnergyBins - 1> kDiffRmsEdges = {
  512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f,
  16384.0f, 32768.0f, 65536.0f, 131072.0f};
constexpr std::size_t kCrossActivityBins = 10;
constexpr std::size_t kCrossOverlapGroups = 8;
constexpr std::size_t kCrossDeployableThresholdCount = 3;
constexpr std::array<std::uint8_t, kCrossDeployableThresholdCount>
  kCrossDeployableThresholds = {125, 126, 127};
constexpr std::array<std::uint16_t, kCrossActivityBins - 1> kCrossMeanAbsEdges = {
  4, 8, 16, 24, 32, 48, 64, 80, 96};
constexpr std::array<std::uint8_t, kCrossActivityBins - 1> kCrossMaxAbsEdges = {
  16, 32, 48, 64, 80, 96, 112, 120, 126};
constexpr std::size_t kMainGateSignalCount = 5;
constexpr std::size_t kMainGateActivityBins = 8;
constexpr std::size_t kFmActivitySignalCount = 6;
constexpr std::size_t kFmActivityBins = 8;
constexpr std::size_t kCalibrationRouterBins = 10;
constexpr std::size_t kCalibrationRouterThresholdCount = 4;
constexpr std::array<std::int32_t, kCalibrationRouterThresholdCount>
  kCalibrationRouterThresholds = {128, 256, 512, 1024};
constexpr std::size_t kCalibrationCrossThresholdCount = 3;
constexpr std::array<std::uint8_t, kCalibrationCrossThresholdCount>
  kCalibrationCrossThresholds = {125, 126, 127};
constexpr std::array<std::uint32_t, kLcaPercentileCount> kLcaTopBasisPoints = {
  100, 250, 500, 1000, 2000};
constexpr std::size_t kFmJointBaseStrata =
  kConditionBins * kConditionBins * kBucketCount * kConditionBins * kRouterMarginGroups;
constexpr std::size_t kMainGateJointBaseStrata = kFmJointBaseStrata * 2 * 2;

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

#if defined(ENABLE_NNUE_RFP_SHADOW)
struct ReverseFutilityShadowBucket {
    std::uint64_t rfp_conditions = 0;
    std::uint64_t shadow_samples = 0;
    std::uint64_t completed_samples = 0;
    std::uint64_t correct_cuts = 0;
    std::uint64_t wrong_cuts = 0;
    std::int64_t result_minus_beta_sum = 0;
    std::int64_t result_minus_static_sum = 0;
    std::int64_t result_minus_beta_min = std::numeric_limits<std::int64_t>::max();
    std::int64_t result_minus_beta_max = std::numeric_limits<std::int64_t>::lowest();
    std::int64_t result_minus_static_min = std::numeric_limits<std::int64_t>::max();
    std::int64_t result_minus_static_max = std::numeric_limits<std::int64_t>::lowest();
};
#endif

struct LmrConditionalBucket {
    std::uint64_t events = 0;
    std::uint64_t researches = 0;
};

// Per-move LMR outcomes used only by signal diagnostics.  Unlike the older
// node-level LMR table, these counters follow one reduced move through an
// optional re-search and the final beta/cutoff decision.
struct PhaseFmLmrBucket {
    std::uint64_t moves = 0;
    std::uint64_t reduced_fail_highs = 0;
    std::uint64_t researches = 0;
    std::uint64_t beta_exceeded = 0;
    std::uint64_t final_cutoffs = 0;
    std::uint64_t research_score_samples = 0;
    std::uint64_t research_score_abs_delta_sum = 0;
    std::uint64_t research_score_abs_delta_max = 0;
};

template<std::size_t Bins>
struct ConditionedLmrSignalStats {
    std::array<PhaseFmLmrBucket, Bins> all{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, kConditionBins> by_depth{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, kConditionBins> by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, kBucketCount> by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, kConditionBins> by_static_eval{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, kRouterMarginGroups> by_router{};
    std::array<std::array<PhaseFmLmrBucket, Bins>, 2> by_lca_tail{};
    std::array<PhaseFmLmrBucket, kFmJointBaseStrata * 2 * Bins> joint{};
};

struct MainGateActivityStats {
    std::array<PhaseFmLmrBucket, kMainGateActivityBins> all{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, kConditionBins>
      by_depth{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, kConditionBins>
      by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, kBucketCount>
      by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, kConditionBins>
      by_static_eval{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, kRouterMarginGroups>
      by_router{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, 2> by_lca_tail{};
    std::array<std::array<PhaseFmLmrBucket, kMainGateActivityBins>, 2>
      by_cross_saturation{};
    // Moves untouched by all three deployed NNUE-LMR signals. This measures
    // whether a gate signal would add coverage rather than rediscovering them.
    std::array<PhaseFmLmrBucket, kMainGateActivityBins> uncovered{};
    // Flat index: depth x moveCount x bucket x |staticEval| x router group x
    // calibrated LCA tail x Cross saturation x activity bin.
    std::array<PhaseFmLmrBucket,
               kMainGateJointBaseStrata * kMainGateActivityBins> joint{};
};

// FM activity uses the same conditioning axes and eight-bin layout as Main
// gate activity, but stores six independently classified summaries.
using FmActivityStats = MainGateActivityStats;

struct SignalCalibrationStats {
    std::uint64_t signal_valid_lmr_moves = 0;
    std::uint64_t router_structural_moves = 0;
    std::uint64_t router_actual_adjusted_moves = 0;
    std::uint64_t lca_calibration_eligible_moves = 0;
    std::uint64_t cross_structural_eligible_moves = 0;
    std::array<PhaseFmLmrBucket, kCalibrationRouterBins> router_distribution{};
    std::array<PhaseFmLmrBucket, kCalibrationRouterThresholdCount> router_cohorts{};
    std::array<PhaseFmLmrBucket, kLcaDeltaSumMax + 1> lca_eligible_by_delta_sum{};
    std::array<PhaseFmLmrBucket, kCalibrationCrossThresholdCount> cross_all_cohorts{};
    std::array<PhaseFmLmrBucket, kLcaDeltaSumMax + 1>
      cross_after_router_all_by_lca_sum{};
    // Cross threshold x exact LCA sum, after Router and positive-reduction
    // eligibility. Report-time LCA calibration removes the top tail from this
    // table to reproduce the sequential Router -> LCA -> Cross deployment.
    std::array<std::array<PhaseFmLmrBucket, kLcaDeltaSumMax + 1>,
               kCalibrationCrossThresholdCount> cross_after_router_by_lca_sum{};
    // Exact LCA sum x {Router danger, Cross danger}.  The LCA top-1% cutoff is
    // unknown until the run is complete, so keeping this joint histogram lets
    // the report derive model-calibrated overlap cohorts without a fixed sum.
    // Low two bits: bit0=Router production predicate, bit1=Cross production
    // structural predicate (depth/moveCount and max>=127).
    std::array<PhaseFmLmrBucket, (kLcaDeltaSumMax + 1) * 4> overlap_by_lca_sum{};
};

// Counterfactual LCA cohorts. These are diagnostic-only observations of moves
// which production Router-LMR did not actually deepen; they never affect d/r.
struct LcaThresholdCohortStats {
    PhaseFmLmrBucket total{};
    std::array<PhaseFmLmrBucket, kConditionBins> by_depth{};
    std::array<PhaseFmLmrBucket, kConditionBins> by_move_count{};
    std::array<PhaseFmLmrBucket, kBucketCount> by_bucket{};
    std::array<PhaseFmLmrBucket, kConditionBins> by_static_eval{};
    std::array<PhaseFmLmrBucket, kRouterMarginGroups> by_router_margin{};
};

struct PhaseFmLmrExperimentStats {
    std::uint64_t candidate_moves = 0;
    std::uint64_t adjusted_moves = 0;
    PhaseFmLmrBucket candidate_outcomes{};
    PhaseFmLmrBucket adjusted_outcomes{};
    std::array<std::uint64_t, kBucketCount> adjusted_by_bucket{};
    std::array<std::uint64_t, kConditionBins> adjusted_by_depth{};
    std::array<std::uint64_t, kConditionBins> adjusted_by_move_count{};
};

struct LcaLmrExperimentStats {
    PhaseFmLmrBucket candidate_outcomes{};
    PhaseFmLmrBucket adjusted_outcomes{};
    std::array<PhaseFmLmrBucket, kBucketCount> candidate_by_bucket{};
    std::array<PhaseFmLmrBucket, kConditionBins> candidate_by_depth{};
    std::array<PhaseFmLmrBucket, kConditionBins> candidate_by_move_count{};
    std::array<PhaseFmLmrBucket, kConditionBins> candidate_by_static_eval{};
    std::array<PhaseFmLmrBucket, kBucketCount> adjusted_by_bucket{};
    std::array<PhaseFmLmrBucket, kConditionBins> adjusted_by_depth{};
    std::array<PhaseFmLmrBucket, kConditionBins> adjusted_by_move_count{};
    std::array<PhaseFmLmrBucket, kConditionBins> adjusted_by_static_eval{};
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
    std::array<BasicSummary, 19> raw_signal{};
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
#if defined(ENABLE_NNUE_RFP_SHADOW)
    std::uint64_t rfp_conditions = 0;
    std::uint64_t rfp_signal_valid_conditions = 0;
    std::uint64_t rfp_shadow_sequence = 0;
    std::array<ReverseFutilityShadowBucket, kSignalBins> rfp_shadow_by_disagreement{};
    std::array<std::array<ReverseFutilityShadowBucket, kSignalBins>, kConditionBins>
      rfp_shadow_by_static_eval{};
#endif
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kConditionBins>
      router_lmr_by_depth{};
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kConditionBins>
      router_lmr_by_move_count{};
    std::array<std::array<LmrConditionalBucket, kSignalBins>, kBucketCount>
      router_lmr_by_bucket{};
    std::uint64_t phase_fm_lmr_moves = 0;
    std::uint64_t phase_fm_lmr_signal_valid_moves = 0;
    std::array<PhaseFmLmrBucket, kSignalBins> phase_fm_lmr{};
    std::array<std::array<PhaseFmLmrBucket, kSignalBins>, kConditionBins>
      phase_fm_lmr_by_depth{};
    std::array<std::array<PhaseFmLmrBucket, kSignalBins>, kConditionBins>
      phase_fm_lmr_by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, kSignalBins>, kBucketCount>
      phase_fm_lmr_by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, kSignalBins>, kConditionBins>
      phase_fm_lmr_by_static_eval{};
    std::array<std::array<PhaseFmLmrBucket, kFmRelianceGroups>, kRouterMarginGroups>
      phase_fm_lmr_by_router_and_fm{};
    // Flat index: (depth, moveCount, bucket, |staticEval|, router group, FM group).
    std::array<PhaseFmLmrBucket, kFmJointBaseStrata * kFmRelianceGroups>
      phase_fm_lmr_joint{};
    // [actual Router +1 ply][fine FM reliance group].
    std::array<std::array<PhaseFmLmrBucket, kFmRelianceFineGroups>, 2>
      phase_fm_lmr_by_router_adjustment{};
    // [actual Router +1 ply][|staticEval|][fine FM reliance group].
    std::array<std::array<std::array<PhaseFmLmrBucket, kFmRelianceFineGroups>,
                          kConditionBins>, 2>
      phase_fm_lmr_by_router_adjustment_and_static{};
    // [staticEval sign: >=+1200, <=-1200][fine FM reliance group].
    // Only moves not actually deepened by production Router-LMR are recorded.
    std::array<std::array<PhaseFmLmrBucket, kFmRelianceFineGroups>, 2>
      phase_fm_lmr_extreme_by_sign{};
    // [staticEval sign][fine FM reliance group][selected bucket].
    std::array<std::array<std::array<PhaseFmLmrBucket, kBucketCount>,
                          kFmRelianceFineGroups>, 2>
      phase_fm_lmr_extreme_by_sign_and_bucket{};
    PhaseFmLmrExperimentStats phase_fm_lmr_experiment{};
    LcaLmrExperimentStats lca_lmr_experiment{};
    LcaLmrExperimentStats cross_lmr_experiment{};
    std::uint64_t lca_lmr_moves = 0;
    std::uint64_t lca_lmr_signal_valid_moves = 0;
    PhaseFmLmrBucket lca_lmr_all_outcomes{};
    std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins> lca_lmr_by_mean_delta{};
    std::array<PhaseFmLmrBucket, kLcaMaxDeltaBins> lca_lmr_by_max_delta{};
    std::array<std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins>, kConditionBins>
      lca_lmr_by_depth{};
    std::array<std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins>, kConditionBins>
      lca_lmr_by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins>, kBucketCount>
      lca_lmr_by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins>, kConditionBins>
      lca_lmr_by_static_eval{};
    std::array<std::array<PhaseFmLmrBucket, kLcaMagnitudeGroups>, kRouterMarginGroups>
      lca_lmr_by_router_and_magnitude{};
    // Flat index: (depth, moveCount, bucket, |staticEval|, router group, LCA group).
    std::array<PhaseFmLmrBucket, kFmJointBaseStrata * kLcaMagnitudeGroups>
      lca_lmr_joint{};
    std::array<LcaThresholdCohortStats, kLcaThresholdCount>
      lca_non_router_thresholds{};
    PhaseFmLmrBucket lca_non_router_all{};
    // Exact byte-domain delta-sum histogram (mean = sum / 32).  This permits
    // per-network percentile thresholds without feeding a calibrated value
    // back into production search.
    std::array<PhaseFmLmrBucket, kLcaDeltaSumMax + 1>
      lca_non_router_by_delta_sum{};
    // A=Router only, B=LCA only, C=both, D=neither. Keep Router danger exactly
    // aligned with the production variant-3 predicate: non-PV, configured
    // margin threshold, depth 3..8, moveCount 2..8, no bucket exclusion.
    std::array<PhaseFmLmrBucket, kCombinedSignalGroups> combined_signal{};
    std::array<std::array<PhaseFmLmrBucket, kConditionBins>, kCombinedSignalGroups>
      combined_signal_by_depth{};
    std::array<std::array<PhaseFmLmrBucket, kConditionBins>, kCombinedSignalGroups>
      combined_signal_by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, kBucketCount>, kCombinedSignalGroups>
      combined_signal_by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, kConditionBins>, kCombinedSignalGroups>
      combined_signal_by_static_eval{};
    // Diff RMS energy reuses the existing RMSNorm sum-of-squares. Detailed
    // bins are expressed as equivalent RMS ranges, but classification compares
    // sum_sq directly and therefore performs no diagnostic sqrt or reduction.
    std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins> diff_rms_lmr{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, kConditionBins>
      diff_rms_lmr_by_depth{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, kConditionBins>
      diff_rms_lmr_by_move_count{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, kBucketCount>
      diff_rms_lmr_by_bucket{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, kConditionBins>
      diff_rms_lmr_by_static_eval{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, kRouterMarginGroups>
      diff_rms_lmr_by_router{};
    std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, 2>
      diff_rms_lmr_by_lca_tail{};
    // Flat index: (depth, moveCount, bucket, |staticEval|, router group,
    // calibrated LCA high-tail, Diff RMS energy bin).
    std::array<PhaseFmLmrBucket,
               kFmJointBaseStrata * 2 * kDiffRmsEnergyBins> diff_rms_lmr_joint{};
    ConditionedLmrSignalStats<kCrossActivityBins> cross_mean_abs_lmr{};
    ConditionedLmrSignalStats<kCrossActivityBins> cross_max_abs_lmr{};
    // Bit layout: bit2=Cross max>=126, bit1=Router actual +1 ply,
    // bit0=LCA. Raw uses the calibrated LCA threshold irrespective of Router;
    // eligible uses the actual shadow application predicate after Router.
    std::array<PhaseFmLmrBucket, kCrossOverlapGroups> cross_overlap_raw{};
    std::array<PhaseFmLmrBucket, kCrossOverlapGroups> cross_overlap_eligible{};
    // Counterfactual Cross cohorts which neither production Router-LMR nor
    // the calibrated LCA-LMR predicate would deepen. Thresholds are cumulative.
    std::array<LcaThresholdCohortStats, kCrossDeployableThresholdCount>
      cross_deployable_thresholds{};
    std::array<MainGateActivityStats, kMainGateSignalCount> main_gate_activity{};
    std::array<FmActivityStats, kFmActivitySignalCount> fm_activity{};
    SignalCalibrationStats signal_calibration{};
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

inline std::size_t LcaMeanDeltaBin(const float value) {
    if (value <= 0.0f) return 0;
    if (value < 0.25f) return 1;
    if (value < 0.50f) return 2;
    if (value < 1.00f) return 3;
    if (value < 2.00f) return 4;
    if (value < 4.00f) return 5;
    if (value < 8.00f) return 6;
    if (value < 16.0f) return 7;
    if (value < 32.0f) return 8;
    return 9;
}

inline std::size_t LcaMaxDeltaBin(const std::int32_t value) {
    if (value <= 0) return 0;
    if (value == 1) return 1;
    if (value == 2) return 2;
    if (value <= 4) return 3;
    if (value <= 8) return 4;
    if (value <= 16) return 5;
    if (value <= 32) return 6;
    if (value <= 64) return 7;
    return 8;
}

inline std::size_t LcaMagnitudeGroup(const float value) {
    if (value < 2.00f) return 0;
    if (value < 8.00f) return 1;
    if (value < 32.0f) return 2;
    return 3;
}

inline std::size_t DiffRmsEnergyBin(const float sum_sq) {
    // sum_sq is the already-computed sum over 32 value channels. Compare
    // against 32 * RMS^2 so diagnostics do not add sqrt or another reduction.
    for (std::size_t bin = 0; bin < kDiffRmsEdges.size(); ++bin)
        if (sum_sq < 32.0f * kDiffRmsEdges[bin] * kDiffRmsEdges[bin])
            return bin;
    return kDiffRmsEnergyBins - 1;
}

inline std::string DiffRmsEnergyLabel(const std::size_t bin) {
    if (bin == 0)
        return "RMS<" + std::to_string(static_cast<int>(kDiffRmsEdges[0]));
    if (bin + 1 == kDiffRmsEnergyBins)
        return "RMS>=" + std::to_string(
          static_cast<int>(kDiffRmsEdges[kDiffRmsEdges.size() - 1]));
    return "RMS[" + std::to_string(static_cast<int>(kDiffRmsEdges[bin - 1]))
         + "," + std::to_string(static_cast<int>(kDiffRmsEdges[bin])) + ")";
}

inline std::size_t CrossMeanAbsBin(const std::uint16_t sum) {
    for (std::size_t bin = 0; bin < kCrossMeanAbsEdges.size(); ++bin)
        if (sum < 32U * kCrossMeanAbsEdges[bin])
            return bin;
    return kCrossActivityBins - 1;
}

inline std::size_t CrossMaxAbsBin(const std::uint8_t maximum) {
    for (std::size_t bin = 0; bin < kCrossMaxAbsEdges.size(); ++bin)
        if (maximum < kCrossMaxAbsEdges[bin])
            return bin;
    return kCrossActivityBins - 1;
}

inline std::size_t MainGateActivityBin(
  const Eval::NNUE::NnueSignalSnapshot& signal, const std::size_t kind) {
    const auto count_bin = [](const std::uint8_t count) {
        if (count == 0) return std::size_t{0};
        if (count == 1) return std::size_t{1};
        if (count == 2) return std::size_t{2};
        if (count <= 4) return std::size_t{3};
        if (count <= 8) return std::size_t{4};
        if (count <= 16) return std::size_t{5};
        if (count <= 24) return std::size_t{6};
        return std::size_t{7};
    };
    switch (kind) {
    case 0: // Mean, classified by exact sum to avoid diagnostic float work.
        return std::min<std::size_t>(signal.main_gate_sum / (8U * 32U), 7U);
    case 1: {
        const auto value = signal.main_gate_min;
        if (value == 0) return 0;
        if (value == 1) return 1;
        if (value <= 3) return 2;
        if (value <= 7) return 3;
        if (value <= 15) return 4;
        if (value <= 31) return 5;
        if (value <= 47) return 6;
        return 7;
    }
    case 2: {
        const auto value = signal.main_gate_max;
        if (value < 16) return 0;
        if (value < 32) return 1;
        if (value < 48) return 2;
        if (value < 56) return 3;
        if (value < 60) return 4;
        if (value < 62) return 5;
        if (value < 63) return 6;
        return 7;
    }
    case 3: return count_bin(signal.main_gate_saturated_low_count);
    default: return count_bin(signal.main_gate_saturated_high_count);
    }
}

inline const char* MainGateActivityName(const std::size_t kind) {
    static constexpr std::array<const char*, kMainGateSignalCount> names = {
      "mean_main_gate", "min_main_gate", "max_main_gate",
      "saturated_low_count(gate<=1)", "saturated_high_count(gate>=63)"};
    return names[kind];
}

inline const char* MainGateActivityLabel(const std::size_t kind, const std::size_t bin) {
    static constexpr std::array<const char*, kMainGateActivityBins> mean_labels = {
      "mean[0,8)", "mean[8,16)", "mean[16,24)", "mean[24,32)",
      "mean[32,40)", "mean[40,48)", "mean[48,56)", "mean[56,64)"};
    static constexpr std::array<const char*, kMainGateActivityBins> min_labels = {
      "min=0", "min=1", "min=2-3", "min=4-7", "min=8-15", "min=16-31",
      "min=32-47", "min=48-63"};
    static constexpr std::array<const char*, kMainGateActivityBins> max_labels = {
      "max<16", "max=16-31", "max=32-47", "max=48-55", "max=56-59",
      "max=60-61", "max=62", "max=63"};
    static constexpr std::array<const char*, kMainGateActivityBins> count_labels = {
      "count=0", "count=1", "count=2", "count=3-4", "count=5-8",
      "count=9-16", "count=17-24", "count=25-32"};
    if (kind == 0) return mean_labels[bin];
    if (kind == 1) return min_labels[bin];
    if (kind == 2) return max_labels[bin];
    return count_labels[bin];
}

inline std::size_t FmActivityBin(
  const Eval::NNUE::NnueSignalSnapshot& signal, const std::size_t kind) {
    const auto count_bin = [](const std::uint8_t count) {
        if (count == 0) return std::size_t{0};
        if (count == 1) return std::size_t{1};
        if (count == 2) return std::size_t{2};
        if (count <= 4) return std::size_t{3};
        if (count <= 8) return std::size_t{4};
        if (count <= 16) return std::size_t{5};
        if (count <= 24) return std::size_t{6};
        return std::size_t{7};
    };
    const auto edge_bin = [](const auto value, const auto& edges) {
        for (std::size_t bin = 0; bin < edges.size(); ++bin)
            if (value < edges[bin])
                return bin;
        return edges.size();
    };
    static constexpr std::array<std::uint8_t, 7> diff_mean_edges = {
      4, 8, 12, 16, 24, 32, 48};
    static constexpr std::array<std::uint8_t, 7> diff_max_edges = {
      8, 16, 24, 32, 40, 48, 56};
    static constexpr std::array<std::uint8_t, 7> abs_mean_edges = {
      16, 32, 48, 64, 80, 96, 112};
    static constexpr std::array<std::uint8_t, 7> abs_max_edges = {
      32, 64, 80, 96, 112, 120, 126};
    switch (kind) {
    case 0:
        return edge_bin(signal.fm_diff_activity_sum / 32U, diff_mean_edges);
    case 1:
        return edge_bin(signal.fm_diff_activity_max, diff_max_edges);
    case 2:
        return count_bin(signal.fm_diff_saturated_count);
    case 3:
        return edge_bin(signal.fm_abs_activity_sum / 32U, abs_mean_edges);
    case 4:
        return edge_bin(signal.fm_abs_activity_max, abs_max_edges);
    default:
        return count_bin(signal.fm_abs_saturated_count);
    }
}

inline const char* FmActivityName(const std::size_t kind) {
    static constexpr std::array<const char*, kFmActivitySignalCount> names = {
      "mean_diff_activity", "max_diff_activity", "saturated_diff_count",
      "mean_abs_activity", "max_abs_activity", "saturated_abs_count"};
    return names[kind];
}

inline const char* FmActivityLabel(const std::size_t kind, const std::size_t bin) {
    static constexpr std::array<const char*, kFmActivityBins> diff_mean_labels = {
      "mean_abs_diff<4", "mean_abs_diff[4,8)", "mean_abs_diff[8,12)",
      "mean_abs_diff[12,16)", "mean_abs_diff[16,24)", "mean_abs_diff[24,32)",
      "mean_abs_diff[32,48)", "mean_abs_diff>=48"};
    static constexpr std::array<const char*, kFmActivityBins> diff_max_labels = {
      "max_abs_diff<8", "max_abs_diff[8,16)", "max_abs_diff[16,24)",
      "max_abs_diff[24,32)", "max_abs_diff[32,40)", "max_abs_diff[40,48)",
      "max_abs_diff[48,56)", "max_abs_diff>=56"};
    static constexpr std::array<const char*, kFmActivityBins> abs_mean_labels = {
      "mean_abs<16", "mean_abs[16,32)", "mean_abs[32,48)", "mean_abs[48,64)",
      "mean_abs[64,80)", "mean_abs[80,96)", "mean_abs[96,112)", "mean_abs>=112"};
    static constexpr std::array<const char*, kFmActivityBins> abs_max_labels = {
      "max_abs<32", "max_abs[32,64)", "max_abs[64,80)", "max_abs[80,96)",
      "max_abs[96,112)", "max_abs[112,120)", "max_abs[120,126)", "max_abs>=126"};
    static constexpr std::array<const char*, kFmActivityBins> count_labels = {
      "count=0", "count=1", "count=2", "count=3-4", "count=5-8",
      "count=9-16", "count=17-24", "count=25-32"};
    if (kind == 0) return diff_mean_labels[bin];
    if (kind == 1) return diff_max_labels[bin];
    if (kind == 3) return abs_mean_labels[bin];
    if (kind == 4) return abs_max_labels[bin];
    return count_labels[bin];
}

template<typename EdgeType, std::size_t Size>
inline std::string CrossActivityLabel(const char* name, const std::size_t bin,
                                      const std::array<EdgeType, Size>& edges) {
    if (bin == 0)
        return std::string(name) + "<" + std::to_string(edges[0]);
    if (bin == Size)
        return std::string(name) + ">=" + std::to_string(edges[Size - 1]);
    return std::string(name) + "[" + std::to_string(edges[bin - 1]) + ","
         + std::to_string(edges[bin]) + ")";
}

inline std::size_t FmRelianceGroup(const float value) {
    // Fixed, interpretable groups spanning the observed FM-reliance range.
    // They are diagnostic only and never affect search decisions.
    return value < 1.75f ? 0 : value < 2.0f ? 1 : 2;
}

inline std::size_t FmRelianceFineGroup(const float value) {
    if (value < 1.50f) return 0;
    if (value < 1.625f) return 1;
    if (value < 1.75f) return 2;
    if (value < 2.00f) return 3;
    return 4;
}

inline std::size_t RouterMarginGroup(const std::int32_t margin) {
    // Match the current conservative Router-LMR high-risk threshold.
    return margin <= 256 ? 0 : 1;
}

inline std::size_t CalibrationRouterBin(const std::int32_t margin) {
    static constexpr std::array<std::int32_t, kCalibrationRouterBins - 1> edges = {
      32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const auto non_negative = std::max<std::int32_t>(margin, 0);
    for (std::size_t bin = 0; bin < edges.size(); ++bin)
        if (non_negative <= edges[bin])
            return bin;
    return kCalibrationRouterBins - 1;
}

inline const char* CalibrationRouterBinLabel(const std::size_t bin) {
    static constexpr std::array<const char*, kCalibrationRouterBins> labels = {
      "<=32", "33-64", "65-128", "129-256", "257-512",
      "513-1024", "1025-2048", "2049-4096", "4097-8192", ">8192"};
    return labels[bin];
}

inline std::size_t FmJointBaseIndex(const std::size_t depth_bin,
                                    const std::size_t move_count_bin,
                                    const std::size_t bucket,
                                    const std::size_t static_eval_bin,
                                    const std::size_t router_group) {
    return (((depth_bin * kConditionBins + move_count_bin) * kBucketCount + bucket)
             * kConditionBins + static_eval_bin) * kRouterMarginGroups + router_group;
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
#if defined(ENABLE_NNUE_RFP_SHADOW)
        if (reverse_futility_shadow_sampled_ && has_result_) {
            const auto update_shadow_outcome = [&](ReverseFutilityShadowBucket& bucket) {
                const auto beta_delta = static_cast<std::int64_t>(result_) - beta_;
                const auto static_delta = static_cast<std::int64_t>(result_) - static_eval_;
                ++bucket.completed_samples;
                bucket.correct_cuts += result_ >= beta_;
                bucket.wrong_cuts += result_ < beta_;
                bucket.result_minus_beta_sum += beta_delta;
                bucket.result_minus_static_sum += static_delta;
                bucket.result_minus_beta_min =
                  std::min(bucket.result_minus_beta_min, beta_delta);
                bucket.result_minus_beta_max =
                  std::max(bucket.result_minus_beta_max, beta_delta);
                bucket.result_minus_static_min =
                  std::min(bucket.result_minus_static_min, static_delta);
                bucket.result_minus_static_max =
                  std::max(bucket.result_minus_static_max, static_delta);
            };
            update_shadow_outcome(
              stats.rfp_shadow_by_disagreement[reverse_futility_shadow_disagreement_bin_]);
            update_shadow_outcome(stats.rfp_shadow_by_static_eval
              [reverse_futility_shadow_static_eval_bin_]
              [reverse_futility_shadow_disagreement_bin_]);
        }
#endif

        if (access_.source == Eval::NNUE::NnueSignalEvalSource::FreshNetwork && valid) {
            const double values[19] = {
              double(access_.signal.deep_output), double(access_.signal.bypass_output),
              double(access_.signal.signed_deep_bypass),
              double(access_.signal.deep_bypass_disagreement),
              double(access_.signal.selected_bucket), double(access_.signal.router_top1_logit),
              double(access_.signal.router_top2_logit), double(access_.signal.router_margin),
              access_.signal.phase_scale[0], access_.signal.phase_scale[1],
              access_.signal.phase_scale[2], access_.signal.phase_scale[3],
              access_.signal.phase_scale[4], access_.signal.phase_scale[5],
              access_.signal.main_reliance, access_.signal.fm_reliance,
              access_.signal.cross_reliance, access_.signal.lca_mean_abs_delta,
              double(access_.signal.lca_max_abs_delta)};
            for (std::size_t i = 0; i < 19; ++i) {
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
#if defined(ENABLE_NNUE_RFP_SHADOW)
    bool SelectReverseFutilityShadowSample() {
        auto& stats = LocalStats();
        ++stats.rfp_conditions;

        // disagreement is meaningful only when the evaluation path supplied a
        // complete NNUE snapshot. Non-valid RFP cuts retain normal behavior.
        if (!access_.signal.valid || !has_static_eval_)
            return false;

        ++stats.rfp_signal_valid_conditions;
        const auto disagreement_bin = SignalBin(access_.signal, 0);
        const auto static_eval_bin = MagnitudeBin(static_eval_);
        auto& disagreement_bucket = stats.rfp_shadow_by_disagreement[disagreement_bin];
        auto& conditioned_bucket =
          stats.rfp_shadow_by_static_eval[static_eval_bin][disagreement_bin];
        ++disagreement_bucket.rfp_conditions;
        ++conditioned_bucket.rfp_conditions;

        constexpr std::uint64_t sample_mask =
          (std::uint64_t(1) << NNUE_RFP_SHADOW_SAMPLE_SHIFT) - 1;
        const bool selected = (stats.rfp_shadow_sequence++ & sample_mask) == 0;
        if (!selected)
            return false;

        ++disagreement_bucket.shadow_samples;
        ++conditioned_bucket.shadow_samples;
        reverse_futility_shadow_sampled_ = true;
        reverse_futility_shadow_disagreement_bin_ = disagreement_bin;
        reverse_futility_shadow_static_eval_bin_ = static_eval_bin;
        return true;
    }
#endif
    void RecordLmrEvent(const int move_count, const bool researched) {
        const auto bin = MoveCountBin(move_count);
        ++lmr_events_[bin];
        lmr_researches_[bin] += researched;
    }
    void RecordPhaseFmLmrOutcome(const int depth, const int move_count,
                                 const bool router_adjusted, const bool reduced_fail_high,
                                 const bool researched, const int reduced_value,
                                 const int final_value, const bool beta_exceeded,
                                 const bool final_cutoff,
                                 const bool positive_reduction_before_nnue,
                                 const bool phase_fm_candidate,
                                 const bool phase_fm_adjusted,
                                 const bool lca_candidate, const bool lca_adjusted,
                                 const bool cross_candidate, const bool cross_adjusted) {
        auto& stats = LocalStats();
        const auto add = [&](PhaseFmLmrBucket& value) {
            ++value.moves;
            value.reduced_fail_highs += reduced_fail_high;
            value.researches += researched;
            value.beta_exceeded += beta_exceeded;
            value.final_cutoffs += final_cutoff;
            if (researched) {
                const auto delta = static_cast<std::int64_t>(final_value)
                                 - static_cast<std::int64_t>(reduced_value);
                const auto abs_delta = static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
                ++value.research_score_samples;
                value.research_score_abs_delta_sum += abs_delta;
                value.research_score_abs_delta_max =
                  std::max(value.research_score_abs_delta_max, abs_delta);
            }
        };
        ++stats.phase_fm_lmr_moves;
        ++stats.lca_lmr_moves;
        add(stats.lca_lmr_all_outcomes);
        if (!access_.signal.valid || !has_static_eval_)
            return;

        ++stats.phase_fm_lmr_signal_valid_moves;
        ++stats.lca_lmr_signal_valid_moves;
        const auto fm_bin = PhaseSignalBin(access_.signal.fm_reliance);
        const auto depth_bin = ConditionalDepthBin(depth);
        const auto move_bin = MoveCountBin(move_count);
        const auto bucket = static_cast<std::size_t>(std::clamp(
          access_.signal.selected_bucket, 0, static_cast<int>(kBucketCount - 1)));
        const auto static_bin = MagnitudeBin(static_eval_);
        const auto router_group = RouterMarginGroup(access_.signal.router_margin);
        const auto fm_group = FmRelianceGroup(access_.signal.fm_reliance);
        const auto fm_fine_group = FmRelianceFineGroup(access_.signal.fm_reliance);

        const bool router_danger = !pv_ && depth >= 3 && depth <= 8
          && move_count >= 2 && move_count <= 8
          && access_.signal.router_margin <= NNUE_ROUTER_LMR_MARGIN_THRESHOLD;
        const bool lca_danger =
          access_.signal.lca_abs_delta_sum >= NNUE_COMBINED_LCA_SUM_THRESHOLD;
        const auto lca_sum = static_cast<std::size_t>(std::clamp(
          access_.signal.lca_abs_delta_sum, 0,
          static_cast<std::int32_t>(kLcaDeltaSumMax)));

        // Model-calibration diagnostics. These counters observe the completed
        // LMR move only; they never feed a threshold or decision back to search.
        auto& calibration = stats.signal_calibration;
        ++calibration.signal_valid_lmr_moves;
        const bool router_structural = !pv_ && depth >= 3 && depth <= 8
          && move_count >= 2 && move_count <= 8;
        if (router_structural)
            ++calibration.router_structural_moves;
        calibration.router_actual_adjusted_moves += router_adjusted;
        add(calibration.router_distribution[
          CalibrationRouterBin(access_.signal.router_margin)]);
        if (router_structural)
            for (std::size_t threshold = 0;
                 threshold < kCalibrationRouterThresholdCount; ++threshold)
                if (access_.signal.router_margin
                    <= kCalibrationRouterThresholds[threshold])
                    add(calibration.router_cohorts[threshold]);

        const bool cross_structural = depth >= 3 && depth <= 8
          && move_count >= 2 && move_count <= 8;
        // Keep this population identical to the historical model-relative
        // LCA percentile analysis: every signal-valid LMR move which was not
        // actually deepened by Router-LMR.  Positive-reduction eligibility is
        // deliberately applied later to deployable Cross candidates, not to
        // the model calibration itself.
        const bool lca_calibration_eligible = !router_adjusted;
        if (lca_calibration_eligible) {
            ++calibration.lca_calibration_eligible_moves;
            add(calibration.lca_eligible_by_delta_sum[lca_sum]);
        }
        const bool cross_after_router_eligible = cross_structural
          && positive_reduction_before_nnue && !router_adjusted;
        if (cross_after_router_eligible)
            ++calibration.cross_structural_eligible_moves;
        if (cross_after_router_eligible)
            add(calibration.cross_after_router_all_by_lca_sum[lca_sum]);
        for (std::size_t threshold = 0;
             threshold < kCalibrationCrossThresholdCount; ++threshold) {
            if (access_.signal.cross_abs_max
                < kCalibrationCrossThresholds[threshold])
                continue;
            add(calibration.cross_all_cohorts[threshold]);
            if (cross_after_router_eligible)
                add(calibration.cross_after_router_by_lca_sum[threshold][lca_sum]);
        }
        const bool cross_danger_calibration = cross_structural
          && access_.signal.cross_abs_max >= 127;
        const std::size_t overlap_flags = (router_danger ? 1U : 0U)
                                        | (cross_danger_calibration ? 2U : 0U);
        add(calibration.overlap_by_lca_sum[lca_sum * 4 + overlap_flags]);

        const std::size_t combined_group = router_danger
          ? (lca_danger ? 2U : 0U)
          : (lca_danger ? 1U : 3U);
        add(stats.combined_signal[combined_group]);
        add(stats.combined_signal_by_depth[combined_group][depth_bin]);
        add(stats.combined_signal_by_move_count[combined_group][move_bin]);
        add(stats.combined_signal_by_bucket[combined_group][bucket]);
        add(stats.combined_signal_by_static_eval[combined_group][static_bin]);

        const auto diff_rms_bin = DiffRmsEnergyBin(access_.signal.diff_rms_energy_sum);
        const std::size_t lca_tail = lca_danger ? 1U : 0U;
        add(stats.diff_rms_lmr[diff_rms_bin]);
        add(stats.diff_rms_lmr_by_depth[depth_bin][diff_rms_bin]);
        add(stats.diff_rms_lmr_by_move_count[move_bin][diff_rms_bin]);
        add(stats.diff_rms_lmr_by_bucket[bucket][diff_rms_bin]);
        add(stats.diff_rms_lmr_by_static_eval[static_bin][diff_rms_bin]);
        add(stats.diff_rms_lmr_by_router[router_group][diff_rms_bin]);
        add(stats.diff_rms_lmr_by_lca_tail[lca_tail][diff_rms_bin]);
        const auto diff_rms_joint =
          (FmJointBaseIndex(depth_bin, move_bin, bucket, static_bin, router_group) * 2
             + lca_tail) * kDiffRmsEnergyBins + diff_rms_bin;
        add(stats.diff_rms_lmr_joint[diff_rms_joint]);

        const auto add_cross_activity = [&](auto& activity, const std::size_t bin) {
            add(activity.all[bin]);
            add(activity.by_depth[depth_bin][bin]);
            add(activity.by_move_count[move_bin][bin]);
            add(activity.by_bucket[bucket][bin]);
            add(activity.by_static_eval[static_bin][bin]);
            add(activity.by_router[router_group][bin]);
            add(activity.by_lca_tail[lca_tail][bin]);
            const auto joint =
              (FmJointBaseIndex(depth_bin, move_bin, bucket, static_bin, router_group) * 2
                 + lca_tail) * kCrossActivityBins + bin;
            add(activity.joint[joint]);
        };
        add_cross_activity(stats.cross_mean_abs_lmr,
                           CrossMeanAbsBin(access_.signal.cross_abs_sum));
        add_cross_activity(stats.cross_max_abs_lmr,
                           CrossMaxAbsBin(access_.signal.cross_abs_max));
        const bool cross_danger = access_.signal.cross_abs_max >= 126;
        const std::size_t raw_overlap = (cross_danger ? 4U : 0U)
                                      | (router_adjusted ? 2U : 0U)
                                      | (lca_danger ? 1U : 0U);
        const std::size_t eligible_overlap = (cross_danger ? 4U : 0U)
                                           | (router_adjusted ? 2U : 0U)
                                           | (lca_candidate ? 1U : 0U);
        add(stats.cross_overlap_raw[raw_overlap]);
        add(stats.cross_overlap_eligible[eligible_overlap]);
        if (!router_adjusted && !lca_candidate) {
            for (std::size_t threshold = 0;
                 threshold < kCrossDeployableThresholdCount; ++threshold) {
                if (access_.signal.cross_abs_max < kCrossDeployableThresholds[threshold])
                    continue;
                auto& cohort = stats.cross_deployable_thresholds[threshold];
                add(cohort.total);
                add(cohort.by_depth[depth_bin]);
                add(cohort.by_move_count[move_bin]);
                add(cohort.by_bucket[bucket]);
                add(cohort.by_static_eval[static_bin]);
            }
        }

        // Main-gate summaries were accumulated while the 32 Q64 values were
        // produced. Classification here adds no NNUE-side reduction. The full
        // joint table controls for the requested Cross saturation axis too.
        const std::size_t cross_saturation =
          access_.signal.cross_abs_max >= 127 ? 1U : 0U;
        const auto main_gate_joint_base =
          (FmJointBaseIndex(depth_bin, move_bin, bucket, static_bin, router_group) * 2
             + lca_tail) * 2 + cross_saturation;
        const bool uncovered_by_deployed_nnue_lmr =
          !router_adjusted && !lca_adjusted && !cross_adjusted;
        for (std::size_t kind = 0; kind < kMainGateSignalCount; ++kind) {
            auto& activity = stats.main_gate_activity[kind];
            const auto bin = MainGateActivityBin(access_.signal, kind);
            add(activity.all[bin]);
            add(activity.by_depth[depth_bin][bin]);
            add(activity.by_move_count[move_bin][bin]);
            add(activity.by_bucket[bucket][bin]);
            add(activity.by_static_eval[static_bin][bin]);
            add(activity.by_router[router_group][bin]);
            add(activity.by_lca_tail[lca_tail][bin]);
            add(activity.by_cross_saturation[cross_saturation][bin]);
            if (uncovered_by_deployed_nnue_lmr)
                add(activity.uncovered[bin]);
            add(activity.joint[main_gate_joint_base * kMainGateActivityBins + bin]);
        }

        // FM summaries were accumulated in the existing Diff/Abs output loop.
        // This records only completed-move outcomes; it does not rescan the 32
        // activations and does not affect NNUE or LMR decisions.
        for (std::size_t kind = 0; kind < kFmActivitySignalCount; ++kind) {
            auto& activity = stats.fm_activity[kind];
            const auto bin = FmActivityBin(access_.signal, kind);
            add(activity.all[bin]);
            add(activity.by_depth[depth_bin][bin]);
            add(activity.by_move_count[move_bin][bin]);
            add(activity.by_bucket[bucket][bin]);
            add(activity.by_static_eval[static_bin][bin]);
            add(activity.by_router[router_group][bin]);
            add(activity.by_lca_tail[lca_tail][bin]);
            add(activity.by_cross_saturation[cross_saturation][bin]);
            if (uncovered_by_deployed_nnue_lmr)
                add(activity.uncovered[bin]);
            add(activity.joint[main_gate_joint_base * kFmActivityBins + bin]);
        }

        add(stats.phase_fm_lmr[fm_bin]);
        add(stats.phase_fm_lmr_by_depth[depth_bin][fm_bin]);
        add(stats.phase_fm_lmr_by_move_count[move_bin][fm_bin]);
        add(stats.phase_fm_lmr_by_bucket[bucket][fm_bin]);
        add(stats.phase_fm_lmr_by_static_eval[static_bin][fm_bin]);
        add(stats.phase_fm_lmr_by_router_and_fm[router_group][fm_group]);
        const auto joint = FmJointBaseIndex(depth_bin, move_bin, bucket, static_bin,
                                            router_group) * kFmRelianceGroups + fm_group;
        add(stats.phase_fm_lmr_joint[joint]);
        const auto adjusted = router_adjusted ? 1U : 0U;
        add(stats.phase_fm_lmr_by_router_adjustment[adjusted][fm_fine_group]);
        add(stats.phase_fm_lmr_by_router_adjustment_and_static
              [adjusted][static_bin][fm_fine_group]);
        if (!router_adjusted && (static_eval_ >= 1200 || static_eval_ <= -1200)) {
            const auto sign = static_eval_ >= 1200 ? 0U : 1U;
            add(stats.phase_fm_lmr_extreme_by_sign[sign][fm_fine_group]);
            add(stats.phase_fm_lmr_extreme_by_sign_and_bucket
                  [sign][fm_fine_group][bucket]);
        }
        auto& experiment = stats.phase_fm_lmr_experiment;
        if (phase_fm_candidate) {
            ++experiment.candidate_moves;
            add(experiment.candidate_outcomes);
        }
        if (phase_fm_adjusted) {
            ++experiment.adjusted_moves;
            add(experiment.adjusted_outcomes);
            ++experiment.adjusted_by_bucket[bucket];
            ++experiment.adjusted_by_depth[depth_bin];
            ++experiment.adjusted_by_move_count[move_bin];
        }

        // The LCA signal is obtained in the existing final blend loop: no Q/K/V,
        // sigmoid, or correction is recomputed for diagnostics.  Reuse this
        // completed-move outcome and the same conditioning axes as the FM study.
        const auto lca_mean_bin = LcaMeanDeltaBin(access_.signal.lca_mean_abs_delta);
        const auto lca_max_bin = LcaMaxDeltaBin(access_.signal.lca_max_abs_delta);
        const auto lca_group = LcaMagnitudeGroup(access_.signal.lca_mean_abs_delta);
        add(stats.lca_lmr_by_mean_delta[lca_mean_bin]);
        add(stats.lca_lmr_by_max_delta[lca_max_bin]);
        add(stats.lca_lmr_by_depth[depth_bin][lca_mean_bin]);
        add(stats.lca_lmr_by_move_count[move_bin][lca_mean_bin]);
        add(stats.lca_lmr_by_bucket[bucket][lca_mean_bin]);
        add(stats.lca_lmr_by_static_eval[static_bin][lca_mean_bin]);
        add(stats.lca_lmr_by_router_and_magnitude[router_group][lca_group]);
        const auto lca_joint = FmJointBaseIndex(depth_bin, move_bin, bucket, static_bin,
                                                router_group) * kLcaMagnitudeGroups
                             + lca_group;
        add(stats.lca_lmr_joint[lca_joint]);
        if (!router_adjusted) {
            add(stats.lca_non_router_all);
            add(stats.lca_non_router_by_delta_sum[lca_sum]);
            for (std::size_t threshold = 0; threshold < kLcaThresholdCount; ++threshold) {
                if (access_.signal.lca_mean_abs_delta < kLcaThresholds[threshold])
                    continue;
                auto& cohort = stats.lca_non_router_thresholds[threshold];
                add(cohort.total);
                add(cohort.by_depth[depth_bin]);
                add(cohort.by_move_count[move_bin]);
                add(cohort.by_bucket[bucket]);
                add(cohort.by_static_eval[static_bin]);
                add(cohort.by_router_margin[router_group]);
            }
        }
        auto& lca_experiment = stats.lca_lmr_experiment;
        if (lca_candidate) {
            add(lca_experiment.candidate_outcomes);
            add(lca_experiment.candidate_by_bucket[bucket]);
            add(lca_experiment.candidate_by_depth[depth_bin]);
            add(lca_experiment.candidate_by_move_count[move_bin]);
            add(lca_experiment.candidate_by_static_eval[static_bin]);
        }
        if (lca_adjusted) {
            add(lca_experiment.adjusted_outcomes);
            add(lca_experiment.adjusted_by_bucket[bucket]);
            add(lca_experiment.adjusted_by_depth[depth_bin]);
            add(lca_experiment.adjusted_by_move_count[move_bin]);
            add(lca_experiment.adjusted_by_static_eval[static_bin]);
        }
        auto& cross_experiment = stats.cross_lmr_experiment;
        if (cross_candidate) {
            add(cross_experiment.candidate_outcomes);
            add(cross_experiment.candidate_by_bucket[bucket]);
            add(cross_experiment.candidate_by_depth[depth_bin]);
            add(cross_experiment.candidate_by_move_count[move_bin]);
            add(cross_experiment.candidate_by_static_eval[static_bin]);
        }
        if (cross_adjusted) {
            add(cross_experiment.adjusted_outcomes);
            add(cross_experiment.adjusted_by_bucket[bucket]);
            add(cross_experiment.adjusted_by_depth[depth_bin]);
            add(cross_experiment.adjusted_by_move_count[move_bin]);
            add(cross_experiment.adjusted_by_static_eval[static_bin]);
        }
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
#if defined(ENABLE_NNUE_RFP_SHADOW)
    bool reverse_futility_shadow_sampled_ = false;
    std::size_t reverse_futility_shadow_disagreement_bin_ = 0;
    std::size_t reverse_futility_shadow_static_eval_bin_ = 0;
#endif
    std::array<std::uint32_t, kConditionBins> lmr_events_{};
    std::array<std::uint32_t, kConditionBins> lmr_researches_{};
#if defined(ENABLE_NNUE_ROUTER_LMR_EXPERIMENT)
    bool router_lmr_node_adjusted_ = false;
#endif
    Eval::NNUE::NnueSignalEvalAccess access_{};
};

inline void Reset() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (auto* stats : g_thread_stats) {
        // ThreadStats contains large diagnostic matrices. Reconstruct it in
        // place so reset does not materialize a full-size temporary on the
        // engine thread's fixed stack.
        stats->~ThreadStats();
        ::new (static_cast<void*>(stats)) ThreadStats();
    }
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

inline void AddPhaseFmLmr(PhaseFmLmrBucket& dst, const PhaseFmLmrBucket& src) {
    dst.moves += src.moves;
    dst.reduced_fail_highs += src.reduced_fail_highs;
    dst.researches += src.researches;
    dst.beta_exceeded += src.beta_exceeded;
    dst.final_cutoffs += src.final_cutoffs;
    dst.research_score_samples += src.research_score_samples;
    dst.research_score_abs_delta_sum += src.research_score_abs_delta_sum;
    dst.research_score_abs_delta_max =
      std::max(dst.research_score_abs_delta_max, src.research_score_abs_delta_max);
}

inline void AddPhaseFmLmrExperiment(PhaseFmLmrExperimentStats& dst,
                                    const PhaseFmLmrExperimentStats& src) {
    dst.candidate_moves += src.candidate_moves;
    dst.adjusted_moves += src.adjusted_moves;
    AddPhaseFmLmr(dst.candidate_outcomes, src.candidate_outcomes);
    AddPhaseFmLmr(dst.adjusted_outcomes, src.adjusted_outcomes);
    for (std::size_t i = 0; i < kBucketCount; ++i)
        dst.adjusted_by_bucket[i] += src.adjusted_by_bucket[i];
    for (std::size_t i = 0; i < kConditionBins; ++i) {
        dst.adjusted_by_depth[i] += src.adjusted_by_depth[i];
        dst.adjusted_by_move_count[i] += src.adjusted_by_move_count[i];
    }
}

inline void AddLcaLmrExperiment(LcaLmrExperimentStats& dst,
                                const LcaLmrExperimentStats& src) {
    AddPhaseFmLmr(dst.candidate_outcomes, src.candidate_outcomes);
    AddPhaseFmLmr(dst.adjusted_outcomes, src.adjusted_outcomes);
    for (std::size_t i = 0; i < kBucketCount; ++i) {
        AddPhaseFmLmr(dst.candidate_by_bucket[i], src.candidate_by_bucket[i]);
        AddPhaseFmLmr(dst.adjusted_by_bucket[i], src.adjusted_by_bucket[i]);
    }
    for (std::size_t i = 0; i < kConditionBins; ++i) {
        AddPhaseFmLmr(dst.candidate_by_depth[i], src.candidate_by_depth[i]);
        AddPhaseFmLmr(dst.candidate_by_move_count[i], src.candidate_by_move_count[i]);
        AddPhaseFmLmr(dst.candidate_by_static_eval[i], src.candidate_by_static_eval[i]);
        AddPhaseFmLmr(dst.adjusted_by_depth[i], src.adjusted_by_depth[i]);
        AddPhaseFmLmr(dst.adjusted_by_move_count[i], src.adjusted_by_move_count[i]);
        AddPhaseFmLmr(dst.adjusted_by_static_eval[i], src.adjusted_by_static_eval[i]);
    }
}

#if defined(ENABLE_NNUE_RFP_SHADOW)
inline void AddReverseFutilityShadow(ReverseFutilityShadowBucket& dst,
                                     const ReverseFutilityShadowBucket& src) {
    dst.rfp_conditions += src.rfp_conditions;
    dst.shadow_samples += src.shadow_samples;
    dst.completed_samples += src.completed_samples;
    dst.correct_cuts += src.correct_cuts;
    dst.wrong_cuts += src.wrong_cuts;
    dst.result_minus_beta_sum += src.result_minus_beta_sum;
    dst.result_minus_static_sum += src.result_minus_static_sum;
    if (src.completed_samples) {
        dst.result_minus_beta_min =
          std::min(dst.result_minus_beta_min, src.result_minus_beta_min);
        dst.result_minus_beta_max =
          std::max(dst.result_minus_beta_max, src.result_minus_beta_max);
        dst.result_minus_static_min =
          std::min(dst.result_minus_static_min, src.result_minus_static_min);
        dst.result_minus_static_max =
          std::max(dst.result_minus_static_max, src.result_minus_static_max);
    }
}
#endif

inline void AddCorrelation(CorrelationSums& dst, const CorrelationSums& src) {
    dst.count += src.count;
    dst.sx += src.sx;
    dst.sy += src.sy;
    dst.sxx += src.sxx;
    dst.syy += src.syy;
    dst.sxy += src.sxy;
}

inline void Snapshot(ThreadStats& result) {
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
        result.phase_fm_lmr_moves += stats->phase_fm_lmr_moves;
        result.phase_fm_lmr_signal_valid_moves += stats->phase_fm_lmr_signal_valid_moves;
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            AddPhaseFmLmr(result.phase_fm_lmr[bin], stats->phase_fm_lmr[bin]);
            for (std::size_t group = 0; group < kConditionBins; ++group) {
                AddPhaseFmLmr(result.phase_fm_lmr_by_depth[group][bin],
                              stats->phase_fm_lmr_by_depth[group][bin]);
                AddPhaseFmLmr(result.phase_fm_lmr_by_move_count[group][bin],
                              stats->phase_fm_lmr_by_move_count[group][bin]);
                AddPhaseFmLmr(result.phase_fm_lmr_by_static_eval[group][bin],
                              stats->phase_fm_lmr_by_static_eval[group][bin]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(result.phase_fm_lmr_by_bucket[bucket][bin],
                              stats->phase_fm_lmr_by_bucket[bucket][bin]);
        }
        for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
            for (std::size_t fm = 0; fm < kFmRelianceGroups; ++fm)
                AddPhaseFmLmr(result.phase_fm_lmr_by_router_and_fm[router][fm],
                              stats->phase_fm_lmr_by_router_and_fm[router][fm]);
        for (std::size_t i = 0; i < result.phase_fm_lmr_joint.size(); ++i)
            AddPhaseFmLmr(result.phase_fm_lmr_joint[i], stats->phase_fm_lmr_joint[i]);
        for (std::size_t adjusted = 0; adjusted < 2; ++adjusted)
            for (std::size_t fm = 0; fm < kFmRelianceFineGroups; ++fm) {
                AddPhaseFmLmr(result.phase_fm_lmr_by_router_adjustment[adjusted][fm],
                              stats->phase_fm_lmr_by_router_adjustment[adjusted][fm]);
                for (std::size_t static_bin = 0; static_bin < kConditionBins; ++static_bin)
                    AddPhaseFmLmr(
                      result.phase_fm_lmr_by_router_adjustment_and_static
                        [adjusted][static_bin][fm],
                      stats->phase_fm_lmr_by_router_adjustment_and_static
                        [adjusted][static_bin][fm]);
            }
        for (std::size_t sign = 0; sign < 2; ++sign)
            for (std::size_t fm = 0; fm < kFmRelianceFineGroups; ++fm) {
                AddPhaseFmLmr(result.phase_fm_lmr_extreme_by_sign[sign][fm],
                              stats->phase_fm_lmr_extreme_by_sign[sign][fm]);
                for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                    AddPhaseFmLmr(
                      result.phase_fm_lmr_extreme_by_sign_and_bucket[sign][fm][bucket],
                      stats->phase_fm_lmr_extreme_by_sign_and_bucket[sign][fm][bucket]);
            }
        AddPhaseFmLmrExperiment(result.phase_fm_lmr_experiment,
                                stats->phase_fm_lmr_experiment);
        AddLcaLmrExperiment(result.lca_lmr_experiment,
                            stats->lca_lmr_experiment);
        AddLcaLmrExperiment(result.cross_lmr_experiment,
                            stats->cross_lmr_experiment);
        result.lca_lmr_moves += stats->lca_lmr_moves;
        result.lca_lmr_signal_valid_moves += stats->lca_lmr_signal_valid_moves;
        AddPhaseFmLmr(result.lca_lmr_all_outcomes, stats->lca_lmr_all_outcomes);
        for (std::size_t bin = 0; bin < kLcaMeanDeltaBins; ++bin) {
            AddPhaseFmLmr(result.lca_lmr_by_mean_delta[bin],
                          stats->lca_lmr_by_mean_delta[bin]);
            for (std::size_t group = 0; group < kConditionBins; ++group) {
                AddPhaseFmLmr(result.lca_lmr_by_depth[group][bin],
                              stats->lca_lmr_by_depth[group][bin]);
                AddPhaseFmLmr(result.lca_lmr_by_move_count[group][bin],
                              stats->lca_lmr_by_move_count[group][bin]);
                AddPhaseFmLmr(result.lca_lmr_by_static_eval[group][bin],
                              stats->lca_lmr_by_static_eval[group][bin]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(result.lca_lmr_by_bucket[bucket][bin],
                              stats->lca_lmr_by_bucket[bucket][bin]);
        }
        for (std::size_t bin = 0; bin < kLcaMaxDeltaBins; ++bin)
            AddPhaseFmLmr(result.lca_lmr_by_max_delta[bin],
                          stats->lca_lmr_by_max_delta[bin]);
        for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
            for (std::size_t lca = 0; lca < kLcaMagnitudeGroups; ++lca)
                AddPhaseFmLmr(result.lca_lmr_by_router_and_magnitude[router][lca],
                              stats->lca_lmr_by_router_and_magnitude[router][lca]);
        for (std::size_t i = 0; i < result.lca_lmr_joint.size(); ++i)
            AddPhaseFmLmr(result.lca_lmr_joint[i], stats->lca_lmr_joint[i]);
        for (std::size_t threshold = 0; threshold < kLcaThresholdCount; ++threshold) {
            auto& dst = result.lca_non_router_thresholds[threshold];
            const auto& src = stats->lca_non_router_thresholds[threshold];
            AddPhaseFmLmr(dst.total, src.total);
            for (std::size_t group = 0; group < kConditionBins; ++group) {
                AddPhaseFmLmr(dst.by_depth[group], src.by_depth[group]);
                AddPhaseFmLmr(dst.by_move_count[group], src.by_move_count[group]);
                AddPhaseFmLmr(dst.by_static_eval[group], src.by_static_eval[group]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(dst.by_bucket[bucket], src.by_bucket[bucket]);
            for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
                AddPhaseFmLmr(dst.by_router_margin[router], src.by_router_margin[router]);
        }
        AddPhaseFmLmr(result.lca_non_router_all, stats->lca_non_router_all);
        for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum)
            AddPhaseFmLmr(result.lca_non_router_by_delta_sum[sum],
                          stats->lca_non_router_by_delta_sum[sum]);
        auto& calibration = result.signal_calibration;
        const auto& source_calibration = stats->signal_calibration;
        calibration.signal_valid_lmr_moves += source_calibration.signal_valid_lmr_moves;
        calibration.router_structural_moves += source_calibration.router_structural_moves;
        calibration.router_actual_adjusted_moves +=
          source_calibration.router_actual_adjusted_moves;
        calibration.lca_calibration_eligible_moves +=
          source_calibration.lca_calibration_eligible_moves;
        calibration.cross_structural_eligible_moves +=
          source_calibration.cross_structural_eligible_moves;
        for (std::size_t bin = 0; bin < kCalibrationRouterBins; ++bin)
            AddPhaseFmLmr(calibration.router_distribution[bin],
                          source_calibration.router_distribution[bin]);
        for (std::size_t threshold = 0;
             threshold < kCalibrationRouterThresholdCount; ++threshold)
            AddPhaseFmLmr(calibration.router_cohorts[threshold],
                          source_calibration.router_cohorts[threshold]);
        for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum)
            AddPhaseFmLmr(calibration.lca_eligible_by_delta_sum[sum],
                          source_calibration.lca_eligible_by_delta_sum[sum]);
        for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum)
            AddPhaseFmLmr(calibration.cross_after_router_all_by_lca_sum[sum],
                          source_calibration.cross_after_router_all_by_lca_sum[sum]);
        for (std::size_t threshold = 0;
             threshold < kCalibrationCrossThresholdCount; ++threshold) {
            AddPhaseFmLmr(calibration.cross_all_cohorts[threshold],
                          source_calibration.cross_all_cohorts[threshold]);
            for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum)
                AddPhaseFmLmr(
                  calibration.cross_after_router_by_lca_sum[threshold][sum],
                  source_calibration.cross_after_router_by_lca_sum[threshold][sum]);
        }
        for (std::size_t index = 0;
             index < calibration.overlap_by_lca_sum.size(); ++index)
            AddPhaseFmLmr(calibration.overlap_by_lca_sum[index],
                          source_calibration.overlap_by_lca_sum[index]);
        for (std::size_t group = 0; group < kCombinedSignalGroups; ++group) {
            AddPhaseFmLmr(result.combined_signal[group], stats->combined_signal[group]);
            for (std::size_t bin = 0; bin < kConditionBins; ++bin) {
                AddPhaseFmLmr(result.combined_signal_by_depth[group][bin],
                              stats->combined_signal_by_depth[group][bin]);
                AddPhaseFmLmr(result.combined_signal_by_move_count[group][bin],
                              stats->combined_signal_by_move_count[group][bin]);
                AddPhaseFmLmr(result.combined_signal_by_static_eval[group][bin],
                              stats->combined_signal_by_static_eval[group][bin]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(result.combined_signal_by_bucket[group][bucket],
                              stats->combined_signal_by_bucket[group][bucket]);
        }
        for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin) {
            AddPhaseFmLmr(result.diff_rms_lmr[bin], stats->diff_rms_lmr[bin]);
            for (std::size_t group = 0; group < kConditionBins; ++group) {
                AddPhaseFmLmr(result.diff_rms_lmr_by_depth[group][bin],
                              stats->diff_rms_lmr_by_depth[group][bin]);
                AddPhaseFmLmr(result.diff_rms_lmr_by_move_count[group][bin],
                              stats->diff_rms_lmr_by_move_count[group][bin]);
                AddPhaseFmLmr(result.diff_rms_lmr_by_static_eval[group][bin],
                              stats->diff_rms_lmr_by_static_eval[group][bin]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(result.diff_rms_lmr_by_bucket[bucket][bin],
                              stats->diff_rms_lmr_by_bucket[bucket][bin]);
            for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
                AddPhaseFmLmr(result.diff_rms_lmr_by_router[router][bin],
                              stats->diff_rms_lmr_by_router[router][bin]);
            for (std::size_t tail = 0; tail < 2; ++tail)
                AddPhaseFmLmr(result.diff_rms_lmr_by_lca_tail[tail][bin],
                              stats->diff_rms_lmr_by_lca_tail[tail][bin]);
        }
        for (std::size_t index = 0; index < result.diff_rms_lmr_joint.size(); ++index)
            AddPhaseFmLmr(result.diff_rms_lmr_joint[index],
                          stats->diff_rms_lmr_joint[index]);
        const auto add_cross_activity = [&](auto& dst, const auto& src) {
            for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin) {
                AddPhaseFmLmr(dst.all[bin], src.all[bin]);
                for (std::size_t group = 0; group < kConditionBins; ++group) {
                    AddPhaseFmLmr(dst.by_depth[group][bin], src.by_depth[group][bin]);
                    AddPhaseFmLmr(dst.by_move_count[group][bin], src.by_move_count[group][bin]);
                    AddPhaseFmLmr(dst.by_static_eval[group][bin], src.by_static_eval[group][bin]);
                }
                for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                    AddPhaseFmLmr(dst.by_bucket[bucket][bin], src.by_bucket[bucket][bin]);
                for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
                    AddPhaseFmLmr(dst.by_router[router][bin], src.by_router[router][bin]);
                for (std::size_t tail = 0; tail < 2; ++tail)
                    AddPhaseFmLmr(dst.by_lca_tail[tail][bin], src.by_lca_tail[tail][bin]);
            }
            for (std::size_t index = 0; index < dst.joint.size(); ++index)
                AddPhaseFmLmr(dst.joint[index], src.joint[index]);
        };
        add_cross_activity(result.cross_mean_abs_lmr, stats->cross_mean_abs_lmr);
        add_cross_activity(result.cross_max_abs_lmr, stats->cross_max_abs_lmr);
        for (std::size_t group = 0; group < kCrossOverlapGroups; ++group) {
            AddPhaseFmLmr(result.cross_overlap_raw[group],
                          stats->cross_overlap_raw[group]);
            AddPhaseFmLmr(result.cross_overlap_eligible[group],
                          stats->cross_overlap_eligible[group]);
        }
        for (std::size_t threshold = 0;
             threshold < kCrossDeployableThresholdCount; ++threshold) {
            auto& dst = result.cross_deployable_thresholds[threshold];
            const auto& src = stats->cross_deployable_thresholds[threshold];
            AddPhaseFmLmr(dst.total, src.total);
            for (std::size_t group = 0; group < kConditionBins; ++group) {
                AddPhaseFmLmr(dst.by_depth[group], src.by_depth[group]);
                AddPhaseFmLmr(dst.by_move_count[group], src.by_move_count[group]);
                AddPhaseFmLmr(dst.by_static_eval[group], src.by_static_eval[group]);
            }
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                AddPhaseFmLmr(dst.by_bucket[bucket], src.by_bucket[bucket]);
        }
        for (std::size_t kind = 0; kind < kMainGateSignalCount; ++kind) {
            auto& dst = result.main_gate_activity[kind];
            const auto& src = stats->main_gate_activity[kind];
            for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin) {
                AddPhaseFmLmr(dst.all[bin], src.all[bin]);
                AddPhaseFmLmr(dst.uncovered[bin], src.uncovered[bin]);
                for (std::size_t group = 0; group < kConditionBins; ++group) {
                    AddPhaseFmLmr(dst.by_depth[group][bin], src.by_depth[group][bin]);
                    AddPhaseFmLmr(dst.by_move_count[group][bin], src.by_move_count[group][bin]);
                    AddPhaseFmLmr(dst.by_static_eval[group][bin], src.by_static_eval[group][bin]);
                }
                for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                    AddPhaseFmLmr(dst.by_bucket[bucket][bin], src.by_bucket[bucket][bin]);
                for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
                    AddPhaseFmLmr(dst.by_router[router][bin], src.by_router[router][bin]);
                for (std::size_t tail = 0; tail < 2; ++tail) {
                    AddPhaseFmLmr(dst.by_lca_tail[tail][bin], src.by_lca_tail[tail][bin]);
                    AddPhaseFmLmr(dst.by_cross_saturation[tail][bin],
                                  src.by_cross_saturation[tail][bin]);
                }
            }
            for (std::size_t index = 0; index < dst.joint.size(); ++index)
                AddPhaseFmLmr(dst.joint[index], src.joint[index]);
        }
        for (std::size_t kind = 0; kind < kFmActivitySignalCount; ++kind) {
            auto& dst = result.fm_activity[kind];
            const auto& src = stats->fm_activity[kind];
            for (std::size_t bin = 0; bin < kFmActivityBins; ++bin) {
                AddPhaseFmLmr(dst.all[bin], src.all[bin]);
                AddPhaseFmLmr(dst.uncovered[bin], src.uncovered[bin]);
                for (std::size_t group = 0; group < kConditionBins; ++group) {
                    AddPhaseFmLmr(dst.by_depth[group][bin], src.by_depth[group][bin]);
                    AddPhaseFmLmr(dst.by_move_count[group][bin], src.by_move_count[group][bin]);
                    AddPhaseFmLmr(dst.by_static_eval[group][bin], src.by_static_eval[group][bin]);
                }
                for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                    AddPhaseFmLmr(dst.by_bucket[bucket][bin], src.by_bucket[bucket][bin]);
                for (std::size_t router = 0; router < kRouterMarginGroups; ++router)
                    AddPhaseFmLmr(dst.by_router[router][bin], src.by_router[router][bin]);
                for (std::size_t tail = 0; tail < 2; ++tail) {
                    AddPhaseFmLmr(dst.by_lca_tail[tail][bin], src.by_lca_tail[tail][bin]);
                    AddPhaseFmLmr(dst.by_cross_saturation[tail][bin],
                                  src.by_cross_saturation[tail][bin]);
                }
            }
            for (std::size_t index = 0; index < dst.joint.size(); ++index)
                AddPhaseFmLmr(dst.joint[index], src.joint[index]);
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
#if defined(ENABLE_NNUE_RFP_SHADOW)
        result.rfp_conditions += stats->rfp_conditions;
        result.rfp_signal_valid_conditions += stats->rfp_signal_valid_conditions;
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            AddReverseFutilityShadow(result.rfp_shadow_by_disagreement[bin],
                                     stats->rfp_shadow_by_disagreement[bin]);
            for (std::size_t static_bin = 0; static_bin < kConditionBins; ++static_bin)
                AddReverseFutilityShadow(
                  result.rfp_shadow_by_static_eval[static_bin][bin],
                  stats->rfp_shadow_by_static_eval[static_bin][bin]);
        }
#endif
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

inline void PrintPhaseFmLmrValue(std::ostream& out, const PhaseFmLmrBucket& value) {
    out << " moves=" << value.moves
        << " reduced-FH=" << Percent(value.reduced_fail_highs, value.moves) << '%'
        << " re-search=" << Percent(value.researches, value.moves) << '%'
        << " beta=" << Percent(value.beta_exceeded, value.moves) << '%'
        << " cutoff=" << Percent(value.final_cutoffs, value.moves) << '%'
        << " research-mean-abs-delta="
        << (value.research_score_samples
              ? double(value.research_score_abs_delta_sum) / value.research_score_samples : 0.0)
        << " research-samples=" << value.research_score_samples;
}

template<std::size_t Groups>
inline void PrintConditionalPhaseFmLmr(
  std::ostream& out, const char* title, const std::array<const char*, Groups>& labels,
  const std::array<std::array<PhaseFmLmrBucket, kSignalBins>, Groups>& table) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.moves)
                continue;
            out << "    phase.fm_reliance=";
            PrintSignalRange(out, 3, bin);
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }
}

inline const char* LcaMeanDeltaLabel(const std::size_t bin) {
    static constexpr const char* labels[kLcaMeanDeltaBins] = {
      "0", "(0,0.25)", "[0.25,0.50)", "[0.50,1.00)", "[1.00,2.00)",
      "[2.00,4.00)", "[4.00,8.00)", "[8.00,16.00)", "[16.00,32.00)", "32+"};
    return labels[bin];
}

inline const char* LcaMaxDeltaLabel(const std::size_t bin) {
    static constexpr const char* labels[kLcaMaxDeltaBins] = {
      "0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"};
    return labels[bin];
}

template<std::size_t Groups>
inline void PrintConditionalLcaLmr(
  std::ostream& out, const char* title, const std::array<const char*, Groups>& labels,
  const std::array<std::array<PhaseFmLmrBucket, kLcaMeanDeltaBins>, Groups>& table) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kLcaMeanDeltaBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.moves)
                continue;
            out << "    lca.mean_abs_delta=" << LcaMeanDeltaLabel(bin);
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }
}

template<std::size_t Groups>
inline void PrintConditionalDiffRms(
  std::ostream& out, const char* title, const std::array<const char*, Groups>& labels,
  const std::array<std::array<PhaseFmLmrBucket, kDiffRmsEnergyBins>, Groups>& table) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.moves)
                continue;
            out << "    " << DiffRmsEnergyLabel(bin);
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }
}

template<std::size_t Groups, typename EdgeType, std::size_t EdgeCount>
inline void PrintConditionalCrossActivity(
  std::ostream& out, const char* title, const char* signal_name,
  const std::array<const char*, Groups>& labels,
  const std::array<std::array<PhaseFmLmrBucket, kCrossActivityBins>, Groups>& table,
  const std::array<EdgeType, EdgeCount>& edges) {
    out << '[' << title << "]\n";
    for (std::size_t group = 0; group < Groups; ++group) {
        out << "  {" << labels[group] << "}\n";
        for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin) {
            const auto& value = table[group][bin];
            if (!value.moves)
                continue;
            out << "    " << CrossActivityLabel(signal_name, bin, edges);
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }
}

inline void Report(std::ostream& out) {
    // Diagnostic matrices can be several MiB. Keep the aggregate off the
    // search thread's fixed-size stack while producing the report.
    const auto stats_storage = std::make_unique<ThreadStats>();
    Snapshot(*stats_storage);
    const auto& stats = *stats_storage;
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

    static constexpr const char* raw_names[19] = {
      "deep_output", "bypass_output", "signed_deep_bypass",
      "deep_bypass_disagreement", "selected_bucket",
      "router_top1_logit", "router_top2_logit", "router_margin",
      "phase.main_sqr_scale", "phase.main_raw_scale", "phase.diff_scale",
      "phase.abs_raw_scale", "phase.abs_sqr_scale", "phase.cross_scale",
      "phase.main_reliance", "phase.fm_reliance", "phase.cross_reliance",
      "lca.mean_abs_delta", "lca.max_abs_delta"};
    out << "[fresh NNUE signal ranges]\n";
    for (std::size_t i = 0; i < 19; ++i) {
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

#if defined(ENABLE_NNUE_RFP_SHADOW)
    ReverseFutilityShadowBucket rfp_shadow_total{};
    for (const auto& value : stats.rfp_shadow_by_disagreement)
        AddReverseFutilityShadow(rfp_shadow_total, value);
    const auto print_rfp_shadow_bucket = [&](const ReverseFutilityShadowBucket& value,
                                             const std::size_t bin,
                                             const char* indent) {
        if (!value.rfp_conditions)
            return;
        out << indent << "disagreement=";
        PrintSignalRange(out, 0, bin);
        out << " RFP=" << value.rfp_conditions
            << " shadow=" << value.shadow_samples
            << " completed=" << value.completed_samples
            << " correct=" << value.correct_cuts
            << " wrong=" << value.wrong_cuts
            << " correct-rate=" << Percent(value.correct_cuts, value.completed_samples) << '%'
            << " wrong-rate=" << Percent(value.wrong_cuts, value.completed_samples) << '%';
        if (value.completed_samples)
            out << " result-beta(mean/min/max)="
                << double(value.result_minus_beta_sum) / value.completed_samples << '/'
                << value.result_minus_beta_min << '/' << value.result_minus_beta_max
                << " result-staticEval(mean/min/max)="
                << double(value.result_minus_static_sum) / value.completed_samples << '/'
                << value.result_minus_static_min << '/' << value.result_minus_static_max;
        out << '\n';
    };

    out << "[reverse-futility shadow verification]\n"
        << "  sampling: 1/" << (std::uint64_t(1) << NNUE_RFP_SHADOW_SAMPLE_SHIFT)
        << " of signal-valid RFP conditions, deterministic per thread\n"
        << "  correct cut: shadow result >= original beta; wrong cut: shadow result < original beta\n"
        << "  all RFP conditions          : " << stats.rfp_conditions << '\n'
        << "  signal-valid RFP conditions : " << stats.rfp_signal_valid_conditions
        << " (" << Percent(stats.rfp_signal_valid_conditions, stats.rfp_conditions) << "%)\n"
        << "  shadow samples/completed    : " << rfp_shadow_total.shadow_samples
        << '/' << rfp_shadow_total.completed_samples << '\n'
        << "  correct/wrong cuts          : " << rfp_shadow_total.correct_cuts
        << '/' << rfp_shadow_total.wrong_cuts
        << " (wrong " << Percent(rfp_shadow_total.wrong_cuts,
                                  rfp_shadow_total.completed_samples) << "%)\n"
        << "  by disagreement\n";
    for (std::size_t bin = 0; bin < kSignalBins; ++bin)
        print_rfp_shadow_bucket(stats.rfp_shadow_by_disagreement[bin], bin, "    ");

    out << "  by |staticEval| and disagreement\n";
    for (std::size_t static_bin = 0; static_bin < kConditionBins; ++static_bin) {
        out << "    {" << magnitude_labels[static_bin] << "}\n";
        for (std::size_t bin = 0; bin < kSignalBins; ++bin)
            print_rfp_shadow_bucket(stats.rfp_shadow_by_static_eval[static_bin][bin],
                                    bin, "      ");
    }
#endif

    PrintConditionalLmr(out, "router-margin LMR events conditioned by depth", depth_labels,
                        stats.router_lmr_by_depth);
    PrintConditionalLmr(out, "router-margin LMR events conditioned by move count", move_labels,
                        stats.router_lmr_by_move_count);
    PrintConditionalLmr(out, "router-margin LMR events conditioned by selected bucket",
                        bucket_labels, stats.router_lmr_by_bucket);

    out << "[phase.fm_reliance LMR move outcomes]\n"
        << "  population: actual LMR moves completed through final beta/cutoff decision\n"
        << "  all LMR moves          : " << stats.phase_fm_lmr_moves << '\n'
        << "  signal-valid LMR moves : " << stats.phase_fm_lmr_signal_valid_moves << " ("
        << Percent(stats.phase_fm_lmr_signal_valid_moves, stats.phase_fm_lmr_moves) << "%)\n";
    for (std::size_t bin = 0; bin < kSignalBins; ++bin) {
        const auto& value = stats.phase_fm_lmr[bin];
        if (!value.moves)
            continue;
        out << "  phase.fm_reliance=";
        PrintSignalRange(out, 3, bin);
        PrintPhaseFmLmrValue(out, value);
        out << '\n';
    }
    PrintConditionalPhaseFmLmr(out, "phase.fm_reliance LMR conditioned by depth",
                               depth_labels, stats.phase_fm_lmr_by_depth);
    PrintConditionalPhaseFmLmr(out, "phase.fm_reliance LMR conditioned by move count",
                               move_labels, stats.phase_fm_lmr_by_move_count);
    PrintConditionalPhaseFmLmr(out, "phase.fm_reliance LMR conditioned by selected bucket",
                               bucket_labels, stats.phase_fm_lmr_by_bucket);
    PrintConditionalPhaseFmLmr(out, "phase.fm_reliance LMR conditioned by |staticEval|",
                               magnitude_labels, stats.phase_fm_lmr_by_static_eval);

    static constexpr std::array<const char*, kRouterMarginGroups> router_group_labels = {
      "router margin <=256", "router margin >256"};
    static constexpr std::array<const char*, kFmRelianceGroups> fm_group_labels = {
      "FM <1.75", "FM 1.75-<2.00", "FM >=2.00"};
    out << "[router margin x phase.fm_reliance LMR outcomes]\n";
    for (std::size_t router = 0; router < kRouterMarginGroups; ++router) {
        out << "  {" << router_group_labels[router] << "}\n";
        for (std::size_t fm = 0; fm < kFmRelianceGroups; ++fm) {
            out << "    " << fm_group_labels[fm];
            PrintPhaseFmLmrValue(out, stats.phase_fm_lmr_by_router_and_fm[router][fm]);
            out << '\n';
        }
    }

    // Exact joint control: compare FM groups only inside cells having identical
    // depth, move-count, bucket, |staticEval| and Router-margin groups.  Macro
    // averaging prevents large buckets from recreating the original confounding.
    std::size_t qualifying_strata = 0;
    std::array<double, kFmRelianceGroups> joint_fh{};
    std::array<double, kFmRelianceGroups> joint_research{};
    std::array<double, kFmRelianceGroups> joint_beta{};
    std::array<double, kFmRelianceGroups> joint_cutoff{};
    std::array<std::uint64_t, kFmRelianceGroups> joint_samples{};
    for (std::size_t base = 0; base < kFmJointBaseStrata; ++base) {
        bool usable = true;
        for (std::size_t fm = 0; fm < kFmRelianceGroups; ++fm)
            usable &= stats.phase_fm_lmr_joint[base * kFmRelianceGroups + fm].moves >= 20;
        if (!usable)
            continue;
        ++qualifying_strata;
        for (std::size_t fm = 0; fm < kFmRelianceGroups; ++fm) {
            const auto& value = stats.phase_fm_lmr_joint[base * kFmRelianceGroups + fm];
            joint_samples[fm] += value.moves;
            joint_fh[fm] += Percent(value.reduced_fail_highs, value.moves);
            joint_research[fm] += Percent(value.researches, value.moves);
            joint_beta[fm] += Percent(value.beta_exceeded, value.moves);
            joint_cutoff[fm] += Percent(value.final_cutoffs, value.moves);
        }
    }
    out << "[phase.fm_reliance jointly controlled LMR outcomes]\n"
        << "  fixed strata: depth x moveCount x bucket x |staticEval| x router-margin group\n"
        << "  inclusion: at least 20 moves in every FM group; rates are stratum-macro means\n"
        << "  qualifying strata: " << qualifying_strata << '\n';
    for (std::size_t fm = 0; fm < kFmRelianceGroups; ++fm) {
        const double divisor = qualifying_strata ? double(qualifying_strata) : 1.0;
        out << "  " << fm_group_labels[fm] << " samples=" << joint_samples[fm]
            << " reduced-FH=" << joint_fh[fm] / divisor << '%'
            << " re-search=" << joint_research[fm] / divisor << '%'
            << " beta=" << joint_beta[fm] / divisor << '%'
            << " cutoff=" << joint_cutoff[fm] / divisor << "%\n";
    }

    static constexpr std::array<const char*, kFmRelianceFineGroups> fm_fine_labels = {
      "FM <1.50", "FM 1.50-<1.625", "FM 1.625-<1.75", "FM 1.75-<2.00", "FM >=2.00"};
    out << "[phase.fm_reliance incremental coverage beyond production Router-LMR]\n"
        << "  Router adjusted=yes means variant 3 actually increased d by one ply.\n"
        << "  Router adjusted=no is the primary counterfactual population.\n";
    for (std::size_t adjusted = 0; adjusted < 2; ++adjusted) {
        out << "  {Router actual +1 ply: " << (adjusted ? "yes" : "no") << "}\n";
        for (std::size_t fm = 0; fm < kFmRelianceFineGroups; ++fm) {
            const auto& value = stats.phase_fm_lmr_by_router_adjustment[adjusted][fm];
            out << "    " << fm_fine_labels[fm]
                << " all-LMR=" << Percent(value.moves, stats.phase_fm_lmr_moves) << '%';
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }
    out << "[phase.fm_reliance incremental coverage by |staticEval|]\n";
    for (std::size_t static_bin = 0; static_bin < kConditionBins; ++static_bin) {
        out << "  {Router actual +1 ply: no; " << magnitude_labels[static_bin] << "}\n";
        for (std::size_t fm = 0; fm < kFmRelianceFineGroups; ++fm) {
            const auto& value = stats.phase_fm_lmr_by_router_adjustment_and_static
              [0][static_bin][fm];
            out << "    " << fm_fine_labels[fm]
                << " all-LMR=" << Percent(value.moves, stats.phase_fm_lmr_moves) << '%';
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    }

    PhaseFmLmrBucket non_router_static_ge_100_lt_1625{};
    PhaseFmLmrBucket non_router_static_ge_100_lt_1750{};
    for (std::size_t static_bin = 1; static_bin < kConditionBins; ++static_bin) {
        for (std::size_t fm = 0; fm < 2; ++fm)
            AddPhaseFmLmr(non_router_static_ge_100_lt_1625,
              stats.phase_fm_lmr_by_router_adjustment_and_static[0][static_bin][fm]);
        for (std::size_t fm = 0; fm < 3; ++fm)
            AddPhaseFmLmr(non_router_static_ge_100_lt_1750,
              stats.phase_fm_lmr_by_router_adjustment_and_static[0][static_bin][fm]);
    }
    out << "[phase.fm_reliance incremental threshold candidates]\n"
        << "  population: Router actual +1 ply=no AND |staticEval|>=100\n"
        << "  FM <1.625 all-LMR="
        << Percent(non_router_static_ge_100_lt_1625.moves, stats.phase_fm_lmr_moves) << '%';
    PrintPhaseFmLmrValue(out, non_router_static_ge_100_lt_1625);
    out << "\n  FM <1.75 all-LMR="
        << Percent(non_router_static_ge_100_lt_1750.moves, stats.phase_fm_lmr_moves) << '%';
    PrintPhaseFmLmrValue(out, non_router_static_ge_100_lt_1750);
    out << '\n';

    static constexpr std::array<const char*, 2> extreme_sign_labels = {
      "staticEval >= +1200", "staticEval <= -1200"};
    static constexpr std::array<const char*, 3> extreme_threshold_labels = {
      "FM <1.50", "FM <1.625", "FM <1.75"};
    out << "[phase.fm_reliance signed extreme-eval counterfactual cohorts]\n"
        << "  population: Router actual +1 ply=no; thresholds are cumulative\n";
    for (std::size_t sign = 0; sign < 2; ++sign) {
        out << "  {" << extreme_sign_labels[sign] << "}\n";
        for (std::size_t threshold = 0; threshold < 3; ++threshold) {
            PhaseFmLmrBucket total{};
            std::array<PhaseFmLmrBucket, kBucketCount> by_bucket{};
            for (std::size_t fm = 0; fm <= threshold; ++fm) {
                AddPhaseFmLmr(total, stats.phase_fm_lmr_extreme_by_sign[sign][fm]);
                for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                    AddPhaseFmLmr(by_bucket[bucket],
                      stats.phase_fm_lmr_extreme_by_sign_and_bucket[sign][fm][bucket]);
            }
            out << "    " << extreme_threshold_labels[threshold]
                << " all-LMR=" << Percent(total.moves, stats.phase_fm_lmr_moves) << '%';
            PrintPhaseFmLmrValue(out, total);
            out << "\n      buckets";
            for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
                out << " B" << std::setw(2) << std::setfill('0') << bucket
                    << std::setfill(' ') << ':' << by_bucket[bucket].moves
                    << '/' << Percent(by_bucket[bucket].researches,
                                      by_bucket[bucket].moves) << '%';
            out << '\n';
        }
    }

    const auto& phase_experiment = stats.phase_fm_lmr_experiment;
    out << "[Phase FM LMR explicit +1 experiment]\n"
        << "  variant                 : ";
#if defined(USE_NNUE_PHASE_FM_LMR)
    out << "1 (Router-LMR + Phase FM explicit +1 ply)\n";
#else
    out << "0 (Router-LMR only; Phase FM counterfactual cohort)\n";
#endif
    out << "  predicate               : Router actual +1=no, staticEval>=1200, FM<1.625\n"
        << "  candidate moves         : " << phase_experiment.candidate_moves << '\n'
        << "  actual depth +1 moves   : " << phase_experiment.adjusted_moves << " ("
        << Percent(phase_experiment.adjusted_moves, stats.phase_fm_lmr_moves)
        << "% of all LMR)\n"
        << "  candidate outcomes      :";
    PrintPhaseFmLmrValue(out, phase_experiment.candidate_outcomes);
    out << "\n  actual +1 outcomes       :";
    PrintPhaseFmLmrValue(out, phase_experiment.adjusted_outcomes);
    out << "\n  actual +1 by bucket      :";
    for (std::size_t bucket = 0; bucket < kBucketCount; ++bucket)
        out << " B" << std::setw(2) << std::setfill('0') << bucket << std::setfill(' ')
            << '=' << phase_experiment.adjusted_by_bucket[bucket];
    out << "\n  actual +1 by depth       :";
    for (std::size_t group = 0; group < kConditionBins; ++group)
        out << ' ' << depth_labels[group] << '=' << phase_experiment.adjusted_by_depth[group];
    out << "\n  actual +1 by moveCount   :";
    for (std::size_t group = 0; group < kConditionBins; ++group)
        out << ' ' << move_labels[group] << '=' << phase_experiment.adjusted_by_move_count[group];
    out << '\n';

    out << "[LCA correction magnitude LMR move outcomes]\n"
        << "  signal: byte-domain |Diff after LCA - Diff before LCA| over 32 channels\n"
        << "  all LMR moves          : " << stats.lca_lmr_moves << '\n'
        << "  signal-valid LMR moves : " << stats.lca_lmr_signal_valid_moves << " ("
        << Percent(stats.lca_lmr_signal_valid_moves, stats.lca_lmr_moves) << "%)\n"
        << "  by mean_abs_delta\n";
    for (std::size_t bin = 0; bin < kLcaMeanDeltaBins; ++bin) {
        const auto& value = stats.lca_lmr_by_mean_delta[bin];
        if (!value.moves)
            continue;
        out << "    " << LcaMeanDeltaLabel(bin);
        PrintPhaseFmLmrValue(out, value);
        out << '\n';
    }
    out << "  by max_abs_delta\n";
    for (std::size_t bin = 0; bin < kLcaMaxDeltaBins; ++bin) {
        const auto& value = stats.lca_lmr_by_max_delta[bin];
        if (!value.moves)
            continue;
        out << "    " << LcaMaxDeltaLabel(bin);
        PrintPhaseFmLmrValue(out, value);
        out << '\n';
    }
    PrintConditionalLcaLmr(out, "LCA correction LMR conditioned by depth",
                           depth_labels, stats.lca_lmr_by_depth);
    PrintConditionalLcaLmr(out, "LCA correction LMR conditioned by move count",
                           move_labels, stats.lca_lmr_by_move_count);
    PrintConditionalLcaLmr(out, "LCA correction LMR conditioned by selected bucket",
                           bucket_labels, stats.lca_lmr_by_bucket);
    PrintConditionalLcaLmr(out, "LCA correction LMR conditioned by |staticEval|",
                           magnitude_labels, stats.lca_lmr_by_static_eval);

    static constexpr std::array<const char*, kLcaMagnitudeGroups> lca_group_labels = {
      "mean <2.00", "mean 2.00-<8.00", "mean 8.00-<32.00", "mean >=32.00"};
    out << "[router margin x LCA correction LMR outcomes]\n";
    for (std::size_t router = 0; router < kRouterMarginGroups; ++router) {
        out << "  {" << router_group_labels[router] << "}\n";
        for (std::size_t lca = 0; lca < kLcaMagnitudeGroups; ++lca) {
            out << "    " << lca_group_labels[lca];
            PrintPhaseFmLmrValue(out, stats.lca_lmr_by_router_and_magnitude[router][lca]);
            out << '\n';
        }
    }

    std::size_t lca_qualifying_strata = 0;
    std::array<double, kLcaMagnitudeGroups> lca_joint_fh{};
    std::array<double, kLcaMagnitudeGroups> lca_joint_research{};
    std::array<double, kLcaMagnitudeGroups> lca_joint_cutoff{};
    for (std::size_t base = 0; base < kFmJointBaseStrata; ++base) {
        bool usable = true;
        for (std::size_t lca = 0; lca < kLcaMagnitudeGroups; ++lca)
            usable &= stats.lca_lmr_joint[base * kLcaMagnitudeGroups + lca].moves >= 20;
        if (!usable)
            continue;
        ++lca_qualifying_strata;
        for (std::size_t lca = 0; lca < kLcaMagnitudeGroups; ++lca) {
            const auto& value = stats.lca_lmr_joint[base * kLcaMagnitudeGroups + lca];
            lca_joint_fh[lca] += Percent(value.reduced_fail_highs, value.moves);
            lca_joint_research[lca] += Percent(value.researches, value.moves);
            lca_joint_cutoff[lca] += Percent(value.final_cutoffs, value.moves);
        }
    }
    out << "[LCA correction jointly controlled LMR outcomes]\n"
        << "  controls: depth x moveCount x bucket x |staticEval| x router-margin\n"
        << "  qualifying strata (>=20 moves in every LCA group): "
        << lca_qualifying_strata << '\n';
    const double lca_divisor = lca_qualifying_strata ? double(lca_qualifying_strata) : 1.0;
    for (std::size_t lca = 0; lca < kLcaMagnitudeGroups; ++lca)
        out << "  " << lca_group_labels[lca]
            << " macro reduced-FH=" << lca_joint_fh[lca] / lca_divisor << '%'
            << " re-search=" << lca_joint_research[lca] / lca_divisor << '%'
            << " cutoff=" << lca_joint_cutoff[lca] / lca_divisor << "%\n";

    out << "[LCA high-tail counterfactual cohorts: no actual Router +1 ply]\n";
    for (std::size_t threshold = 0; threshold < kLcaThresholdCount; ++threshold) {
        const auto& cohort = stats.lca_non_router_thresholds[threshold];
        out << "  threshold >=" << kLcaThresholds[threshold]
            << " all-LMR-rate=" << Percent(cohort.total.moves, stats.lca_lmr_moves) << '%';
        PrintPhaseFmLmrValue(out, cohort.total);
        out << '\n';
    }
    const auto print_lca_threshold_breakdown = [&](const char* title, const auto& labels,
                                                    const auto member) {
        out << "[LCA high-tail counterfactual by " << title << "]\n";
        for (std::size_t threshold = 0; threshold < kLcaThresholdCount; ++threshold) {
            const auto& table = stats.lca_non_router_thresholds[threshold].*member;
            out << "  {threshold >=" << kLcaThresholds[threshold] << "}\n";
            for (std::size_t group = 0; group < table.size(); ++group) {
                if (!table[group].moves)
                    continue;
                out << "    " << labels[group];
                PrintPhaseFmLmrValue(out, table[group]);
                out << '\n';
            }
        }
    };
    print_lca_threshold_breakdown("depth", depth_labels,
                                  &LcaThresholdCohortStats::by_depth);
    print_lca_threshold_breakdown("move count", move_labels,
                                  &LcaThresholdCohortStats::by_move_count);
    print_lca_threshold_breakdown("selected bucket", bucket_labels,
                                  &LcaThresholdCohortStats::by_bucket);
    print_lca_threshold_breakdown("|staticEval|", magnitude_labels,
                                  &LcaThresholdCohortStats::by_static_eval);
    print_lca_threshold_breakdown("router margin", router_group_labels,
                                  &LcaThresholdCohortStats::by_router_margin);

    out << "[LCA model-relative high-tail percentiles: no actual Router +1 ply]\n"
        << "  tie policy: include every move equal to the discrete cutoff\n"
        << "  population";
    PrintPhaseFmLmrValue(out, stats.lca_non_router_all);
    out << '\n';
    for (const auto basis_points : kLcaTopBasisPoints) {
        const auto target = (stats.lca_non_router_all.moves * basis_points + 9999) / 10000;
        PhaseFmLmrBucket tail{};
        std::size_t cutoff_sum = kLcaDeltaSumMax;
        for (std::size_t sum = kLcaDeltaSumMax + 1; sum-- > 0;) {
            AddPhaseFmLmr(tail, stats.lca_non_router_by_delta_sum[sum]);
            cutoff_sum = sum;
            if (tail.moves >= target)
                break;
        }
        out << "  top " << (double(basis_points) / 100.0) << '%'
            << " threshold=" << (double(cutoff_sum) / 32.0)
            << " actual-rate=" << Percent(tail.moves, stats.lca_non_router_all.moves) << '%';
        PrintPhaseFmLmrValue(out, tail);
        out << '\n';
    }

    static constexpr std::array<const char*, kCombinedSignalGroups>
      combined_group_labels = {
        "A Router danger only", "B LCA danger only",
        "C Router danger AND LCA danger", "D neither danger"};
    out << "[Router margin x calibrated LCA top-tail complementary groups]\n"
        << "  Router danger: production variant-3 predicate; non-PV, margin<="
        << NNUE_ROUTER_LMR_MARGIN_THRESHOLD << ", "
           "depth=3..8, moveCount=2..8, no bucket exclusion\n"
        << "  LCA danger   : exact delta sum >=" << NNUE_COMBINED_LCA_SUM_THRESHOLD
        << " (mean >=" << (double(NNUE_COMBINED_LCA_SUM_THRESHOLD) / 32.0) << ")\n";
    std::uint64_t combined_classified_moves = 0;
    for (const auto& group : stats.combined_signal)
        combined_classified_moves += group.moves;
    out << "  classified signal-valid LMR moves: " << combined_classified_moves
        << " / " << stats.lca_lmr_moves << " ("
        << Percent(combined_classified_moves, stats.lca_lmr_moves) << "%)\n";
    for (std::size_t group = 0; group < kCombinedSignalGroups; ++group) {
        out << "  " << combined_group_labels[group]
            << " all-LMR-rate="
            << Percent(stats.combined_signal[group].moves, stats.lca_lmr_moves) << '%';
        PrintPhaseFmLmrValue(out, stats.combined_signal[group]);
        out << '\n';
    }
    const auto print_combined_breakdown =
      [&](const char* title, const auto& labels, const auto member) {
        out << "  by " << title << '\n';
        for (std::size_t group = 0; group < kCombinedSignalGroups; ++group) {
            out << "    {" << combined_group_labels[group] << "}\n";
            const auto& table = stats.*member;
            for (std::size_t bin = 0; bin < table[group].size(); ++bin) {
                if (!table[group][bin].moves)
                    continue;
                out << "      " << labels[bin];
                PrintPhaseFmLmrValue(out, table[group][bin]);
                out << '\n';
            }
        }
      };
    print_combined_breakdown("depth", depth_labels,
                             &ThreadStats::combined_signal_by_depth);
    print_combined_breakdown("moveCount", move_labels,
                             &ThreadStats::combined_signal_by_move_count);
    print_combined_breakdown("selected bucket", bucket_labels,
                             &ThreadStats::combined_signal_by_bucket);
    print_combined_breakdown("|staticEval|", magnitude_labels,
                             &ThreadStats::combined_signal_by_static_eval);

    out << "[Diff RMS energy LMR move outcomes]\n"
        << "  signal: existing RMSNorm sum_sq over 32 Diff value channels; "
           "labels show sqrt(sum_sq/32) ranges\n"
        << "  implementation: reuses sum_sq; no extra channel reduction or sqrt\n";
    for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin) {
        const auto& value = stats.diff_rms_lmr[bin];
        if (!value.moves)
            continue;
        out << "  " << DiffRmsEnergyLabel(bin);
        PrintPhaseFmLmrValue(out, value);
        out << '\n';
    }
    PrintConditionalDiffRms(out, "Diff RMS conditioned by depth",
                            depth_labels, stats.diff_rms_lmr_by_depth);
    PrintConditionalDiffRms(out, "Diff RMS conditioned by move count",
                            move_labels, stats.diff_rms_lmr_by_move_count);
    PrintConditionalDiffRms(out, "Diff RMS conditioned by selected bucket",
                            bucket_labels, stats.diff_rms_lmr_by_bucket);
    PrintConditionalDiffRms(out, "Diff RMS conditioned by |staticEval|",
                            magnitude_labels, stats.diff_rms_lmr_by_static_eval);
    PrintConditionalDiffRms(out, "Diff RMS conditioned by router margin",
                            router_group_labels, stats.diff_rms_lmr_by_router);
    static constexpr std::array<const char*, 2> lca_tail_labels = {
      "below calibrated LCA tail", "calibrated LCA high-tail"};
    PrintConditionalDiffRms(out, "Diff RMS conditioned by LCA high-tail",
                            lca_tail_labels, stats.diff_rms_lmr_by_lca_tail);

    // Joint control without requiring every RMS bin to be populated in every
    // stratum. For each sufficiently populated cell, subtract its own stratum's
    // overall rate, then macro-average those within-stratum deltas per RMS bin.
    std::array<double, kDiffRmsEnergyBins> rms_joint_fh_delta{};
    std::array<double, kDiffRmsEnergyBins> rms_joint_research_delta{};
    std::array<double, kDiffRmsEnergyBins> rms_joint_cutoff_delta{};
    std::array<std::size_t, kDiffRmsEnergyBins> rms_joint_qualifying{};
    constexpr std::size_t rms_joint_base_count = kFmJointBaseStrata * 2;
    for (std::size_t base = 0; base < rms_joint_base_count; ++base) {
        PhaseFmLmrBucket total{};
        for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin)
            AddPhaseFmLmr(total,
              stats.diff_rms_lmr_joint[base * kDiffRmsEnergyBins + bin]);
        if (!total.moves)
            continue;
        const double total_fh = Percent(total.reduced_fail_highs, total.moves);
        const double total_research = Percent(total.researches, total.moves);
        const double total_cutoff = Percent(total.final_cutoffs, total.moves);
        for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin) {
            const auto& value = stats.diff_rms_lmr_joint[
              base * kDiffRmsEnergyBins + bin];
            if (value.moves < 20)
                continue;
            ++rms_joint_qualifying[bin];
            rms_joint_fh_delta[bin] +=
              Percent(value.reduced_fail_highs, value.moves) - total_fh;
            rms_joint_research_delta[bin] +=
              Percent(value.researches, value.moves) - total_research;
            rms_joint_cutoff_delta[bin] +=
              Percent(value.final_cutoffs, value.moves) - total_cutoff;
        }
    }
    out << "[Diff RMS jointly controlled LMR outcomes]\n"
        << "  controls: depth x moveCount x bucket x |staticEval| x "
           "router-margin x calibrated-LCA-tail\n"
        << "  metric: macro mean within-stratum rate delta; cells require >=20 moves\n";
    for (std::size_t bin = 0; bin < kDiffRmsEnergyBins; ++bin) {
        const double divisor = rms_joint_qualifying[bin]
                             ? double(rms_joint_qualifying[bin]) : 1.0;
        out << "  " << DiffRmsEnergyLabel(bin)
            << " qualifying=" << rms_joint_qualifying[bin]
            << " FH-delta=" << rms_joint_fh_delta[bin] / divisor << "%"
            << " re-search-delta=" << rms_joint_research_delta[bin] / divisor << "%"
            << " cutoff-delta=" << rms_joint_cutoff_delta[bin] / divisor << "%\n";
    }

    const auto print_cross_activity = [&](const char* title, const char* signal_name,
                                           const auto& activity, const auto& edges) {
        out << '[' << title << " LMR move outcomes]\n";
        for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin) {
            const auto& value = activity.all[bin];
            if (!value.moves)
                continue;
            out << "  " << CrossActivityLabel(signal_name, bin, edges);
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
        const std::string prefix = std::string(title) + " conditioned by ";
        PrintConditionalCrossActivity(out, (prefix + "depth").c_str(), signal_name,
                                      depth_labels, activity.by_depth, edges);
        PrintConditionalCrossActivity(out, (prefix + "move count").c_str(), signal_name,
                                      move_labels, activity.by_move_count, edges);
        PrintConditionalCrossActivity(out, (prefix + "selected bucket").c_str(), signal_name,
                                      bucket_labels, activity.by_bucket, edges);
        PrintConditionalCrossActivity(out, (prefix + "|staticEval|").c_str(), signal_name,
                                      magnitude_labels, activity.by_static_eval, edges);
        PrintConditionalCrossActivity(out, (prefix + "router margin").c_str(), signal_name,
                                      router_group_labels, activity.by_router, edges);
        PrintConditionalCrossActivity(out, (prefix + "LCA high-tail").c_str(), signal_name,
                                      lca_tail_labels, activity.by_lca_tail, edges);

        std::array<double, kCrossActivityBins> fh_delta{};
        std::array<double, kCrossActivityBins> research_delta{};
        std::array<double, kCrossActivityBins> cutoff_delta{};
        std::array<std::size_t, kCrossActivityBins> qualifying{};
        constexpr std::size_t base_count = kFmJointBaseStrata * 2;
        for (std::size_t base = 0; base < base_count; ++base) {
            PhaseFmLmrBucket total{};
            for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin)
                AddPhaseFmLmr(total,
                  activity.joint[base * kCrossActivityBins + bin]);
            if (!total.moves)
                continue;
            const double total_fh = Percent(total.reduced_fail_highs, total.moves);
            const double total_research = Percent(total.researches, total.moves);
            const double total_cutoff = Percent(total.final_cutoffs, total.moves);
            for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin) {
                const auto& value = activity.joint[base * kCrossActivityBins + bin];
                if (value.moves < 20)
                    continue;
                ++qualifying[bin];
                fh_delta[bin] += Percent(value.reduced_fail_highs, value.moves) - total_fh;
                research_delta[bin] += Percent(value.researches, value.moves) - total_research;
                cutoff_delta[bin] += Percent(value.final_cutoffs, value.moves) - total_cutoff;
            }
        }
        out << '[' << title << " jointly controlled LMR outcomes]\n"
            << "  controls: depth x moveCount x bucket x |staticEval| x "
               "router-margin x calibrated-LCA-tail\n"
            << "  metric: macro mean within-stratum rate delta; cells require >=20 moves\n";
        for (std::size_t bin = 0; bin < kCrossActivityBins; ++bin) {
            const double divisor = qualifying[bin] ? double(qualifying[bin]) : 1.0;
            out << "  " << CrossActivityLabel(signal_name, bin, edges)
                << " qualifying=" << qualifying[bin]
                << " FH-delta=" << fh_delta[bin] / divisor << "%"
                << " re-search-delta=" << research_delta[bin] / divisor << "%"
                << " cutoff-delta=" << cutoff_delta[bin] / divisor << "%\n";
        }
    };
    print_cross_activity("Cross mean absolute activity", "mean_abs_cross",
                         stats.cross_mean_abs_lmr, kCrossMeanAbsEdges);
    print_cross_activity("Cross max absolute activity", "max_abs_cross",
                         stats.cross_max_abs_lmr, kCrossMaxAbsEdges);
    static constexpr std::array<const char*, kCrossOverlapGroups> overlap_labels = {
      "none", "LCA only", "Router only", "Router + LCA",
      "Cross only", "Cross + LCA", "Cross + Router", "Cross + Router + LCA"};
    const auto print_cross_overlap = [&](const char* title, const auto& groups) {
        std::uint64_t total = 0;
        for (const auto& group : groups)
            total += group.moves;
        out << '[' << title << "]\n";
        for (std::size_t group = 0; group < groups.size(); ++group) {
            const auto& value = groups[group];
            out << "  " << overlap_labels[group]
                << " count=" << value.moves
                << " signal-valid-LMR=" << Percent(value.moves, total) << "%";
            PrintPhaseFmLmrValue(out, value);
            out << '\n';
        }
    };
    print_cross_overlap(
      "Cross/Router/LCA raw-signal overlap",
      stats.cross_overlap_raw);
    print_cross_overlap(
      "Cross/Router/LCA deployable-cohort overlap",
      stats.cross_overlap_eligible);

    out << "[Cross deployable counterfactual thresholds]\n"
        << "  population: signal-valid LMR AND Router actual +1 ply=no"
           " AND calibrated LCA candidate=no\n"
        << "  thresholds are cumulative\n";
    for (std::size_t threshold = 0;
         threshold < kCrossDeployableThresholdCount; ++threshold) {
        const auto& value = stats.cross_deployable_thresholds[threshold].total;
        out << "  max_abs_cross >= "
            << static_cast<unsigned>(kCrossDeployableThresholds[threshold])
            << " count=" << value.moves
            << " all-LMR=" << Percent(value.moves, stats.lca_lmr_moves) << "%"
            << " signal-valid-LMR="
            << Percent(value.moves, stats.lca_lmr_signal_valid_moves) << "%";
        PrintPhaseFmLmrValue(out, value);
        out << '\n';
    }
    const auto print_cross_deployable_breakdown =
      [&](const char* title, const auto& labels, const auto member) {
        out << "[Cross deployable cohorts by " << title << "]\n";
        for (std::size_t group = 0; group < labels.size(); ++group) {
            out << "  {" << labels[group] << "}\n";
            for (std::size_t threshold = 0;
                 threshold < kCrossDeployableThresholdCount; ++threshold) {
                const auto& value =
                  (stats.cross_deployable_thresholds[threshold].*member)[group];
                out << "    max_abs_cross >= "
                    << static_cast<unsigned>(kCrossDeployableThresholds[threshold])
                    << " count=" << value.moves
                    << " all-LMR=" << Percent(value.moves, stats.lca_lmr_moves) << "%";
                PrintPhaseFmLmrValue(out, value);
                out << '\n';
            }
        }
    };
    print_cross_deployable_breakdown(
      "depth", depth_labels, &LcaThresholdCohortStats::by_depth);
    print_cross_deployable_breakdown(
      "moveCount", move_labels, &LcaThresholdCohortStats::by_move_count);
    print_cross_deployable_breakdown(
      "selected bucket", bucket_labels, &LcaThresholdCohortStats::by_bucket);
    print_cross_deployable_breakdown(
      "|staticEval|", magnitude_labels, &LcaThresholdCohortStats::by_static_eval);

    static constexpr std::array<const char*, 2> cross_saturation_labels = {
      "Cross max <127", "Cross max >=127"};
    out << "[Main gate activity definition]\n"
        << "  representation: integer Q64 sigmoid, observed/design range [0,63]\n"
        << "  mean: exact gate sum / 32\n"
        << "  saturated low: gate <=1\n"
        << "  saturated high: gate >=63\n";
    for (std::size_t kind = 0; kind < kMainGateSignalCount; ++kind) {
        const auto& activity = stats.main_gate_activity[kind];
        const auto print_values = [&](const char* title, const auto& values) {
            out << '[' << title << "]\n";
            for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin) {
                const auto& value = values[bin];
                if (!value.moves)
                    continue;
                out << "  " << MainGateActivityLabel(kind, bin)
                    << " all-LMR=" << Percent(value.moves, stats.lca_lmr_moves) << '%';
                PrintPhaseFmLmrValue(out, value);
                out << '\n';
            }
        };
        const auto print_conditioned = [&](const char* title, const auto& labels,
                                            const auto& table) {
            out << '[' << title << "]\n";
            for (std::size_t group = 0; group < labels.size(); ++group) {
                out << "  {" << labels[group] << "}\n";
                for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin) {
                    const auto& value = table[group][bin];
                    if (!value.moves)
                        continue;
                    out << "    " << MainGateActivityLabel(kind, bin);
                    PrintPhaseFmLmrValue(out, value);
                    out << '\n';
                }
            }
        };
        const std::string signal_name = MainGateActivityName(kind);
        print_values((signal_name + " LMR move outcomes").c_str(), activity.all);
        print_conditioned((signal_name + " conditioned by depth").c_str(),
                          depth_labels, activity.by_depth);
        print_conditioned((signal_name + " conditioned by moveCount").c_str(),
                          move_labels, activity.by_move_count);
        print_conditioned((signal_name + " conditioned by selected bucket").c_str(),
                          bucket_labels, activity.by_bucket);
        print_conditioned((signal_name + " conditioned by |staticEval|").c_str(),
                          magnitude_labels, activity.by_static_eval);
        print_conditioned((signal_name + " conditioned by router margin").c_str(),
                          router_group_labels, activity.by_router);
        print_conditioned((signal_name + " conditioned by LCA high-tail").c_str(),
                          lca_tail_labels, activity.by_lca_tail);
        print_conditioned((signal_name + " conditioned by Cross saturation").c_str(),
                          cross_saturation_labels, activity.by_cross_saturation);
        print_values((signal_name + " additional coverage (Router/LCA/Cross actual +1=no)").c_str(),
                     activity.uncovered);

        std::array<double, kMainGateActivityBins> fh_delta{};
        std::array<double, kMainGateActivityBins> research_delta{};
        std::array<double, kMainGateActivityBins> cutoff_delta{};
        std::array<std::size_t, kMainGateActivityBins> qualifying{};
        for (std::size_t base = 0; base < kMainGateJointBaseStrata; ++base) {
            PhaseFmLmrBucket total{};
            for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin)
                AddPhaseFmLmr(total,
                  activity.joint[base * kMainGateActivityBins + bin]);
            if (!total.moves)
                continue;
            const double total_fh = Percent(total.reduced_fail_highs, total.moves);
            const double total_research = Percent(total.researches, total.moves);
            const double total_cutoff = Percent(total.final_cutoffs, total.moves);
            for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin) {
                const auto& value = activity.joint[base * kMainGateActivityBins + bin];
                if (value.moves < 20)
                    continue;
                ++qualifying[bin];
                fh_delta[bin] += Percent(value.reduced_fail_highs, value.moves) - total_fh;
                research_delta[bin] += Percent(value.researches, value.moves) - total_research;
                cutoff_delta[bin] += Percent(value.final_cutoffs, value.moves) - total_cutoff;
            }
        }
        out << '[' << signal_name << " jointly controlled LMR outcomes]\n"
            << "  controls: depth x moveCount x bucket x |staticEval| x router-margin"
               " x calibrated-LCA-tail x Cross-saturation\n"
            << "  metric: macro mean within-stratum rate delta; cells require >=20 moves\n";
        for (std::size_t bin = 0; bin < kMainGateActivityBins; ++bin) {
            const double divisor = qualifying[bin] ? double(qualifying[bin]) : 1.0;
            out << "  " << MainGateActivityLabel(kind, bin)
                << " qualifying=" << qualifying[bin]
                << " FH-delta=" << fh_delta[bin] / divisor << '%'
                << " re-search-delta=" << research_delta[bin] / divisor << '%'
                << " cutoff-delta=" << cutoff_delta[bin] / divisor << "%\n";
        }
    }

    out << "[FM activity definition]\n"
        << "  Diff output: uint8 [0,127], neutral=64, activity=abs(q-64)\n"
        << "  Diff saturation: q<=1 or q>=126\n"
        << "  Abs output: uint8 [0,127], activity=q\n"
        << "  Abs saturation: q>=126\n"
        << "  mean classification uses exact integer sum / 32\n";
    for (std::size_t kind = 0; kind < kFmActivitySignalCount; ++kind) {
        const auto& activity = stats.fm_activity[kind];
        const auto print_values = [&](const char* title, const auto& values) {
            out << '[' << title << "]\n";
            for (std::size_t bin = 0; bin < kFmActivityBins; ++bin) {
                const auto& value = values[bin];
                if (!value.moves)
                    continue;
                out << "  " << FmActivityLabel(kind, bin)
                    << " all-LMR=" << Percent(value.moves, stats.lca_lmr_moves) << '%';
                PrintPhaseFmLmrValue(out, value);
                out << '\n';
            }
        };
        const auto print_conditioned = [&](const char* title, const auto& labels,
                                            const auto& table) {
            out << '[' << title << "]\n";
            for (std::size_t group = 0; group < labels.size(); ++group) {
                out << "  {" << labels[group] << "}\n";
                for (std::size_t bin = 0; bin < kFmActivityBins; ++bin) {
                    const auto& value = table[group][bin];
                    if (!value.moves)
                        continue;
                    out << "    " << FmActivityLabel(kind, bin);
                    PrintPhaseFmLmrValue(out, value);
                    out << '\n';
                }
            }
        };
        const std::string signal_name = FmActivityName(kind);
        print_values((signal_name + " LMR move outcomes").c_str(), activity.all);
        print_conditioned((signal_name + " conditioned by depth").c_str(),
                          depth_labels, activity.by_depth);
        print_conditioned((signal_name + " conditioned by moveCount").c_str(),
                          move_labels, activity.by_move_count);
        print_conditioned((signal_name + " conditioned by selected bucket").c_str(),
                          bucket_labels, activity.by_bucket);
        print_conditioned((signal_name + " conditioned by |staticEval|").c_str(),
                          magnitude_labels, activity.by_static_eval);
        print_conditioned((signal_name + " conditioned by router margin").c_str(),
                          router_group_labels, activity.by_router);
        print_conditioned((signal_name + " conditioned by LCA high-tail").c_str(),
                          lca_tail_labels, activity.by_lca_tail);
        print_conditioned((signal_name + " conditioned by Cross saturation").c_str(),
                          cross_saturation_labels, activity.by_cross_saturation);
        print_values((signal_name +
          " additional coverage (Router/LCA/Cross actual +1=no)").c_str(),
          activity.uncovered);

        std::array<double, kFmActivityBins> fh_delta{};
        std::array<double, kFmActivityBins> research_delta{};
        std::array<double, kFmActivityBins> cutoff_delta{};
        std::array<std::size_t, kFmActivityBins> qualifying{};
        for (std::size_t base = 0; base < kMainGateJointBaseStrata; ++base) {
            PhaseFmLmrBucket total{};
            for (std::size_t bin = 0; bin < kFmActivityBins; ++bin)
                AddPhaseFmLmr(total, activity.joint[base * kFmActivityBins + bin]);
            if (!total.moves)
                continue;
            const double total_fh = Percent(total.reduced_fail_highs, total.moves);
            const double total_research = Percent(total.researches, total.moves);
            const double total_cutoff = Percent(total.final_cutoffs, total.moves);
            for (std::size_t bin = 0; bin < kFmActivityBins; ++bin) {
                const auto& value = activity.joint[base * kFmActivityBins + bin];
                if (value.moves < 20)
                    continue;
                ++qualifying[bin];
                fh_delta[bin] += Percent(value.reduced_fail_highs, value.moves) - total_fh;
                research_delta[bin] += Percent(value.researches, value.moves) - total_research;
                cutoff_delta[bin] += Percent(value.final_cutoffs, value.moves) - total_cutoff;
            }
        }
        out << '[' << signal_name << " jointly controlled LMR outcomes]\n"
            << "  controls: depth x moveCount x bucket x |staticEval| x router-margin"
               " x calibrated-LCA-tail x Cross-saturation\n"
            << "  metric: macro mean within-stratum rate delta; cells require >=20 moves\n";
        for (std::size_t bin = 0; bin < kFmActivityBins; ++bin) {
            const double divisor = qualifying[bin] ? double(qualifying[bin]) : 1.0;
            out << "  " << FmActivityLabel(kind, bin)
                << " qualifying=" << qualifying[bin]
                << " FH-delta=" << fh_delta[bin] / divisor << '%'
                << " re-search-delta=" << research_delta[bin] / divisor << '%'
                << " cutoff-delta=" << cutoff_delta[bin] / divisor << "%\n";
        }
    }

#if defined(ENABLE_NNUE_LCA_LMR_EXPERIMENT)
    const auto& lca_experiment = stats.lca_lmr_experiment;
    out << "[LCA top-tail explicit +1-ply experiment]\n"
        << "  variant             : " << NNUE_LCA_LMR_VARIANT
        << (NNUE_LCA_LMR_VARIANT == 0 ? " (production Router-LMR only)\n"
                                     : " (Router-LMR + LCA explicit +1 ply)\n")
        << "  exact sum threshold : " << NNUE_LCA_LMR_SUM_THRESHOLD << '\n'
        << "  mean threshold      : " << (double(NNUE_LCA_LMR_SUM_THRESHOLD) / 32.0)
        << '\n'
        << "  overall LMR";
    PrintPhaseFmLmrValue(out, stats.lca_lmr_all_outcomes);
    out << "\n  LCA candidate";
    PrintPhaseFmLmrValue(out, lca_experiment.candidate_outcomes);
    out << "\n  candidate / all LMR : "
        << Percent(lca_experiment.candidate_outcomes.moves, stats.lca_lmr_moves) << "%\n"
        << "  actual +1 ply";
    PrintPhaseFmLmrValue(out, lca_experiment.adjusted_outcomes);
    out << "\n  actual +1 / all LMR : "
        << Percent(lca_experiment.adjusted_outcomes.moves, stats.lca_lmr_moves) << "%\n";

    const auto print_lca_experiment_breakdown =
      [&](const char* title, const auto& labels, const auto candidate_member,
          const auto adjusted_member) {
        const auto& candidate = lca_experiment.*candidate_member;
        const auto& adjusted = lca_experiment.*adjusted_member;
        out << "  by " << title << '\n';
        for (std::size_t group = 0; group < candidate.size(); ++group) {
            if (!candidate[group].moves && !adjusted[group].moves)
                continue;
            out << "    " << labels[group] << " candidate";
            PrintPhaseFmLmrValue(out, candidate[group]);
            out << " actual+1";
            PrintPhaseFmLmrValue(out, adjusted[group]);
            out << '\n';
        }
    };
    print_lca_experiment_breakdown("bucket", bucket_labels,
      &LcaLmrExperimentStats::candidate_by_bucket,
      &LcaLmrExperimentStats::adjusted_by_bucket);
    print_lca_experiment_breakdown("depth", depth_labels,
      &LcaLmrExperimentStats::candidate_by_depth,
      &LcaLmrExperimentStats::adjusted_by_depth);
    print_lca_experiment_breakdown("moveCount", move_labels,
      &LcaLmrExperimentStats::candidate_by_move_count,
      &LcaLmrExperimentStats::adjusted_by_move_count);
    print_lca_experiment_breakdown("|staticEval|", magnitude_labels,
      &LcaLmrExperimentStats::candidate_by_static_eval,
      &LcaLmrExperimentStats::adjusted_by_static_eval);
#endif

#if defined(ENABLE_NNUE_CROSS_LMR_EXPERIMENT)
    const auto& cross_experiment = stats.cross_lmr_experiment;
    out << "[Cross max explicit +1-ply experiment]\n"
        << "  variant             : " << NNUE_CROSS_LMR_VARIANT
        << (NNUE_CROSS_LMR_VARIANT == 0 ? " (shadow current Router + LCA)\n"
                                        : " (Router + LCA + Cross explicit +1 ply)\n")
        << "  max threshold       : " << NNUE_CROSS_LMR_MAX_THRESHOLD << '\n'
        << "  effective predicate : depth 3..8, moveCount 2..8; Router/LCA actual +1=no\n"
        << "  overall LMR";
    PrintPhaseFmLmrValue(out, stats.lca_lmr_all_outcomes);
    out << "\n  Cross candidate";
    PrintPhaseFmLmrValue(out, cross_experiment.candidate_outcomes);
    out << "\n  candidate / all LMR : "
        << Percent(cross_experiment.candidate_outcomes.moves, stats.lca_lmr_moves) << "%\n"
        << "  actual +1 ply";
    PrintPhaseFmLmrValue(out, cross_experiment.adjusted_outcomes);
    out << "\n  actual +1 / all LMR : "
        << Percent(cross_experiment.adjusted_outcomes.moves, stats.lca_lmr_moves) << "%\n";

    const auto print_cross_experiment_breakdown =
      [&](const char* title, const auto& labels, const auto candidate_member,
          const auto adjusted_member) {
        const auto& candidate = cross_experiment.*candidate_member;
        const auto& adjusted = cross_experiment.*adjusted_member;
        out << "  by " << title << '\n';
        for (std::size_t group = 0; group < candidate.size(); ++group) {
            if (!candidate[group].moves && !adjusted[group].moves)
                continue;
            out << "    " << labels[group] << " candidate";
            PrintPhaseFmLmrValue(out, candidate[group]);
            out << " actual+1";
            PrintPhaseFmLmrValue(out, adjusted[group]);
            out << '\n';
        }
    };
    print_cross_experiment_breakdown("bucket", bucket_labels,
      &LcaLmrExperimentStats::candidate_by_bucket,
      &LcaLmrExperimentStats::adjusted_by_bucket);
    print_cross_experiment_breakdown("depth", depth_labels,
      &LcaLmrExperimentStats::candidate_by_depth,
      &LcaLmrExperimentStats::adjusted_by_depth);
    print_cross_experiment_breakdown("moveCount", move_labels,
      &LcaLmrExperimentStats::candidate_by_move_count,
      &LcaLmrExperimentStats::adjusted_by_move_count);
    print_cross_experiment_breakdown("|staticEval|", magnitude_labels,
      &LcaLmrExperimentStats::candidate_by_static_eval,
      &LcaLmrExperimentStats::adjusted_by_static_eval);
#endif

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

struct CalibrationLcaTail {
    std::uint32_t basis_points = 0;
    std::size_t threshold_sum = 0;
    PhaseFmLmrBucket outcome{};
};

inline std::array<CalibrationLcaTail, 3> BuildCalibrationLcaTails(
  const ThreadStats& stats) {
    static constexpr std::array<std::uint32_t, 3> basis_points = {100, 250, 500};
    std::array<CalibrationLcaTail, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        auto& tail = result[index];
        tail.basis_points = basis_points[index];
        const auto target =
          (stats.signal_calibration.lca_calibration_eligible_moves
             * tail.basis_points + 9999) / 10000;
        tail.threshold_sum = kLcaDeltaSumMax;
        if (!target)
            continue;
        for (std::size_t sum = kLcaDeltaSumMax + 1; sum-- > 0;) {
            AddPhaseFmLmr(tail.outcome,
              stats.signal_calibration.lca_eligible_by_delta_sum[sum]);
            tail.threshold_sum = sum;
            if (tail.outcome.moves >= target)
                break;
        }
    }
    return result;
}

inline std::size_t CalibrationLcaQuantile(const ThreadStats& stats,
                                          const std::uint32_t basis_points) {
    if (!stats.signal_calibration.lca_calibration_eligible_moves)
        return 0;
    const auto target =
      std::max<std::uint64_t>(1,
        (stats.signal_calibration.lca_calibration_eligible_moves
           * basis_points + 9999) / 10000);
    std::uint64_t cumulative = 0;
    for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum) {
        cumulative += stats.signal_calibration.lca_eligible_by_delta_sum[sum].moves;
        if (cumulative >= target)
            return sum;
    }
    return kLcaDeltaSumMax;
}

inline std::array<PhaseFmLmrBucket, kCalibrationCrossThresholdCount>
BuildCalibrationCrossEligible(const ThreadStats& stats,
                              const std::size_t lca_top1_sum) {
    std::array<PhaseFmLmrBucket, kCalibrationCrossThresholdCount> result{};
    for (std::size_t threshold = 0;
         threshold < kCalibrationCrossThresholdCount; ++threshold)
        for (std::size_t sum = 0; sum < lca_top1_sum; ++sum)
            AddPhaseFmLmr(result[threshold],
              stats.signal_calibration.cross_after_router_by_lca_sum[threshold][sum]);
    return result;
}

inline PhaseFmLmrBucket BuildCalibrationCrossEligiblePopulation(
  const ThreadStats& stats, const std::size_t lca_top1_sum) {
    PhaseFmLmrBucket result{};
    for (std::size_t sum = 0; sum < lca_top1_sum; ++sum)
        AddPhaseFmLmr(result,
          stats.signal_calibration.cross_after_router_all_by_lca_sum[sum]);
    return result;
}

inline std::array<PhaseFmLmrBucket, 8> BuildCalibrationOverlap(
  const ThreadStats& stats, const std::size_t lca_top1_sum) {
    std::array<PhaseFmLmrBucket, 8> result{};
    for (std::size_t sum = 0; sum <= kLcaDeltaSumMax; ++sum)
        for (std::size_t flags = 0; flags < 4; ++flags) {
            const bool router = (flags & 1U) != 0;
            const bool cross = (flags & 2U) != 0;
            const bool lca = sum >= lca_top1_sum;
            const auto mask = (router ? 1U : 0U) | (lca ? 2U : 0U)
                            | (cross ? 4U : 0U);
            AddPhaseFmLmr(result[mask],
              stats.signal_calibration.overlap_by_lca_sum[sum * 4 + flags]);
        }
    return result;
}

inline void PrintCalibrationRow(std::ostream& out, const char* label,
                                const PhaseFmLmrBucket& value,
                                const std::uint64_t population) {
    out << "  " << std::left << std::setw(22) << label << std::right
        << " count=" << std::setw(10) << value.moves
        << " target=" << std::fixed << std::setprecision(3)
        << std::setw(8) << Percent(value.moves, population) << '%'
        << " re-search=" << std::setw(8) << Percent(value.researches, value.moves) << '%'
        << " fail-high=" << std::setw(8)
        << Percent(value.reduced_fail_highs, value.moves) << '%'
        << " cutoff=" << std::setw(8) << Percent(value.final_cutoffs, value.moves) << "%\n";
}

inline void CalibrationReport(std::ostream& out) {
    const auto stats_storage = std::make_unique<ThreadStats>();
    Snapshot(*stats_storage);
    const auto& stats = *stats_storage;
    const auto& calibration = stats.signal_calibration;
    const auto lca_tails = BuildCalibrationLcaTails(stats);
    const auto overlap = BuildCalibrationOverlap(stats, lca_tails[0].threshold_sum);
    const auto cross_eligible =
      BuildCalibrationCrossEligible(stats, lca_tails[0].threshold_sum);
    const auto cross_eligible_population =
      BuildCalibrationCrossEligiblePopulation(stats, lca_tails[0].threshold_sum);

    out << "[NNUE signal calibration]\n"
        << "  scope: completed signal-valid LMR moves; search behavior is unchanged\n"
        << "  signal-valid LMR moves: " << calibration.signal_valid_lmr_moves << "\n"
        << "  Router population: non-PV, depth 3..8, moveCount 2..8\n"
        << "  LCA population: moves not actually deepened by Router-LMR\n"
        << "  Cross eligible population: depth 3..8, moveCount 2..8, not actually deepened by Router/LCA\n";

    out << "[Router margin distribution]\n";
    for (std::size_t bin = 0; bin < kCalibrationRouterBins; ++bin)
        PrintCalibrationRow(out, CalibrationRouterBinLabel(bin),
                            calibration.router_distribution[bin],
                            calibration.signal_valid_lmr_moves);
    out << "[Router threshold cohorts]\n";
    for (std::size_t index = 0; index < kCalibrationRouterThresholdCount; ++index) {
        const auto label = std::string("margin <= ")
                         + std::to_string(kCalibrationRouterThresholds[index]);
        PrintCalibrationRow(out, label.c_str(), calibration.router_cohorts[index],
                            calibration.router_structural_moves);
    }

    static constexpr std::array<std::uint32_t, 9> quantiles = {
      1, 1000, 2500, 5000, 7500, 9000, 9500, 9900, 10000};
    static constexpr std::array<const char*, 9> quantile_labels = {
      "min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"};
    out << "[LCA abs-delta-sum distribution]\n";
    for (std::size_t index = 0; index < quantiles.size(); ++index) {
        const auto sum = CalibrationLcaQuantile(stats, quantiles[index]);
        out << "  " << std::left << std::setw(5) << quantile_labels[index] << std::right
            << " sum=" << std::setw(4) << sum << " mean="
            << std::fixed << std::setprecision(5) << (double(sum) / 32.0) << '\n';
    }
    out << "[LCA model-relative high tails]\n"
        << "  tie policy: include all moves equal to the integer cutoff\n";
    for (const auto& tail : lca_tails) {
        const auto label = std::string("top ")
          + std::to_string(double(tail.basis_points) / 100.0) + "%";
        out << "  cutoff: " << label << " sum >= " << tail.threshold_sum
            << " (mean >= " << std::fixed << std::setprecision(5)
            << (double(tail.threshold_sum) / 32.0) << ")\n";
        PrintCalibrationRow(out, label.c_str(), tail.outcome,
                            calibration.lca_calibration_eligible_moves);
    }

    out << "[Cross max activity thresholds: all signal-valid LMR]\n";
    for (std::size_t index = 0; index < kCalibrationCrossThresholdCount; ++index) {
        const auto label = std::string("max_cross >= ")
                         + std::to_string(kCalibrationCrossThresholds[index]);
        PrintCalibrationRow(out, label.c_str(), calibration.cross_all_cohorts[index],
                            calibration.signal_valid_lmr_moves);
    }
    out << "[Cross max activity thresholds: production-eligible]\n";
    for (std::size_t index = 0; index < kCalibrationCrossThresholdCount; ++index) {
        const auto label = std::string("max_cross >= ")
                         + std::to_string(kCalibrationCrossThresholds[index]);
        PrintCalibrationRow(out, label.c_str(), cross_eligible[index],
                            cross_eligible_population.moves);
    }

    out << "[Production candidate/application coverage]\n"
        << "  Router actual +1 ply : " << calibration.router_actual_adjusted_moves
        << " / " << calibration.signal_valid_lmr_moves << " ("
        << Percent(calibration.router_actual_adjusted_moves,
                   calibration.signal_valid_lmr_moves) << "%)\n"
        << "  LCA calibrated top1 candidate: " << lca_tails[0].outcome.moves
        << " / " << calibration.signal_valid_lmr_moves << " ("
        << Percent(lca_tails[0].outcome.moves,
                   calibration.signal_valid_lmr_moves) << "%)\n"
        << "  Cross calibrated >=127 candidate: "
        << cross_eligible[kCalibrationCrossThresholdCount - 1].moves
        << " / " << calibration.signal_valid_lmr_moves << " ("
        << Percent(cross_eligible[kCalibrationCrossThresholdCount - 1].moves,
                   calibration.signal_valid_lmr_moves) << "%)\n"
        << "  note: LCA/Cross rates above are calibrated counterfactual candidates;"
           " no LMR decision is changed.\n";

    static constexpr std::array<const char*, 8> overlap_labels = {
      "none", "Router-only", "LCA-only", "Router+LCA",
      "Cross-only", "Router+Cross", "LCA+Cross", "Router+LCA+Cross"};
    out << "[Calibrated signal overlap]\n"
        << "  Router=production danger predicate; LCA=run-specific top1%; Cross=max>=127 with production depth/moveCount\n";
    PhaseFmLmrBucket any_overlap{};
    for (std::size_t mask = 0; mask < overlap.size(); ++mask) {
        PrintCalibrationRow(out, overlap_labels[mask], overlap[mask],
                            calibration.signal_valid_lmr_moves);
        if (mask == 3 || mask == 5 || mask == 6 || mask == 7)
            AddPhaseFmLmr(any_overlap, overlap[mask]);
    }
    PrintCalibrationRow(out, "any overlap", any_overlap,
                        calibration.signal_valid_lmr_moves);
}

inline void WriteCalibrationOutcomeJson(std::ostream& out,
                                        const PhaseFmLmrBucket& value,
                                        const std::uint64_t population) {
    out << "{\"count\":" << value.moves
        << ",\"target_rate_percent\":" << Percent(value.moves, population)
        << ",\"research_rate_percent\":" << Percent(value.researches, value.moves)
        << ",\"fail_high_rate_percent\":"
        << Percent(value.reduced_fail_highs, value.moves)
        << ",\"cutoff_rate_percent\":" << Percent(value.final_cutoffs, value.moves)
        << '}';
}

inline void CalibrationReportJson(std::ostream& out) {
    const auto stats_storage = std::make_unique<ThreadStats>();
    Snapshot(*stats_storage);
    const auto& stats = *stats_storage;
    const auto& calibration = stats.signal_calibration;
    const auto lca_tails = BuildCalibrationLcaTails(stats);
    const auto overlap = BuildCalibrationOverlap(stats, lca_tails[0].threshold_sum);
    const auto cross_eligible =
      BuildCalibrationCrossEligible(stats, lca_tails[0].threshold_sum);
    const auto cross_eligible_population =
      BuildCalibrationCrossEligiblePopulation(stats, lca_tails[0].threshold_sum);
    out << std::fixed << std::setprecision(6)
        << "{\n  \"schema_version\":1,\n  \"population\":{"
        << "\"signal_valid_lmr\":" << calibration.signal_valid_lmr_moves
        << ",\"router_structural\":" << calibration.router_structural_moves
        << ",\"lca_non_router\":" << calibration.lca_calibration_eligible_moves
        << ",\"cross_eligible\":" << cross_eligible_population.moves
        << "},\n  \"router\":{\n    \"distribution\":[";
    for (std::size_t bin = 0; bin < kCalibrationRouterBins; ++bin) {
        if (bin) out << ',';
        out << "{\"range\":\"" << CalibrationRouterBinLabel(bin) << "\",\"outcome\":";
        WriteCalibrationOutcomeJson(out, calibration.router_distribution[bin],
                                    calibration.signal_valid_lmr_moves);
        out << '}';
    }
    out << "],\n    \"cohorts\":[";
    for (std::size_t index = 0; index < kCalibrationRouterThresholdCount; ++index) {
        if (index) out << ',';
        out << "{\"threshold\":" << kCalibrationRouterThresholds[index]
            << ",\"outcome\":";
        WriteCalibrationOutcomeJson(out, calibration.router_cohorts[index],
                                    calibration.router_structural_moves);
        out << '}';
    }
    out << "]\n  },\n  \"lca\":{\n    \"quantiles\":{";
    static constexpr std::array<std::uint32_t, 9> quantiles = {
      1, 1000, 2500, 5000, 7500, 9000, 9500, 9900, 10000};
    static constexpr std::array<const char*, 9> quantile_labels = {
      "min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"};
    for (std::size_t index = 0; index < quantiles.size(); ++index) {
        if (index) out << ',';
        out << '\"' << quantile_labels[index] << "\":"
            << CalibrationLcaQuantile(stats, quantiles[index]);
    }
    out << "},\n    \"tails\":[";
    for (std::size_t index = 0; index < lca_tails.size(); ++index) {
        if (index) out << ',';
        const auto& tail = lca_tails[index];
        out << "{\"top_percent\":" << (double(tail.basis_points) / 100.0)
            << ",\"threshold_sum\":" << tail.threshold_sum
            << ",\"threshold_mean\":" << (double(tail.threshold_sum) / 32.0)
            << ",\"outcome\":";
        WriteCalibrationOutcomeJson(out, tail.outcome,
                                    calibration.lca_calibration_eligible_moves);
        out << '}';
    }
    out << "]\n  },\n  \"cross\":{\n    \"all\":[";
    for (std::size_t index = 0; index < kCalibrationCrossThresholdCount; ++index) {
        if (index) out << ',';
        out << "{\"threshold\":" << unsigned(kCalibrationCrossThresholds[index])
            << ",\"outcome\":";
        WriteCalibrationOutcomeJson(out, calibration.cross_all_cohorts[index],
                                    calibration.signal_valid_lmr_moves);
        out << '}';
    }
    out << "],\n    \"eligible\":[";
    for (std::size_t index = 0; index < kCalibrationCrossThresholdCount; ++index) {
        if (index) out << ',';
        out << "{\"threshold\":" << unsigned(kCalibrationCrossThresholds[index])
            << ",\"outcome\":";
        WriteCalibrationOutcomeJson(out, cross_eligible[index],
                                    cross_eligible_population.moves);
        out << '}';
    }
    out << "]\n  },\n  \"deployment\":{"
        << "\"router_actual_plus_one\":" << calibration.router_actual_adjusted_moves
        << ",\"router_actual_plus_one_rate_percent\":"
        << Percent(calibration.router_actual_adjusted_moves,
                   calibration.signal_valid_lmr_moves)
        << ",\"lca_calibrated_top1_candidate\":" << lca_tails[0].outcome.moves
        << ",\"lca_calibrated_top1_candidate_rate_percent\":"
        << Percent(lca_tails[0].outcome.moves, calibration.signal_valid_lmr_moves)
        << ",\"cross_calibrated_127_candidate\":"
        << cross_eligible[kCalibrationCrossThresholdCount - 1].moves
        << ",\"cross_calibrated_127_candidate_rate_percent\":"
        << Percent(cross_eligible[kCalibrationCrossThresholdCount - 1].moves,
                   calibration.signal_valid_lmr_moves)
        << "},\n  \"overlap\":[";
    static constexpr std::array<const char*, 8> overlap_labels = {
      "none", "Router-only", "LCA-only", "Router+LCA",
      "Cross-only", "Router+Cross", "LCA+Cross", "Router+LCA+Cross"};
    for (std::size_t mask = 0; mask < overlap.size(); ++mask) {
        if (mask) out << ',';
        out << "{\"group\":\"" << overlap_labels[mask] << "\",\"outcome\":";
        WriteCalibrationOutcomeJson(out, overlap[mask],
                                    calibration.signal_valid_lmr_moves);
        out << '}';
    }
    out << "]\n}\n";
}

}  // namespace YaneuraOu::Search::NnueSignalLog

#endif  // defined(ENABLE_NNUE_SIGNAL_LOG)
#endif  // YANEURAOU_NNUE_SIGNAL_LOGGER_H_INCLUDED
