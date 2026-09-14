#ifndef YANEURAOU_NNUE_UNCERTAINTY_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_UNCERTAINTY_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_UNCERTAINTY_SIGNAL)

#include "../../eval/nnue/nnue_signal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <vector>

namespace YaneuraOu::Search::NnueUncertaintyLog {

constexpr std::size_t kQ8Values = 256;
constexpr std::size_t kBuckets = 12;
constexpr std::size_t kPlyGroups = 5;
constexpr std::size_t kOtherSignals = 3;
constexpr std::size_t kOverlapGroups = 8;

struct NodeOutcome {
    std::uint64_t count = 0;
    std::uint64_t abs_error_sum = 0;
    std::uint64_t fail_high = 0;
    std::uint64_t fail_low = 0;
    std::uint64_t futility_pruned = 0;
};

struct LmrOutcome {
    std::uint64_t count = 0;
    std::uint64_t reduced_fail_high = 0;
    std::uint64_t research = 0;
    std::uint64_t cutoff = 0;
};

struct Correlation {
    std::uint64_t count = 0;
    long double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
};

struct RfpShadow {
    std::uint64_t samples = 0;
    std::uint64_t completed = 0;
    std::uint64_t wrong = 0;
    std::int64_t result_minus_beta_sum = 0;
    std::int64_t result_minus_static_sum = 0;
};

struct FutilityShadow {
    std::uint64_t conditions = 0;
    std::uint64_t samples = 0;
    std::uint64_t completed = 0;
    std::uint64_t wrong = 0;
    std::int64_t result_minus_alpha_sum = 0;
    std::int64_t result_minus_static_sum = 0;
};

struct Stats {
    std::array<std::uint64_t, kQ8Values> fresh_distribution{};
    std::array<NodeOutcome, kQ8Values> node{};
    std::array<LmrOutcome, kQ8Values> lmr{};
    std::array<LmrOutcome, kQ8Values> uncovered_lmr{};
    std::array<std::array<LmrOutcome, kQ8Values>, kBuckets> by_bucket{};
    std::array<std::array<LmrOutcome, kQ8Values>, kPlyGroups> by_ply{};
    std::array<LmrOutcome, kOverlapGroups> overlap{};
    std::array<Correlation, kOtherSignals> pearson{};
    // 256x256 histograms permit a deterministic discrete-rank Spearman
    // without retaining individual search moves.
    std::array<std::array<std::uint64_t, kQ8Values * kQ8Values>,
               kOtherSignals> rank_joint{};
    std::array<RfpShadow, kQ8Values> rfp{};
    std::array<FutilityShadow, kQ8Values> futility_shadow{};
    std::uint64_t futility_shadow_sequence = 0;
};

inline std::mutex g_mutex;
inline std::vector<Stats*> g_stats;

inline Stats& Local() {
    thread_local Stats* value = [] {
        auto* result = new Stats();
        std::lock_guard<std::mutex> lock(g_mutex);
        g_stats.push_back(result);
        return result;
    }();
    return *value;
}

inline std::size_t PlyGroup(const int ply) {
    if (ply < 32) return 0;
    if (ply < 64) return 1;
    if (ply < 96) return 2;
    if (ply < 128) return 3;
    return 4;
}

inline std::uint8_t RankBin(const Eval::NNUE::NnueSignalSnapshot& signal,
                            const std::size_t kind) {
    if (kind == 0) {
        const double value = std::log2(std::max(0, signal.router_margin) + 1.0) * 16.0;
        return static_cast<std::uint8_t>(std::clamp<int>(int(value), 0, 255));
    }
    if (kind == 1)
        return static_cast<std::uint8_t>(std::clamp(
          signal.lca_abs_delta_sum * 255 / 4064, 0, 255));
    return static_cast<std::uint8_t>(std::min<int>(signal.cross_abs_max * 2, 255));
}

inline double OtherValue(const Eval::NNUE::NnueSignalSnapshot& signal,
                         const std::size_t kind) {
    if (kind == 0) return signal.router_margin;
    if (kind == 1) return signal.lca_abs_delta_sum;
    return signal.cross_abs_max;
}

inline void AddCorrelation(Correlation& value, const double x, const double y) {
    ++value.count;
    value.sx += x; value.sy += y;
    value.sxx += x * x; value.syy += y * y; value.sxy += x * y;
}

inline void RecordFresh(const Eval::NNUE::NnueSignalSnapshot& signal) {
    ++Local().fresh_distribution[signal.uncertainty_q8];
}

inline void RecordNode(const Eval::NNUE::NnueSignalSnapshot& signal,
                       const std::uint32_t abs_error, const int result,
                       const int alpha, const int beta,
                       const bool futility_pruned) {
    auto& value = Local().node[signal.uncertainty_q8];
    ++value.count;
    value.abs_error_sum += abs_error;
    value.fail_high += result >= beta;
    value.fail_low += result <= alpha;
    value.futility_pruned += futility_pruned;
}

inline void RecordLmr(const Eval::NNUE::NnueSignalSnapshot& signal,
                      const int ply, const bool reduced_fail_high,
                      const bool researched, const bool cutoff,
                      const bool router_detected, const bool lca_detected,
                      const bool cross_detected) {
    auto add = [&](LmrOutcome& value) {
        ++value.count;
        value.reduced_fail_high += reduced_fail_high;
        value.research += researched;
        value.cutoff += cutoff;
    };
    auto& stats = Local();
    const auto q = signal.uncertainty_q8;
    add(stats.lmr[q]);
    const auto bucket = static_cast<std::size_t>(std::clamp(
      signal.selected_bucket, 0, static_cast<int>(kBuckets - 1)));
    add(stats.by_bucket[bucket][q]);
    add(stats.by_ply[PlyGroup(ply)][q]);
    if (!router_detected && !lca_detected && !cross_detected)
        add(stats.uncovered_lmr[q]);
    const std::size_t flags = (router_detected ? 1U : 0U)
                            | (lca_detected ? 2U : 0U)
                            | (cross_detected ? 4U : 0U);
    add(stats.overlap[flags]);
    for (std::size_t kind = 0; kind < kOtherSignals; ++kind) {
        AddCorrelation(stats.pearson[kind], q, OtherValue(signal, kind));
        const auto other = RankBin(signal, kind);
        ++stats.rank_joint[kind][std::size_t(q) * kQ8Values + other];
    }
}

inline void RecordRfpSelected(const std::uint8_t q) {
    ++Local().rfp[q].samples;
}

inline void RecordRfpOutcome(const std::uint8_t q, const int result,
                             const int beta, const int static_eval) {
    auto& value = Local().rfp[q];
    ++value.completed;
    value.wrong += result < beta;
    value.result_minus_beta_sum += std::int64_t(result) - beta;
    value.result_minus_static_sum += std::int64_t(result) - static_eval;
}

inline bool SelectForwardFutilityShadowSample(const std::uint8_t q) {
    auto& stats = Local();
    auto& value = stats.futility_shadow[q];
    ++value.conditions;
    constexpr std::uint64_t kSampleMask = 255;
    const bool selected = (stats.futility_shadow_sequence++ & kSampleMask) == 0;
    value.samples += selected;
    return selected;
}

inline void RecordForwardFutilityShadowOutcome(
  const std::uint8_t q, const int result, const int alpha,
  const int static_eval) {
    auto& value = Local().futility_shadow[q];
    ++value.completed;
    value.wrong += result > alpha;
    value.result_minus_alpha_sum += std::int64_t(result) - alpha;
    value.result_minus_static_sum += std::int64_t(result) - static_eval;
}

inline void Add(NodeOutcome& dst, const NodeOutcome& src) {
    dst.count += src.count; dst.abs_error_sum += src.abs_error_sum;
    dst.fail_high += src.fail_high; dst.fail_low += src.fail_low;
    dst.futility_pruned += src.futility_pruned;
}
inline void Add(LmrOutcome& dst, const LmrOutcome& src) {
    dst.count += src.count; dst.reduced_fail_high += src.reduced_fail_high;
    dst.research += src.research; dst.cutoff += src.cutoff;
}
inline void Add(Correlation& dst, const Correlation& src) {
    dst.count += src.count; dst.sx += src.sx; dst.sy += src.sy;
    dst.sxx += src.sxx; dst.syy += src.syy; dst.sxy += src.sxy;
}
inline void Add(RfpShadow& dst, const RfpShadow& src) {
    dst.samples += src.samples; dst.completed += src.completed;
    dst.wrong += src.wrong;
    dst.result_minus_beta_sum += src.result_minus_beta_sum;
    dst.result_minus_static_sum += src.result_minus_static_sum;
}
inline void Add(FutilityShadow& dst, const FutilityShadow& src) {
    dst.conditions += src.conditions; dst.samples += src.samples;
    dst.completed += src.completed; dst.wrong += src.wrong;
    dst.result_minus_alpha_sum += src.result_minus_alpha_sum;
    dst.result_minus_static_sum += src.result_minus_static_sum;
}

inline void Reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto* value : g_stats) {
        value->~Stats();
        ::new (static_cast<void*>(value)) Stats();
    }
}

inline std::unique_ptr<Stats> Snapshot() {
    auto result = std::make_unique<Stats>();
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto* src : g_stats) {
        for (std::size_t q = 0; q < kQ8Values; ++q) {
            result->fresh_distribution[q] += src->fresh_distribution[q];
            Add(result->node[q], src->node[q]);
            Add(result->lmr[q], src->lmr[q]);
            Add(result->uncovered_lmr[q], src->uncovered_lmr[q]);
            result->rfp[q].samples += src->rfp[q].samples;
            result->rfp[q].completed += src->rfp[q].completed;
            result->rfp[q].wrong += src->rfp[q].wrong;
            result->rfp[q].result_minus_beta_sum += src->rfp[q].result_minus_beta_sum;
            result->rfp[q].result_minus_static_sum += src->rfp[q].result_minus_static_sum;
            Add(result->futility_shadow[q], src->futility_shadow[q]);
            for (std::size_t b = 0; b < kBuckets; ++b)
                Add(result->by_bucket[b][q], src->by_bucket[b][q]);
            for (std::size_t p = 0; p < kPlyGroups; ++p)
                Add(result->by_ply[p][q], src->by_ply[p][q]);
        }
        for (std::size_t i = 0; i < kOverlapGroups; ++i)
            Add(result->overlap[i], src->overlap[i]);
        for (std::size_t kind = 0; kind < kOtherSignals; ++kind) {
            Add(result->pearson[kind], src->pearson[kind]);
            for (std::size_t i = 0; i < kQ8Values * kQ8Values; ++i)
                result->rank_joint[kind][i] += src->rank_joint[kind][i];
        }
    }
    return result;
}

inline double Percent(const std::uint64_t n, const std::uint64_t d) {
    return d ? 100.0 * double(n) / double(d) : 0.0;
}

inline double Pearson(const Correlation& value) {
    if (value.count < 2) return 0.0;
    const long double n = value.count;
    const long double numerator = n * value.sxy - value.sx * value.sy;
    const long double dx = n * value.sxx - value.sx * value.sx;
    const long double dy = n * value.syy - value.sy * value.sy;
    return dx > 0 && dy > 0 ? double(numerator / std::sqrt(dx * dy)) : 0.0;
}

inline double DiscreteSpearman(
  const std::array<std::uint64_t, kQ8Values * kQ8Values>& joint) {
    std::array<std::uint64_t, kQ8Values> mx{}, my{};
    std::uint64_t total = 0;
    for (std::size_t x = 0; x < kQ8Values; ++x)
        for (std::size_t y = 0; y < kQ8Values; ++y) {
            const auto n = joint[x * kQ8Values + y];
            mx[x] += n; my[y] += n; total += n;
        }
    if (total < 2) return 0.0;
    std::array<double, kQ8Values> rx{}, ry{};
    std::uint64_t before = 0;
    for (std::size_t i = 0; i < kQ8Values; ++i) {
        rx[i] = double(before) + (double(mx[i]) + 1.0) * 0.5;
        before += mx[i];
    }
    before = 0;
    for (std::size_t i = 0; i < kQ8Values; ++i) {
        ry[i] = double(before) + (double(my[i]) + 1.0) * 0.5;
        before += my[i];
    }
    Correlation ranks{};
    for (std::size_t x = 0; x < kQ8Values; ++x)
        for (std::size_t y = 0; y < kQ8Values; ++y) {
            const auto count = joint[x * kQ8Values + y];
            ranks.count += count;
            ranks.sx += count * rx[x];
            ranks.sy += count * ry[y];
            ranks.sxx += count * rx[x] * rx[x];
            ranks.syy += count * ry[y] * ry[y];
            ranks.sxy += count * rx[x] * ry[y];
        }
    return Pearson(ranks);
}

template<typename Outcome>
inline Outcome SumTail(const std::array<Outcome, kQ8Values>& values,
                       const std::size_t threshold) {
    Outcome result{};
    for (std::size_t q = threshold; q < kQ8Values; ++q)
        Add(result, values[q]);
    return result;
}

inline std::size_t TailThreshold(const std::array<LmrOutcome, kQ8Values>& values,
                                 const unsigned basis_points) {
    std::uint64_t total = 0;
    for (const auto& value : values) total += value.count;
    const auto wanted = (total * basis_points + 9999) / 10000;
    std::uint64_t accumulated = 0;
    for (std::size_t q = kQ8Values; q-- > 0;) {
        accumulated += values[q].count;
        if (accumulated >= wanted) return q;
    }
    return 0;
}

inline void PrintLmr(std::ostream& out, const LmrOutcome& value) {
    out << " count=" << value.count
        << " FH=" << Percent(value.reduced_fail_high, value.count) << '%'
        << " re-search=" << Percent(value.research, value.count) << '%'
        << " cutoff=" << Percent(value.cutoff, value.count) << '%';
}

inline void Report(std::ostream& out) {
    const auto storage = Snapshot();
    const auto& stats = *storage;
    out << "[teacher-disagreement uncertainty signal]\n"
        << "  representation: uint8 [0,255], bucket-specific Linear(64,1)+sigmoid\n";
    std::uint64_t fresh = 0;
    for (const auto n : stats.fresh_distribution) fresh += n;
    out << "  fresh distribution count: " << fresh << '\n';
    for (std::size_t q = 0; q < kQ8Values; ++q)
        if (stats.fresh_distribution[q])
            out << "    q=" << q << " count=" << stats.fresh_distribution[q]
                << " rate=" << Percent(stats.fresh_distribution[q], fresh) << "%\n";

    static constexpr std::array<unsigned, 4> tails = {100, 500, 1000, 2000};
    out << "[uncertainty high-tail outcomes]\n";
    for (const auto bp : tails) {
        const auto threshold = TailThreshold(stats.lmr, bp);
        const auto lmr = SumTail(stats.lmr, threshold);
        const auto uncovered = SumTail(stats.uncovered_lmr, threshold);
        const auto node = SumTail(stats.node, threshold);
        const auto rfp = SumTail(stats.rfp, threshold);
        const auto futility_shadow = SumTail(stats.futility_shadow, threshold);
        out << "  top " << double(bp) / 100.0 << "% q>=" << threshold;
        PrintLmr(out, lmr);
        out << " mean|search-static|="
            << (node.count ? double(node.abs_error_sum) / node.count : 0.0)
            << " node-FH=" << Percent(node.fail_high, node.count) << '%'
            << " node-FL=" << Percent(node.fail_low, node.count) << '%'
            << " futility-observed=" << Percent(node.futility_pruned, node.count) << "%\n"
            << "    uncovered by Router/LCA/Cross detection:";
        PrintLmr(out, uncovered);
        out << "\n    RFP shadow samples=" << rfp.completed
            << " wrong=" << Percent(rfp.wrong, rfp.completed) << '%'
            << " mean(result-beta)="
            << (rfp.completed
                  ? double(rfp.result_minus_beta_sum) / rfp.completed : 0.0)
            << " mean(result-static)="
            << (rfp.completed
                  ? double(rfp.result_minus_static_sum) / rfp.completed : 0.0)
            << "\n    forward-futility shadow samples="
            << futility_shadow.completed
            << " wrong=" << Percent(futility_shadow.wrong,
                                      futility_shadow.completed) << '%'
            << " mean(result-alpha)="
            << (futility_shadow.completed
                  ? double(futility_shadow.result_minus_alpha_sum)
                      / futility_shadow.completed : 0.0)
            << " mean(result-static)="
            << (futility_shadow.completed
                  ? double(futility_shadow.result_minus_static_sum)
                      / futility_shadow.completed : 0.0)
            << '\n';
    }

    static constexpr std::array<const char*, kOtherSignals> names = {
      "router_margin", "lca_abs_delta_sum", "cross_abs_max"};
    out << "[uncertainty independence]\n";
    for (std::size_t kind = 0; kind < kOtherSignals; ++kind)
        out << "  " << names[kind]
            << " Pearson=" << Pearson(stats.pearson[kind])
            << " binned-Spearman=" << DiscreteSpearman(stats.rank_joint[kind]) << '\n';
    out << "  overlap flags: bit0=Router predicate, bit1=LCA predicate, bit2=Cross predicate\n";
    for (std::size_t flags = 0; flags < kOverlapGroups; ++flags) {
        out << "    flags=" << flags;
        PrintLmr(out, stats.overlap[flags]);
        out << '\n';
    }

    out << "[uncertainty LMR by bucket]\n";
    for (std::size_t bucket = 0; bucket < kBuckets; ++bucket) {
        const auto threshold = TailThreshold(stats.by_bucket[bucket], 1000);
        out << "  B" << std::setw(2) << std::setfill('0') << bucket
            << std::setfill(' ') << " top10 q>=" << threshold;
        PrintLmr(out, SumTail(stats.by_bucket[bucket], threshold));
        out << '\n';
    }
    static constexpr std::array<const char*, kPlyGroups> ply_names = {
      "0-31", "32-63", "64-95", "96-127", "128+"};
    out << "[uncertainty LMR by game ply]\n";
    for (std::size_t ply = 0; ply < kPlyGroups; ++ply) {
        const auto threshold = TailThreshold(stats.by_ply[ply], 1000);
        out << "  " << ply_names[ply] << " top10 q>=" << threshold;
        PrintLmr(out, SumTail(stats.by_ply[ply], threshold));
        out << '\n';
    }

    RfpShadow rfp{};
    for (const auto& value : stats.rfp) {
        rfp.samples += value.samples; rfp.completed += value.completed;
        rfp.wrong += value.wrong;
        rfp.result_minus_beta_sum += value.result_minus_beta_sum;
        rfp.result_minus_static_sum += value.result_minus_static_sum;
    }
    out << "[uncertainty reverse-futility shadow]\n"
        << "  samples/completed=" << rfp.samples << '/' << rfp.completed
        << " wrong=" << Percent(rfp.wrong, rfp.completed) << '%'
        << " mean(result-beta)="
        << (rfp.completed ? double(rfp.result_minus_beta_sum) / rfp.completed : 0.0)
        << " mean(result-static)="
        << (rfp.completed ? double(rfp.result_minus_static_sum) / rfp.completed : 0.0)
        << '\n';
    FutilityShadow futility_shadow{};
    for (const auto& value : stats.futility_shadow)
        Add(futility_shadow, value);
    out << "[uncertainty forward-futility shadow]\n"
        << "  conditions/samples/completed=" << futility_shadow.conditions
        << '/' << futility_shadow.samples << '/' << futility_shadow.completed
        << " wrong=" << Percent(futility_shadow.wrong,
                                  futility_shadow.completed) << '%'
        << " mean(result-alpha)="
        << (futility_shadow.completed
              ? double(futility_shadow.result_minus_alpha_sum)
                  / futility_shadow.completed : 0.0)
        << " mean(result-static)="
        << (futility_shadow.completed
              ? double(futility_shadow.result_minus_static_sum)
                  / futility_shadow.completed : 0.0)
        << '\n';
}

} // namespace YaneuraOu::Search::NnueUncertaintyLog

#endif
#endif
