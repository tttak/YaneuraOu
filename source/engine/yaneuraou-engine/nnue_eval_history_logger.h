// Diagnostic-only correlation between static-eval history and search outcomes.
#ifndef YANEURAOU_NNUE_EVAL_HISTORY_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_EVAL_HISTORY_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_EVAL_HISTORY_DIAGNOSTIC)

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>

namespace YaneuraOu::Search::NnueEvalHistoryLog {

enum class Outcome : std::size_t { LmrResearch, RfpMiscut, FutilityMiscut, Count };
enum class Signal : std::size_t {
    ParentAbsDelta,
    GrandparentAbsDelta,
    GreatGrandparentAbsDelta,
    MaxAbsDelta,
    ParentSignFlip,
    AnySignFlip,
    Oscillation2,
    Oscillation3,
    Count
};

constexpr std::array<const char*, static_cast<std::size_t>(Outcome::Count)>
  kOutcomeNames = {"lmr_research", "rfp_shadow_miscut", "futility_shadow_miscut"};
constexpr std::array<const char*, static_cast<std::size_t>(Signal::Count)>
  kSignalNames = {"parent_abs_delta", "grandparent_abs_delta",
                  "great_grandparent_abs_delta", "max_abs_delta",
                  "parent_sign_flip", "any_sign_flip", "oscillation2",
                  "oscillation3"};
constexpr std::array<int, 10> kMagnitudeEdges = {
  0, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  std::numeric_limits<int>::max()};
constexpr std::size_t kMagnitudeBins = kMagnitudeEdges.size() - 1;
constexpr std::size_t kBooleanBins = 2;
constexpr std::size_t kMaxBins = kMagnitudeBins;

struct Snapshot {
    std::array<int, static_cast<std::size_t>(Signal::Count)> value{};
    std::array<bool, static_cast<std::size_t>(Signal::Count)> valid{};
};

struct AtomicCell {
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> positive{0};
};

struct AtomicSignalStats {
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> positive{0};
    std::atomic<std::int64_t> sx{0};
    std::atomic<std::uint64_t> sxx{0};
    std::atomic<std::int64_t> sxy{0};
    std::array<AtomicCell, kMaxBins> bins{};
};

struct GlobalStats {
    std::array<std::array<AtomicSignalStats,
      static_cast<std::size_t>(Signal::Count)>,
      static_cast<std::size_t>(Outcome::Count)> outcome{};
};
inline GlobalStats g_stats;

inline bool ValidEval(const int value) {
    return value != static_cast<int>(VALUE_NONE) && std::abs(value) < 30000;
}

inline int Sign(const int value) { return (value > 0) - (value < 0); }

// Stack staticEval values are side-to-move relative.  Parent and great-
// grandparent therefore need a sign flip before comparison with current.
inline Snapshot MakeSnapshot(const int current, const int parent_raw,
                             const int grandparent_raw,
                             const int great_grandparent_raw) {
    Snapshot result;
    if (!ValidEval(current))
        return result;

    const bool p = ValidEval(parent_raw);
    const bool g = ValidEval(grandparent_raw);
    const bool gg = ValidEval(great_grandparent_raw);
    const int parent = -parent_raw;
    const int grandparent = grandparent_raw;
    const int great_grandparent = -great_grandparent_raw;

    const auto set = [&](const Signal signal, const int value, const bool valid) {
        const auto index = static_cast<std::size_t>(signal);
        result.value[index] = value;
        result.valid[index] = valid;
    };
    set(Signal::ParentAbsDelta, std::abs(current - parent), p);
    set(Signal::GrandparentAbsDelta, std::abs(current - grandparent), g);
    set(Signal::GreatGrandparentAbsDelta,
        std::abs(current - great_grandparent), gg);
    set(Signal::MaxAbsDelta,
        std::max({p ? std::abs(current - parent) : 0,
                  g ? std::abs(current - grandparent) : 0,
                  gg ? std::abs(current - great_grandparent) : 0}),
        p || g || gg);

    const bool parent_flip = p && Sign(current) && Sign(parent)
                          && Sign(current) != Sign(parent);
    const bool grand_flip = g && Sign(current) && Sign(grandparent)
                         && Sign(current) != Sign(grandparent);
    const bool great_flip = gg && Sign(current) && Sign(great_grandparent)
                         && Sign(current) != Sign(great_grandparent);
    set(Signal::ParentSignFlip, parent_flip, p);
    set(Signal::AnySignFlip, parent_flip || grand_flip || great_flip, p || g || gg);

    const int d01 = current - parent;
    const int d12 = parent - grandparent;
    const int d23 = grandparent - great_grandparent;
    const bool oscillation2 = p && g && d01 != 0 && d12 != 0
                           && (std::int64_t(d01) * d12 < 0);
    const bool oscillation3 = oscillation2 && gg && d23 != 0
                           && (std::int64_t(d12) * d23 < 0);
    set(Signal::Oscillation2, oscillation2, p && g);
    set(Signal::Oscillation3, oscillation3, p && g && gg);
    return result;
}

inline bool BooleanSignal(const Signal signal) {
    return signal == Signal::ParentSignFlip || signal == Signal::AnySignFlip
        || signal == Signal::Oscillation2 || signal == Signal::Oscillation3;
}

inline std::size_t Bin(const Signal signal, const int value) {
    if (BooleanSignal(signal))
        return value ? 1U : 0U;
    for (std::size_t i = 0; i < kMagnitudeBins; ++i)
        if (value < kMagnitudeEdges[i + 1])
            return i;
    return kMagnitudeBins - 1;
}

inline void Record(const Outcome outcome, const Snapshot& snapshot,
                   const bool positive) {
    const auto oi = static_cast<std::size_t>(outcome);
    for (std::size_t si = 0; si < static_cast<std::size_t>(Signal::Count); ++si) {
        if (!snapshot.valid[si])
            continue;
        const auto signal = static_cast<Signal>(si);
        const int x = snapshot.value[si];
        auto& stats = g_stats.outcome[oi][si];
        stats.count.fetch_add(1, std::memory_order_relaxed);
        stats.positive.fetch_add(positive, std::memory_order_relaxed);
        stats.sx.fetch_add(x, std::memory_order_relaxed);
        stats.sxx.fetch_add(std::uint64_t(x) * std::uint64_t(x),
                            std::memory_order_relaxed);
        stats.sxy.fetch_add(positive ? x : 0, std::memory_order_relaxed);
        auto& bin = stats.bins[Bin(signal, x)];
        bin.count.fetch_add(1, std::memory_order_relaxed);
        bin.positive.fetch_add(positive, std::memory_order_relaxed);
    }
}

inline double Percent(const std::uint64_t part, const std::uint64_t total) {
    return total ? 100.0 * double(part) / double(total) : 0.0;
}

inline double Correlation(const AtomicSignalStats& stats) {
    const double n = double(stats.count.load(std::memory_order_relaxed));
    if (n < 2.0)
        return 0.0;
    const double sx = double(stats.sx.load(std::memory_order_relaxed));
    const double sy = double(stats.positive.load(std::memory_order_relaxed));
    const double sxx = double(stats.sxx.load(std::memory_order_relaxed));
    const double syy = sy;  // binary outcome
    const double sxy = double(stats.sxy.load(std::memory_order_relaxed));
    const double numerator = n * sxy - sx * sy;
    const double denominator = std::sqrt(
      std::max(0.0, (n * sxx - sx * sx) * (n * syy - sy * sy)));
    return denominator ? numerator / denominator : 0.0;
}

inline void Reset() {
    for (auto& outcome : g_stats.outcome)
        for (auto& stats : outcome) {
            stats.count = 0; stats.positive = 0; stats.sx = 0;
            stats.sxx = 0; stats.sxy = 0;
            for (auto& bin : stats.bins) {
                bin.count = 0; bin.positive = 0;
            }
        }
}

inline void Report(std::ostream& out) {
    out << "NNUE static-eval history diagnostic\n"
        << "  perspective: current; parent/great-grandparent signs normalized\n"
        << "  oscillation2: sign(delta current-parent) != sign(delta parent-grandparent)\n";
    for (std::size_t oi = 0; oi < static_cast<std::size_t>(Outcome::Count); ++oi) {
        out << "[" << kOutcomeNames[oi] << "]\n";
        for (std::size_t si = 0; si < static_cast<std::size_t>(Signal::Count); ++si) {
            const auto& stats = g_stats.outcome[oi][si];
            const auto count = stats.count.load(std::memory_order_relaxed);
            const auto positive = stats.positive.load(std::memory_order_relaxed);
            out << "  " << std::left << std::setw(31) << kSignalNames[si]
                << std::right << " n=" << count
                << " rate=" << std::fixed << std::setprecision(4)
                << Percent(positive, count) << "% corr="
                << std::setprecision(6) << Correlation(stats) << '\n';
        }
    }
}

inline void ReportCsv(std::ostream& out) {
    out << "outcome,signal,bin,lower,upper,count,positive,rate_percent,pearson\n";
    for (std::size_t oi = 0; oi < static_cast<std::size_t>(Outcome::Count); ++oi)
        for (std::size_t si = 0; si < static_cast<std::size_t>(Signal::Count); ++si) {
            const auto signal = static_cast<Signal>(si);
            const auto& stats = g_stats.outcome[oi][si];
            const std::size_t bins = BooleanSignal(signal) ? kBooleanBins : kMagnitudeBins;
            for (std::size_t bi = 0; bi < bins; ++bi) {
                const auto count = stats.bins[bi].count.load(std::memory_order_relaxed);
                const auto positive = stats.bins[bi].positive.load(std::memory_order_relaxed);
                const int lower = BooleanSignal(signal) ? int(bi) : kMagnitudeEdges[bi];
                const int upper = BooleanSignal(signal) ? int(bi + 1) : kMagnitudeEdges[bi + 1];
                out << kOutcomeNames[oi] << ',' << kSignalNames[si] << ',' << bi
                    << ',' << lower << ',' << upper << ',' << count << ',' << positive
                    << ',' << std::fixed << std::setprecision(6)
                    << Percent(positive, count) << ',' << Correlation(stats) << '\n';
            }
        }
}

}  // namespace YaneuraOu::Search::NnueEvalHistoryLog

#endif  // ENABLE_NNUE_EVAL_HISTORY_DIAGNOSTIC
#endif  // YANEURAOU_NNUE_EVAL_HISTORY_LOGGER_H_INCLUDED
