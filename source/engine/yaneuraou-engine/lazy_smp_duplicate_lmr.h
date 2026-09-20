#ifndef YANEURAOU_LAZY_SMP_DUPLICATE_LMR_H_INCLUDED
#define YANEURAOU_LAZY_SMP_DUPLICATE_LMR_H_INCLUDED

// Lightweight active-search table for the ABDADA-lite LMR experiment.
// Unlike lazy_smp_duplicate_stats.h this contains no clocks, observer lists,
// subtree accounting, or detailed event logging.  Only deep search frames that
// can possibly match the configured rule are published.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "../../types.h"

namespace YaneuraOu::Search::LazySmpDuplicateLmr {

#ifndef LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD
#define LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD 12
#endif

#ifndef LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF
#define LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF 1
#endif

#ifndef LAZY_SMP_DUP_LMR_APPLY
#define LAZY_SMP_DUP_LMR_APPLY 1
#endif

static_assert(LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD >= 2,
              "LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD must be at least 2");
static_assert(LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF >= 0
                && LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF <= 2,
              "LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF must be in [0, 2]");
static_assert(LAZY_SMP_DUP_LMR_APPLY == 0 || LAZY_SMP_DUP_LMR_APPLY == 1,
              "LAZY_SMP_DUP_LMR_APPLY must be 0 or 1");

constexpr std::size_t SetCount   = 1u << 14;
constexpr std::size_t SetWays    = 4;
constexpr std::size_t MaxThreads = 512;
constexpr std::uint64_t Reserved = ~std::uint64_t(0);
constexpr int RegisterMinDepth =
  LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD - LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF;

struct Slot {
    std::atomic<std::uint64_t> meta{0};
    std::atomic<std::uint64_t> key0{0};
#if HASH_KEY_BITS > 64
    std::atomic<std::uint64_t> key1{0};
#endif
#if HASH_KEY_BITS > 128
    std::atomic<std::uint64_t> key2{0};
    std::atomic<std::uint64_t> key3{0};
#endif
};

struct alignas(64) ThreadStats {
    std::uint64_t searchEntries = 0;
    std::uint64_t deepEntries = 0;
    std::uint64_t nonPvEligibleEntries = 0;
    std::uint64_t duplicateNodes = 0;
    std::uint64_t tableFull = 0;
    std::uint64_t lmrEvents = 0;
    std::uint64_t adjustedMoves = 0;
    std::uint64_t reducedFailHigh = 0;
    std::uint64_t researches = 0;
    std::uint64_t token = 0;
};

inline std::array<Slot, SetCount * SetWays> table{};
inline std::array<ThreadStats, MaxThreads> threadStats{};
inline std::size_t configuredThreads = 0;

inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline std::uint64_t key_part(const Key& key, int part) {
#if HASH_KEY_BITS <= 64
    (void) part;
    return static_cast<Key64>(key);
#elif HASH_KEY_BITS <= 128
    return part == 0 ? key.template extract64<0>() : key.template extract64<1>();
#else
    switch (part) {
    case 0: return key.template extract64<0>();
    case 1: return key.template extract64<1>();
    case 2: return key.template extract64<2>();
    default: return key.template extract64<3>();
    }
#endif
}

inline bool slot_key_equals(const Slot& slot, const Key& key) {
    if (slot.key0.load(std::memory_order_relaxed) != key_part(key, 0))
        return false;
#if HASH_KEY_BITS > 64
    if (slot.key1.load(std::memory_order_relaxed) != key_part(key, 1))
        return false;
#endif
#if HASH_KEY_BITS > 128
    if (slot.key2.load(std::memory_order_relaxed) != key_part(key, 2)
        || slot.key3.load(std::memory_order_relaxed) != key_part(key, 3))
        return false;
#endif
    return true;
}

inline void store_slot_key(Slot& slot, const Key& key) {
    slot.key0.store(key_part(key, 0), std::memory_order_relaxed);
#if HASH_KEY_BITS > 64
    slot.key1.store(key_part(key, 1), std::memory_order_relaxed);
#endif
#if HASH_KEY_BITS > 128
    slot.key2.store(key_part(key, 2), std::memory_order_relaxed);
    slot.key3.store(key_part(key, 3), std::memory_order_relaxed);
#endif
}

// meta layout: token[63:18], depth[17:10], owner+1[9:0].
inline std::uint64_t pack_meta(std::uint64_t token, int depth, std::size_t owner) {
    const auto packedDepth = static_cast<std::uint64_t>(std::clamp(depth, 0, 255));
    return ((token & ((std::uint64_t(1) << 46) - 1)) << 18)
         | (packedDepth << 10) | static_cast<std::uint64_t>(owner + 1);
}

inline std::size_t meta_owner(std::uint64_t meta) {
    return static_cast<std::size_t>((meta & 0x3ffu) - 1);
}

inline int meta_depth(std::uint64_t meta) {
    return static_cast<int>((meta >> 10) & 0xffu);
}

inline void reset(std::size_t threadCount) {
    configuredThreads = std::min(threadCount, MaxThreads);
    for (auto& slot : table)
        slot.meta.store(0, std::memory_order_relaxed);
    for (auto& stats : threadStats)
        stats = ThreadStats{};
}

class ActiveGuard {
   public:
    ActiveGuard(const Key& key, int depth, std::size_t threadId, bool nonPv)
      : threadId_(threadId) {
        if (threadId >= MaxThreads)
            return;

        ThreadStats& stats = threadStats[threadId];
        ++stats.searchEntries;
        if (depth < RegisterMinDepth)
            return;
        ++stats.deepEntries;

        const std::uint64_t mixed = mix64(key_part(key, 0));
        Slot* const begin = &table[(mixed & (SetCount - 1)) * SetWays];

        // Publish our frame. Reserved prevents readers from accepting a partly
        // written exact key. Active slots are never overwritten.
        for (std::size_t way = 0; way < SetWays; ++way) {
            std::uint64_t expected = 0;
            if (begin[way].meta.compare_exchange_strong(
                  expected, Reserved, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
                Slot& slot = begin[way];
                store_slot_key(slot, key);
                std::uint64_t token = ++stats.token;
                if (!token)
                    token = ++stats.token;
                meta_ = pack_meta(token, depth, threadId);
                slot.meta.store(meta_, std::memory_order_release);
                slot_ = &slot;
                break;
            }
        }
        if (!slot_)
            ++stats.tableFull;

        if (!nonPv || depth < LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD)
            return;

        ++stats.nonPvEligibleEntries;
        for (std::size_t way = 0; way < SetWays; ++way) {
            Slot& candidate = begin[way];
            const std::uint64_t candidateMeta =
              candidate.meta.load(std::memory_order_acquire);
            if (!candidateMeta || candidateMeta == Reserved || candidateMeta == meta_)
                continue;
            if (meta_owner(candidateMeta) == threadId || !slot_key_equals(candidate, key))
                continue;
            // A slot may have been cleared and reused while its key words were
            // read. Accept it only if the complete owner/depth/token metadata is
            // still the same; this prevents a reuse race from becoming a false
            // same-key hit.
            if (candidate.meta.load(std::memory_order_acquire) != candidateMeta)
                continue;
            const int difference = std::abs(depth - meta_depth(candidateMeta));
            if (difference <= LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF) {
                shouldReduce_ = true;
                ++stats.duplicateNodes;
                break;
            }
        }
    }

    ActiveGuard(const ActiveGuard&) = delete;
    ActiveGuard& operator=(const ActiveGuard&) = delete;

    ~ActiveGuard() {
        if (!slot_)
            return;
        std::uint64_t expected = meta_;
        slot_->meta.compare_exchange_strong(expected, 0, std::memory_order_release,
                                             std::memory_order_relaxed);
    }

    bool should_reduce() const { return shouldReduce_; }

   private:
    std::size_t threadId_ = MaxThreads;
    Slot* slot_ = nullptr;
    std::uint64_t meta_ = 0;
    bool shouldReduce_ = false;
};

inline void record_lmr_event(std::size_t threadId) {
    if (threadId < MaxThreads)
        ++threadStats[threadId].lmrEvents;
}
inline void record_adjusted(std::size_t threadId) {
    if (threadId < MaxThreads)
        ++threadStats[threadId].adjustedMoves;
}
inline void record_fail_high(std::size_t threadId) {
    if (threadId < MaxThreads)
        ++threadStats[threadId].reducedFailHigh;
}
inline void record_research(std::size_t threadId) {
    if (threadId < MaxThreads)
        ++threadStats[threadId].researches;
}

inline std::string report(std::uint64_t searchedNodes) {
    ThreadStats total{};
    for (std::size_t i = 0; i < configuredThreads; ++i) {
        const auto& s = threadStats[i];
        total.searchEntries += s.searchEntries;
        total.deepEntries += s.deepEntries;
        total.nonPvEligibleEntries += s.nonPvEligibleEntries;
        total.duplicateNodes += s.duplicateNodes;
        total.tableFull += s.tableFull;
        total.lmrEvents += s.lmrEvents;
        total.adjustedMoves += s.adjustedMoves;
        total.reducedFailHigh += s.reducedFailHigh;
        total.researches += s.researches;
    }
    const auto pct = [](std::uint64_t n, std::uint64_t d) {
        return d ? 100.0 * static_cast<double>(n) / static_cast<double>(d) : 0.0;
    };
    std::ostringstream out;
    out << "[LazySMP ABDADA-lite LMR]\n"
        << "threshold/diff          : " << LAZY_SMP_DUP_LMR_DEPTH_THRESHOLD
        << " / " << LAZY_SMP_DUP_LMR_MAX_DEPTH_DIFF << "\n"
        << "depth adjustment        : " << (LAZY_SMP_DUP_LMR_APPLY ? "ON" : "OFF (table-only)") << "\n"
        << "searched nodes          : " << searchedNodes << "\n"
        << "main search entries     : " << total.searchEntries << "\n"
        << "registered deep entries : " << total.deepEntries << "\n"
        << "eligible NonPV nodes    : " << total.nonPvEligibleEntries << "\n"
        << "same-key active nodes   : " << total.duplicateNodes << " ("
        << std::fixed << std::setprecision(4)
        << pct(total.duplicateNodes, total.nonPvEligibleEntries) << "% eligible, "
        << pct(total.duplicateNodes, total.searchEntries) << "% all)\n"
        << "table-full drops        : " << total.tableFull << "\n"
        << "LMR move events         : " << total.lmrEvents << "\n"
        << "LMR +1 adjusted moves   : " << total.adjustedMoves << " ("
        << pct(total.adjustedMoves, total.lmrEvents) << "% LMR)\n"
        << "adjusted fail-high      : " << total.reducedFailHigh << " ("
        << pct(total.reducedFailHigh, total.adjustedMoves) << "% adjusted)\n"
        << "adjusted re-searches    : " << total.researches << " ("
        << pct(total.researches, total.adjustedMoves) << "% adjusted)";
    return out.str();
}

}  // namespace YaneuraOu::Search::LazySmpDuplicateLmr

#endif
