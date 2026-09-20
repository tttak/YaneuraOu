#ifndef YANEURAOU_LAZY_SMP_DUPLICATE_STATS_H_INCLUDED
#define YANEURAOU_LAZY_SMP_DUPLICATE_STATS_H_INCLUDED

// Diagnostic-only active-search table for measuring Lazy SMP duplication.
// This header is included only when MEASURE_LAZY_SMP_DUPLICATION is defined.

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include "../../types.h"

namespace YaneuraOu::Search::LazySmpDuplicateStats {

#ifndef LAZY_SMP_DUP_SAMPLE_LOG2
// Only sampled entries touch the shared table.  Per-thread total counters still
// count every main-search entry.  Override with 0 for an unsampled measurement.
#define LAZY_SMP_DUP_SAMPLE_LOG2 3
#endif

static_assert(LAZY_SMP_DUP_SAMPLE_LOG2 >= 0 && LAZY_SMP_DUP_SAMPLE_LOG2 <= 16,
              "LAZY_SMP_DUP_SAMPLE_LOG2 must be in [0, 16]");

constexpr std::size_t SetCount  = 1u << 15;
constexpr std::size_t SetWays   = 4;
constexpr std::size_t MaxThreads = 512;
constexpr std::uint64_t Reserved = ~std::uint64_t(0);
constexpr std::uint64_t SampleRate = std::uint64_t(1) << LAZY_SMP_DUP_SAMPLE_LOG2;

struct Slot {
    std::atomic<std::uint64_t> meta{0};
    std::atomic<std::uint64_t> key0{0};
    std::atomic<std::uint64_t> startTick{0};
    std::atomic<std::uint64_t> endedMeta{0};
    std::atomic<std::uint64_t> endTick{0};
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
    std::uint64_t sampledEntries = 0;
    std::uint64_t sameKeyActive = 0;
    std::uint64_t sameDepth = 0;
    std::uint64_t depthDiff1 = 0;
    std::uint64_t depthDiff2 = 0;
    std::uint64_t tableFull = 0;
    std::uint64_t tableFullSameKey = 0;
    std::uint64_t tableFullCollision = 0;
    std::uint64_t token = 0;
    std::uint64_t rawDuplicateSubtreeNodes = 0;
    std::uint64_t sameDepthSubtreeNodes = 0;
    std::uint64_t depthDiff1SubtreeNodes = 0;
    std::uint64_t depthDiff2SubtreeNodes = 0;
    std::uint64_t outermostDuplicateEntries = 0;
    std::uint64_t outermostDuplicateSubtreeNodes = 0;
    std::uint64_t outermostSameDepthEntries = 0;
    std::uint64_t outermostSameDepthSubtreeNodes = 0;
    std::uint64_t outermostDepthDiff1Entries = 0;
    std::uint64_t outermostDepthDiff1SubtreeNodes = 0;
    std::uint64_t duplicateNesting = 0;
    std::array<std::uint64_t, 5> bucketEntries{};
    std::array<std::uint64_t, 5> bucketSampled{};
    std::array<std::uint64_t, 5> bucketDuplicate{};
    std::array<std::uint64_t, 5> bucketRawDuplicateSubtreeNodes{};
    std::array<std::uint64_t, 5> bucketOutermostDuplicateEntries{};
    std::array<std::uint64_t, 5> bucketOutermostDuplicateSubtreeNodes{};
    // Index 0=all, 1=same depth, 2=depth diff<=1, 3=depth diff<=2.
    std::array<std::uint64_t, 4> timedEntries{};
    std::array<std::uint64_t, 4> duplicateLifetimeTicks{};
    std::array<std::uint64_t, 4> concurrentOverlapTicks{};
    std::array<std::uint64_t, 4> deep8TimedEntries{};
    std::array<std::uint64_t, 4> deep8LifetimeTicks{};
    std::array<std::uint64_t, 4> deep8OverlapTicks{};
    std::array<std::uint64_t, 4> deep12TimedEntries{};
    std::array<std::uint64_t, 4> deep12LifetimeTicks{};
    std::array<std::uint64_t, 4> deep12OverlapTicks{};
    std::uint64_t timingLostEntries = 0;
    std::uint64_t tscOrderErrors = 0;
    std::array<std::uint64_t, 5> bucketTimedEntries{};
    std::array<std::uint64_t, 5> bucketDuplicateLifetimeTicks{};
    std::array<std::uint64_t, 5> bucketConcurrentOverlapTicks{};
    std::array<std::uint64_t, 5> bucketOwnerEndedFirst{};
    std::array<std::uint64_t, 5> bucketDuplicateEndedFirst{};
};

inline std::array<Slot, SetCount * SetWays> table{};
inline std::array<ThreadStats, MaxThreads> threadStats{};
inline std::size_t configuredThreads = 0;
inline std::uint64_t calibrationStartTick = 0;
inline std::chrono::steady_clock::time_point calibrationStartWall{};

inline std::uint64_t read_tick() {
#if defined(_MSC_VER) || defined(__i386__) || defined(__x86_64__)
    _mm_lfence();
    const std::uint64_t tick = __rdtsc();
    _mm_lfence();
    return tick;
#else
    return static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

inline bool invariant_tsc_supported() {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    int maxLeaf[4]{};
    __cpuid(maxLeaf, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(maxLeaf[0]) < 0x80000007u)
        return false;
    int leaf[4]{};
    __cpuid(leaf, static_cast<int>(0x80000007u));
    return (leaf[3] & (1 << 8)) != 0;
#elif defined(__i386__) || defined(__x86_64__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000007u)
        return false;
    __get_cpuid(0x80000007u, &eax, &ebx, &ecx, &edx);
    return (edx & (1u << 8)) != 0;
#else
    return false;
#endif
}

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

inline int depth_bucket(int depth) {
    if (depth <= 3)  return 0;
    if (depth <= 7)  return 1;
    if (depth <= 11) return 2;
    if (depth <= 15) return 3;
    return 4;
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

inline void reset(std::size_t threadCount) {
    configuredThreads = std::min(threadCount, MaxThreads);
    for (auto& slot : table) {
        slot.meta.store(0, std::memory_order_relaxed);
        slot.endedMeta.store(0, std::memory_order_relaxed);
        slot.startTick.store(0, std::memory_order_relaxed);
        slot.endTick.store(0, std::memory_order_relaxed);
    }
    for (auto& stats : threadStats)
        stats = ThreadStats{};
    calibrationStartWall = std::chrono::steady_clock::now();
    calibrationStartTick = read_tick();
}

class ActiveGuard {
   public:
    ActiveGuard(const Key& key, int depth, std::size_t threadId,
                const std::atomic<std::uint64_t>& nodeCounter) {
        if (threadId >= MaxThreads)
            return;

        ThreadStats& stats = threadStats[threadId];
        const int bucket = depth_bucket(depth);
        ++stats.searchEntries;
        ++stats.bucketEntries[bucket];

        const std::uint64_t mixed = mix64(key_part(key, 0));
        if ((mixed & (SampleRate - 1)) != 0)
            return;

        ++stats.sampledEntries;
        ++stats.bucketSampled[bucket];

        const std::uint64_t sampleTick = read_tick();

        const std::size_t set = (mixed >> LAZY_SMP_DUP_SAMPLE_LOG2) & (SetCount - 1);
        Slot* const begin = &table[set * SetWays];

        // Publish this frame first.  The reserved marker prevents readers from
        // observing a partially written key.
        for (std::size_t way = 0; way < SetWays; ++way) {
            std::uint64_t empty = 0;
            if (begin[way].meta.compare_exchange_strong(empty, Reserved,
                                                        std::memory_order_acquire,
                                                        std::memory_order_relaxed)) {
                slot_ = &begin[way];
                store_slot_key(*slot_, key);
                slot_->startTick.store(sampleTick, std::memory_order_relaxed);
                const std::uint64_t token = (++stats.token) & ((std::uint64_t(1) << 40) - 1);
                const std::uint64_t owner = (threadId + 1) & 0xffff;
                const std::uint64_t d = std::uint64_t(std::max(0, std::min(depth, 255)));
                meta_ = ((token ? token : 1) << 24) | (d << 16) | owner;
                slot_->meta.store(meta_, std::memory_order_release);
                break;
            }
        }

        if (!slot_)
            ++stats.tableFull;

        bool duplicate = false;
        int minDepthDiff = 256;
        bool timingObservationLost = false;
        for (std::size_t way = 0; way < SetWays; ++way) {
            Slot& candidate = begin[way];
            if (&candidate == slot_)
                continue;

            const std::uint64_t meta1 = candidate.meta.load(std::memory_order_acquire);
            if (meta1 == 0 || meta1 == Reserved)
                continue;
            if (!slot_key_equals(candidate, key))
                continue;
            const std::uint64_t meta2 = candidate.meta.load(std::memory_order_acquire);
            if (meta1 != meta2)
                continue;

            const std::size_t owner = std::size_t((meta1 & 0xffff) - 1);
            if (owner == threadId)
                continue;

            const std::uint64_t ownerStart = candidate.startTick.load(std::memory_order_relaxed);
            const std::uint64_t meta3 = candidate.meta.load(std::memory_order_acquire);
            if (meta3 != meta1) {
                timingObservationLost = true;
                continue;
            }

            duplicate = true;
            const int otherDepth = int((meta1 >> 16) & 0xff);
            minDepthDiff = std::min(minDepthDiff, std::abs(depth - otherDepth));
            if (ownerCount_ < owners_.size())
                owners_[ownerCount_++] = OwnerObservation{&candidate, meta1, ownerStart, otherDepth};
            else
                timingObservationLost = true;
        }

        if (duplicate) {
            ++stats.sameKeyActive;
            ++stats.bucketDuplicate[bucket];
            if (minDepthDiff == 0) ++stats.sameDepth;
            if (minDepthDiff <= 1) ++stats.depthDiff1;
            if (minDepthDiff <= 2) ++stats.depthDiff2;

            // The worker node counter is written only by this worker.  Record
            // it after duplicate classification and read it again on scope
            // exit.  +1 at exit attributes the invocation node itself (the
            // caller increments the engine counter before entering it).
            stats_ = &stats;
            nodeCounter_ = &nodeCounter;
            startNodes_ = nodeCounter.load(std::memory_order_relaxed);
            bucket_ = bucket;
            sameDepthMatch_ = minDepthDiff == 0;
            depthDiff1Match_ = minDepthDiff <= 1;
            depthDiff2Match_ = minDepthDiff <= 2;
            depth_ = depth;
            outermost_ = stats.duplicateNesting++ == 0;
            duplicateStartTick_ = read_tick();
            timingObservationLost_ = timingObservationLost || ownerCount_ == 0;
        }

        if (!slot_) {
            if (duplicate)
                ++stats.tableFullSameKey;
            else
                ++stats.tableFullCollision;
        }
    }

    ActiveGuard(const ActiveGuard&) = delete;
    ActiveGuard& operator=(const ActiveGuard&) = delete;

    ~ActiveGuard() {
        const std::uint64_t endTick = (stats_ || slot_) ? read_tick() : 0;

        // Publish the end timestamp before making the slot available for a
        // later generation.  Observers validate endedMeta against the exact
        // owner/depth/token metadata, so reuse can only cause a lost sample.
        if (slot_) {
            slot_->endTick.store(endTick, std::memory_order_relaxed);
            slot_->endedMeta.store(meta_, std::memory_order_release);
            std::uint64_t expected = meta_;
            slot_->meta.compare_exchange_strong(expected, 0, std::memory_order_release,
                                                std::memory_order_relaxed);
        }

        if (stats_) {
            const std::uint64_t subtreeNodes =
              nodeCounter_->load(std::memory_order_relaxed) - startNodes_ + 1;
            stats_->rawDuplicateSubtreeNodes += subtreeNodes;
            stats_->bucketRawDuplicateSubtreeNodes[bucket_] += subtreeNodes;
            if (sameDepthMatch_)
                stats_->sameDepthSubtreeNodes += subtreeNodes;
            if (depthDiff1Match_)
                stats_->depthDiff1SubtreeNodes += subtreeNodes;
            if (depthDiff2Match_)
                stats_->depthDiff2SubtreeNodes += subtreeNodes;

            --stats_->duplicateNesting;
            if (outermost_) {
                ++stats_->outermostDuplicateEntries;
                stats_->outermostDuplicateSubtreeNodes += subtreeNodes;
                ++stats_->bucketOutermostDuplicateEntries[bucket_];
                stats_->bucketOutermostDuplicateSubtreeNodes[bucket_] += subtreeNodes;
                if (sameDepthMatch_) {
                    ++stats_->outermostSameDepthEntries;
                    stats_->outermostSameDepthSubtreeNodes += subtreeNodes;
                }
                if (depthDiff1Match_) {
                    ++stats_->outermostDepthDiff1Entries;
                    stats_->outermostDepthDiff1SubtreeNodes += subtreeNodes;
                }
            }

            bool timingValid = !timingObservationLost_ && endTick >= duplicateStartTick_;
            std::array<std::uint64_t, 4> latestOwnerEnd{};
            for (std::size_t i = 0; timingValid && i < ownerCount_; ++i) {
                const OwnerObservation& observation = owners_[i];
                if (observation.startTick > duplicateStartTick_) {
                    timingValid = false;
                    ++stats_->tscOrderErrors;
                    break;
                }

                std::uint64_t ownerEnd = 0;
                const std::uint64_t endedMeta =
                  observation.slot->endedMeta.load(std::memory_order_acquire);
                if (endedMeta == observation.meta)
                    ownerEnd = observation.slot->endTick.load(std::memory_order_relaxed);
                else if (observation.slot->meta.load(std::memory_order_acquire)
                         == observation.meta)
                    ownerEnd = endTick;  // Owner is still active; duplicate ended first.
                else {
                    // Retry once in case owner exit was between the two loads.
                    const std::uint64_t retry =
                      observation.slot->endedMeta.load(std::memory_order_acquire);
                    if (retry == observation.meta)
                        ownerEnd = observation.slot->endTick.load(std::memory_order_relaxed);
                    else
                        timingValid = false;  // Slot generation was reused before observation.
                }

                if (!timingValid)
                    break;
                latestOwnerEnd[0] = std::max(latestOwnerEnd[0], ownerEnd);
                const int diff = std::abs(depth_ - observation.depth);
                if (diff == 0) latestOwnerEnd[1] = std::max(latestOwnerEnd[1], ownerEnd);
                if (diff <= 1) latestOwnerEnd[2] = std::max(latestOwnerEnd[2], ownerEnd);
                if (diff <= 2) latestOwnerEnd[3] = std::max(latestOwnerEnd[3], ownerEnd);
            }

            if (timingValid) {
                const std::uint64_t lifetime = endTick - duplicateStartTick_;
                for (int c = 0; c < 4; ++c) {
                    if (!latestOwnerEnd[c])
                        continue;
                    const std::uint64_t overlapEnd = std::min(endTick, latestOwnerEnd[c]);
                    const std::uint64_t overlap =
                      overlapEnd > duplicateStartTick_ ? overlapEnd - duplicateStartTick_ : 0;
                    ++stats_->timedEntries[c];
                    stats_->duplicateLifetimeTicks[c] += lifetime;
                    stats_->concurrentOverlapTicks[c] += overlap;
                    if (bucket_ >= 2) {
                        ++stats_->deep8TimedEntries[c];
                        stats_->deep8LifetimeTicks[c] += lifetime;
                        stats_->deep8OverlapTicks[c] += overlap;
                    }
                    if (bucket_ >= 3) {
                        ++stats_->deep12TimedEntries[c];
                        stats_->deep12LifetimeTicks[c] += lifetime;
                        stats_->deep12OverlapTicks[c] += overlap;
                    }
                }
                ++stats_->bucketTimedEntries[bucket_];
                stats_->bucketDuplicateLifetimeTicks[bucket_] += lifetime;
                const std::uint64_t overlapEnd = std::min(endTick, latestOwnerEnd[0]);
                const std::uint64_t overlap =
                  overlapEnd > duplicateStartTick_ ? overlapEnd - duplicateStartTick_ : 0;
                stats_->bucketConcurrentOverlapTicks[bucket_] += overlap;
                if (latestOwnerEnd[0] < endTick)
                    ++stats_->bucketOwnerEndedFirst[bucket_];
                else
                    ++stats_->bucketDuplicateEndedFirst[bucket_];
            } else
                ++stats_->timingLostEntries;
        }
    }

   private:
    struct OwnerObservation {
        Slot* slot = nullptr;
        std::uint64_t meta = 0;
        std::uint64_t startTick = 0;
        int depth = 0;
    };

    Slot* slot_ = nullptr;
    std::uint64_t meta_ = 0;
    ThreadStats* stats_ = nullptr;
    const std::atomic<std::uint64_t>* nodeCounter_ = nullptr;
    std::uint64_t startNodes_ = 0;
    int bucket_ = 0;
    bool sameDepthMatch_ = false;
    bool depthDiff1Match_ = false;
    bool depthDiff2Match_ = false;
    bool outermost_ = false;
    int depth_ = 0;
    std::uint64_t duplicateStartTick_ = 0;
    std::array<OwnerObservation, SetWays> owners_{};
    std::size_t ownerCount_ = 0;
    bool timingObservationLost_ = false;
};

inline std::string report(std::uint64_t totalSearchNodes) {
    ThreadStats total;
    for (std::size_t i = 0; i < configuredThreads; ++i) {
        const ThreadStats& s = threadStats[i];
        total.searchEntries += s.searchEntries;
        total.sampledEntries += s.sampledEntries;
        total.sameKeyActive += s.sameKeyActive;
        total.sameDepth += s.sameDepth;
        total.depthDiff1 += s.depthDiff1;
        total.depthDiff2 += s.depthDiff2;
        total.tableFull += s.tableFull;
        total.tableFullSameKey += s.tableFullSameKey;
        total.tableFullCollision += s.tableFullCollision;
        total.rawDuplicateSubtreeNodes += s.rawDuplicateSubtreeNodes;
        total.sameDepthSubtreeNodes += s.sameDepthSubtreeNodes;
        total.depthDiff1SubtreeNodes += s.depthDiff1SubtreeNodes;
        total.depthDiff2SubtreeNodes += s.depthDiff2SubtreeNodes;
        total.outermostDuplicateEntries += s.outermostDuplicateEntries;
        total.outermostDuplicateSubtreeNodes += s.outermostDuplicateSubtreeNodes;
        total.outermostSameDepthEntries += s.outermostSameDepthEntries;
        total.outermostSameDepthSubtreeNodes += s.outermostSameDepthSubtreeNodes;
        total.outermostDepthDiff1Entries += s.outermostDepthDiff1Entries;
        total.outermostDepthDiff1SubtreeNodes += s.outermostDepthDiff1SubtreeNodes;
        total.timingLostEntries += s.timingLostEntries;
        total.tscOrderErrors += s.tscOrderErrors;
        for (int c = 0; c < 4; ++c) {
            total.timedEntries[c] += s.timedEntries[c];
            total.duplicateLifetimeTicks[c] += s.duplicateLifetimeTicks[c];
            total.concurrentOverlapTicks[c] += s.concurrentOverlapTicks[c];
            total.deep8TimedEntries[c] += s.deep8TimedEntries[c];
            total.deep8LifetimeTicks[c] += s.deep8LifetimeTicks[c];
            total.deep8OverlapTicks[c] += s.deep8OverlapTicks[c];
            total.deep12TimedEntries[c] += s.deep12TimedEntries[c];
            total.deep12LifetimeTicks[c] += s.deep12LifetimeTicks[c];
            total.deep12OverlapTicks[c] += s.deep12OverlapTicks[c];
        }
        for (int b = 0; b < 5; ++b) {
            total.bucketEntries[b] += s.bucketEntries[b];
            total.bucketSampled[b] += s.bucketSampled[b];
            total.bucketDuplicate[b] += s.bucketDuplicate[b];
            total.bucketRawDuplicateSubtreeNodes[b] += s.bucketRawDuplicateSubtreeNodes[b];
            total.bucketOutermostDuplicateEntries[b] += s.bucketOutermostDuplicateEntries[b];
            total.bucketOutermostDuplicateSubtreeNodes[b] +=
              s.bucketOutermostDuplicateSubtreeNodes[b];
            total.bucketTimedEntries[b] += s.bucketTimedEntries[b];
            total.bucketDuplicateLifetimeTicks[b] += s.bucketDuplicateLifetimeTicks[b];
            total.bucketConcurrentOverlapTicks[b] += s.bucketConcurrentOverlapTicks[b];
            total.bucketOwnerEndedFirst[b] += s.bucketOwnerEndedFirst[b];
            total.bucketDuplicateEndedFirst[b] += s.bucketDuplicateEndedFirst[b];
        }
    }

    const auto pct = [](std::uint64_t n, std::uint64_t d) {
        return d ? 100.0 * double(n) / double(d) : 0.0;
    };
    const auto estimate = [](std::uint64_t n) { return n * SampleRate; };
    const auto average = [](std::uint64_t n, std::uint64_t d) {
        return d ? double(n) / double(d) : 0.0;
    };

    const std::uint64_t calibrationEndTick = read_tick();
    const auto calibrationEndWall = std::chrono::steady_clock::now();
    const double calibrationNs = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          calibrationEndWall - calibrationStartWall)
                                          .count());
    const double nsPerTick = calibrationEndTick > calibrationStartTick
                             ? calibrationNs / double(calibrationEndTick - calibrationStartTick)
                             : 0.0;
    const auto average_us = [&](std::uint64_t ticks, std::uint64_t entries) {
        return average(ticks, entries) * nsPerTick / 1000.0;
    };

    std::ostringstream os;
    os << "[LazySMP Duplicate Stats]\n"
       << "Main search entries       : " << total.searchEntries << "\n"
       << "Sampling                  : 1/" << SampleRate << " (key-deterministic)\n"
       << "Sampled entries           : " << total.sampledEntries << "\n"
       << "Same key active           : " << estimate(total.sameKeyActive) << " ("
       << std::fixed << std::setprecision(3) << pct(total.sameKeyActive, total.sampledEntries)
       << "%, sampled=" << total.sameKeyActive << ")\n"
       << "Same key + same depth     : " << estimate(total.sameDepth) << " ("
       << pct(total.sameDepth, total.sampledEntries) << "%, sampled=" << total.sameDepth << ")\n"
       << "Same key + depth diff<=1  : " << estimate(total.depthDiff1) << " ("
       << pct(total.depthDiff1, total.sampledEntries) << "%, sampled=" << total.depthDiff1 << ")\n"
       << "Same key + depth diff<=2  : " << estimate(total.depthDiff2) << " ("
       << pct(total.depthDiff2, total.sampledEntries) << "%, sampled=" << total.depthDiff2 << ")\n"
       << "Table-full registrations  : " << total.tableFull << " ("
       << pct(total.tableFull, total.sampledEntries) << "% of sampled)\n"
       << "  same-key occupancy      : " << total.tableFullSameKey << "\n"
       << "  hash-set collision      : " << total.tableFullCollision << "\n"
       << "Engine nodes              : " << totalSearchNodes << "\n"
       << "Raw duplicate subtree HT  : " << estimate(total.rawDuplicateSubtreeNodes) << " ("
       << pct(estimate(total.rawDuplicateSubtreeNodes), totalSearchNodes)
       << "%, sampled=" << total.rawDuplicateSubtreeNodes << ", nested double-counted)\n"
       << "Outermost sampled entries : " << total.outermostDuplicateEntries << "\n"
       << "Outermost sampled coverage: " << total.outermostDuplicateSubtreeNodes << " ("
       << pct(total.outermostDuplicateSubtreeNodes, totalSearchNodes)
       << "% of engine nodes; conservative lower bound)\n"
       << "Avg nodes / duplicate     : "
       << average(total.rawDuplicateSubtreeNodes, total.sameKeyActive) << " raw, "
       << average(total.outermostDuplicateSubtreeNodes, total.outermostDuplicateEntries)
       << " outermost\n"
       << "Duplicate subtree classes (estimated nodes / ratio / average):\n"
       << "  same key active         : " << estimate(total.rawDuplicateSubtreeNodes) << " / "
       << pct(estimate(total.rawDuplicateSubtreeNodes), totalSearchNodes) << "% / "
       << average(total.rawDuplicateSubtreeNodes, total.sameKeyActive) << "\n"
       << "  same depth              : " << estimate(total.sameDepthSubtreeNodes) << " / "
       << pct(estimate(total.sameDepthSubtreeNodes), totalSearchNodes) << "% / "
       << average(total.sameDepthSubtreeNodes, total.sameDepth) << "\n"
       << "  depth diff<=1           : " << estimate(total.depthDiff1SubtreeNodes) << " / "
       << pct(estimate(total.depthDiff1SubtreeNodes), totalSearchNodes) << "% / "
       << average(total.depthDiff1SubtreeNodes, total.depthDiff1) << "\n"
       << "Depth buckets (actual entries / sampled / same-key-active / ratio):\n";

    static constexpr const char* labels[5] = {"1-3", "4-7", "8-11", "12-15", "16+"};
    for (int b = 0; b < 5; ++b)
        os << "  " << std::setw(5) << labels[b] << " : " << total.bucketEntries[b] << " / "
           << total.bucketSampled[b] << " / " << total.bucketDuplicate[b] << " / "
           << pct(total.bucketDuplicate[b], total.bucketSampled[b]) << "%\n";
    os << "Duplicate cost buckets (duplicate entries / outermost entries / raw HT nodes / "
          "outermost sampled nodes / coverage / avg raw / avg outermost):\n";
    for (int b = 0; b < 5; ++b)
        os << "  " << std::setw(5) << labels[b] << " : " << total.bucketDuplicate[b] << " / "
           << total.bucketOutermostDuplicateEntries[b] << " / "
           << estimate(total.bucketRawDuplicateSubtreeNodes[b]) << " / "
           << total.bucketOutermostDuplicateSubtreeNodes[b] << " / "
           << pct(total.bucketOutermostDuplicateSubtreeNodes[b], totalSearchNodes)
           << "% / "
           << average(total.bucketRawDuplicateSubtreeNodes[b], total.bucketDuplicate[b]) << " / "
           << average(total.bucketOutermostDuplicateSubtreeNodes[b],
                      total.bucketOutermostDuplicateEntries[b]) << "\n";
    os << "Concurrent overlap timing (TSC, no polling):\n"
       << "  invariant TSC CPUID      : " << (invariant_tsc_supported() ? "yes" : "no") << "\n"
       << "  calibrated TSC GHz      : " << (nsPerTick > 0.0 ? 1.0 / nsPerTick : 0.0) << "\n"
       << "  timing lost entries     : " << total.timingLostEntries << "\n"
       << "  TSC order errors        : " << total.tscOrderErrors << "\n"
       << "Timing classes (entries / avg lifetime us / avg overlap us / overlap ratio):\n";
    static constexpr const char* classLabels[4] = {
      "same key active", "same depth", "depth diff<=1", "depth diff<=2"};
    for (int c = 0; c < 4; ++c)
        os << "  " << std::setw(15) << classLabels[c] << " : " << total.timedEntries[c]
           << " / " << average_us(total.duplicateLifetimeTicks[c], total.timedEntries[c])
           << " / " << average_us(total.concurrentOverlapTicks[c], total.timedEntries[c])
           << " / " << pct(total.concurrentOverlapTicks[c], total.duplicateLifetimeTicks[c])
           << "%\n";
    os << "Focused timing classes (scope / class / entries / avg lifetime us / avg overlap us / "
          "overlap ratio):\n";
    for (int c : {1, 2}) {
        os << "  depth>=8  / " << std::setw(13) << classLabels[c] << " : "
           << total.deep8TimedEntries[c] << " / "
           << average_us(total.deep8LifetimeTicks[c], total.deep8TimedEntries[c]) << " / "
           << average_us(total.deep8OverlapTicks[c], total.deep8TimedEntries[c]) << " / "
           << pct(total.deep8OverlapTicks[c], total.deep8LifetimeTicks[c]) << "%\n";
        os << "  depth>=12 / " << std::setw(13) << classLabels[c] << " : "
           << total.deep12TimedEntries[c] << " / "
           << average_us(total.deep12LifetimeTicks[c], total.deep12TimedEntries[c]) << " / "
           << average_us(total.deep12OverlapTicks[c], total.deep12TimedEntries[c]) << " / "
           << pct(total.deep12OverlapTicks[c], total.deep12LifetimeTicks[c]) << "%\n";
    }
    os << "Timing depth buckets (entries / avg lifetime us / avg overlap us / overlap ratio / "
          "owner-first / duplicate-first):\n";
    for (int b = 0; b < 5; ++b)
        os << "  " << std::setw(5) << labels[b] << " : " << total.bucketTimedEntries[b]
           << " / "
           << average_us(total.bucketDuplicateLifetimeTicks[b], total.bucketTimedEntries[b])
           << " / "
           << average_us(total.bucketConcurrentOverlapTicks[b], total.bucketTimedEntries[b])
           << " / "
           << pct(total.bucketConcurrentOverlapTicks[b], total.bucketDuplicateLifetimeTicks[b])
           << "% / " << pct(total.bucketOwnerEndedFirst[b], total.bucketTimedEntries[b])
           << "% / " << pct(total.bucketDuplicateEndedFirst[b], total.bucketTimedEntries[b])
           << "%\n";
    os << "Scope                     : main search() entries; descendant qsearch nodes included in "
          "subtree cost";
    return os.str();
}

}  // namespace YaneuraOu::Search::LazySmpDuplicateStats

#endif
