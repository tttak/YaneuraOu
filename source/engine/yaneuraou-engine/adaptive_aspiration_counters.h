#pragma once

#if defined(ENABLE_NNUE_ADAPTIVE_ASPIRATION_COUNTERS)

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>

namespace YaneuraOu::Search::AdaptiveAspirationCounters {

inline std::atomic<std::uint64_t> total_events{0};
inline std::atomic<std::uint64_t> active_events{0};
inline std::atomic<std::uint64_t> triggered_events{0};
inline std::atomic<std::uint64_t> total_fail_low{0};
inline std::atomic<std::uint64_t> total_fail_high{0};
inline std::atomic<std::uint64_t> total_researches{0};
inline std::atomic<std::uint64_t> total_iteration_nodes{0};
inline std::atomic<std::uint64_t> total_extra_nodes{0};
inline std::atomic<std::uint64_t> triggered_fail_low{0};
inline std::atomic<std::uint64_t> triggered_fail_high{0};
inline std::atomic<std::uint64_t> triggered_researches{0};
inline std::atomic<std::uint64_t> triggered_iteration_nodes{0};
inline std::atomic<std::uint64_t> triggered_extra_nodes{0};
inline std::array<std::atomic<std::uint64_t>, 128> triggered_by_depth{};

inline void Reset() {
    total_events = active_events = triggered_events = 0;
    total_fail_low = total_fail_high = total_researches = 0;
    total_iteration_nodes = total_extra_nodes = 0;
    triggered_fail_low = triggered_fail_high = triggered_researches = 0;
    triggered_iteration_nodes = triggered_extra_nodes = 0;
    for (auto& value : triggered_by_depth)
        value = 0;
}

inline void Begin(int depth, bool active, bool triggered) {
    ++total_events;
    if (active)
        ++active_events;
    if (triggered) {
        ++triggered_events;
        const auto index = static_cast<std::size_t>(depth < 0 ? 0 : depth > 127 ? 127 : depth);
        ++triggered_by_depth[index];
    }
}

inline void End(bool triggered, int failLow, int failHigh, int researches,
                std::uint64_t iterationNodes, std::uint64_t extraNodes) {
    total_fail_low += static_cast<std::uint64_t>(failLow);
    total_fail_high += static_cast<std::uint64_t>(failHigh);
    total_researches += static_cast<std::uint64_t>(researches);
    total_iteration_nodes += iterationNodes;
    total_extra_nodes += extraNodes;
    if (triggered) {
        triggered_fail_low += static_cast<std::uint64_t>(failLow);
        triggered_fail_high += static_cast<std::uint64_t>(failHigh);
        triggered_researches += static_cast<std::uint64_t>(researches);
        triggered_iteration_nodes += iterationNodes;
        triggered_extra_nodes += extraNodes;
    }
}

inline void Report(std::ostream& os) {
    os << "adaptive aspiration counters begin\n"
       << "total_events " << total_events.load() << '\n'
       << "active_events " << active_events.load() << '\n'
       << "triggered_events " << triggered_events.load() << '\n'
       << "total_fail_low " << total_fail_low.load() << '\n'
       << "total_fail_high " << total_fail_high.load() << '\n'
       << "total_researches " << total_researches.load() << '\n'
       << "total_iteration_nodes " << total_iteration_nodes.load() << '\n'
       << "total_extra_nodes " << total_extra_nodes.load() << '\n'
       << "triggered_fail_low " << triggered_fail_low.load() << '\n'
       << "triggered_fail_high " << triggered_fail_high.load() << '\n'
       << "triggered_researches " << triggered_researches.load() << '\n'
       << "triggered_iteration_nodes " << triggered_iteration_nodes.load() << '\n'
       << "triggered_extra_nodes " << triggered_extra_nodes.load() << '\n';
    for (std::size_t depth = 0; depth < triggered_by_depth.size(); ++depth)
        if (const auto count = triggered_by_depth[depth].load())
            os << "triggered_depth " << depth << ' ' << count << '\n';
    os << "adaptive aspiration counters end\n";
}

}  // namespace YaneuraOu::Search::AdaptiveAspirationCounters

#endif
