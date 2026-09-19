// Experiment 57: diagnostic-only root aspiration-window event logger.
#ifndef YANEURAOU_NNUE_ASPIRATION_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_ASPIRATION_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_ASPIRATION_DIAGNOSTIC)

#include "../../position.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <vector>

namespace YaneuraOu::Search::NnueAspirationLog {

constexpr int InvalidScore = 40000;

struct RootContext {
    std::uint64_t root_id = 0;
    std::uint64_t packed_sfen_hash = 0;
};

struct Event {
    std::uint64_t root_id = 0;
    std::uint64_t packed_sfen_hash = 0;
    int depth = 0;
    int pv_index = 0;
    int previous_iteration_score = InvalidScore;
    int average_score = InvalidScore;
    int score_d1 = InvalidScore;
    int score_d2 = InvalidScore;
    int score_d3 = InvalidScore;
    int initial_alpha = 0;
    int initial_beta = 0;
    int initial_window_width = 0;
    int first_search_score = InvalidScore;
    int final_score = InvalidScore;
    int fail_low_count = 0;
    int fail_high_count = 0;
    int re_search_count = 0;
    std::uint64_t total_nodes = 0;
    std::uint64_t first_search_nodes = 0;
    std::uint64_t extra_nodes = 0;
    int static_eval = InvalidScore;
    int material = 0;
    int bucket = -1;
    int router_margin = -1;
    int router_top1_logit = 0;
    int router_top2_logit = 0;
    int lca_abs_delta_sum = -1;
    int lca_abs_delta_max = -1;
    int cross_abs_sum = -1;
    int cross_abs_max = -1;
    int main_gate_sum = -1;
    int main_gate_min = -1;
    int main_gate_max = -1;
    int fm_diff_activity_sum = -1;
    int fm_diff_activity_max = -1;
    int fm_abs_activity_sum = -1;
    int fm_abs_activity_max = -1;
    float phase_main_reliance = -1.0f;
    float phase_fm_reliance = -1.0f;
    float phase_cross_reliance = -1.0f;
    int in_check = 0;
    int root_move_count = 0;
    int previous_pv_length = 0;
    int previous_bestmove_changed = 0;
    int pv_move_changed = 0;
    int final_bestmove_changed = 0;
    int stopped = 0;
    std::uint64_t signal_compute_ns = 0;
    std::uint64_t log_write_ns = 0;
};

inline std::mutex Mutex;
inline std::vector<Event> Events;
inline std::uint64_t NextRootId = 0;

inline std::uint64_t HashPackedSfen(const PackedSfen& packed) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : packed.data) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline RootContext BeginRoot(Position& pos) {
    PackedSfen packed{};
    pos.sfen_pack(packed);
    std::lock_guard<std::mutex> lock(Mutex);
    return {++NextRootId, HashPackedSfen(packed)};
}

inline void Reset() {
    std::lock_guard<std::mutex> lock(Mutex);
    Events.clear();
    NextRootId = 0;
}

inline void Record(Event event) {
    const auto start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(Mutex);
    Events.push_back(event);
    const auto end = std::chrono::steady_clock::now();
    Events.back().log_write_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

inline void Report(std::ostream& output) {
    std::lock_guard<std::mutex> lock(Mutex);
    std::uint64_t fail_low = 0, fail_high = 0, re_searches = 0;
    std::uint64_t extra_nodes = 0, signal_ns = 0, log_ns = 0;
    for (const auto& event : Events) {
        fail_low += event.fail_low_count != 0;
        fail_high += event.fail_high_count != 0;
        re_searches += event.re_search_count;
        extra_nodes += event.extra_nodes;
        signal_ns += event.signal_compute_ns;
        log_ns += event.log_write_ns;
    }
    output << "NNUE aspiration diagnostic events=" << Events.size()
           << " roots=" << NextRootId
           << " fail_low_events=" << fail_low
           << " fail_high_events=" << fail_high
           << " re_searches=" << re_searches
           << " extra_nodes=" << extra_nodes
           << " signal_compute_ns=" << signal_ns
           << " log_write_ns=" << log_ns << '\n';
}

inline bool WriteCsv(const char* path) {
    std::lock_guard<std::mutex> lock(Mutex);
    std::ofstream out(path);
    if (!out)
        return false;
    out << "root_id,packed_sfen_hash,depth,pv_index,previous_iteration_score,average_score,"
           "score_d1,score_d2,score_d3,initial_alpha,initial_beta,initial_window_width,"
           "first_search_score,final_score,fail_low_count,fail_high_count,re_search_count,"
           "total_nodes,first_search_nodes,extra_nodes,static_eval,material,bucket,"
           "router_margin,router_top1_logit,router_top2_logit,"
           "lca_abs_delta_sum,lca_abs_delta_max,cross_abs_sum,cross_abs_max,"
           "main_gate_sum,main_gate_min,main_gate_max,"
           "fm_diff_activity_sum,fm_diff_activity_max,fm_abs_activity_sum,"
           "fm_abs_activity_max,phase_main_reliance,phase_fm_reliance,"
           "phase_cross_reliance,in_check,root_move_count,"
           "previous_pv_length,previous_bestmove_changed,pv_move_changed,"
           "final_bestmove_changed,stopped,signal_compute_ns,log_write_ns\n";
    for (const auto& e : Events) {
        out << e.root_id << ',' << e.packed_sfen_hash << ',' << e.depth << ',' << e.pv_index
            << ',' << e.previous_iteration_score << ',' << e.average_score
            << ',' << e.score_d1 << ',' << e.score_d2 << ',' << e.score_d3
            << ',' << e.initial_alpha << ',' << e.initial_beta
            << ',' << e.initial_window_width << ',' << e.first_search_score
            << ',' << e.final_score << ',' << e.fail_low_count << ',' << e.fail_high_count
            << ',' << e.re_search_count << ',' << e.total_nodes << ',' << e.first_search_nodes
            << ',' << e.extra_nodes << ',' << e.static_eval << ',' << e.material
            << ',' << e.bucket << ',' << e.router_margin << ',' << e.router_top1_logit
            << ',' << e.router_top2_logit << ',' << e.lca_abs_delta_sum
            << ',' << e.lca_abs_delta_max << ',' << e.cross_abs_sum
            << ',' << e.cross_abs_max << ',' << e.main_gate_sum << ','
            << e.main_gate_min << ',' << e.main_gate_max << ','
            << e.fm_diff_activity_sum << ',' << e.fm_diff_activity_max << ','
            << e.fm_abs_activity_sum << ',' << e.fm_abs_activity_max << ','
            << e.phase_main_reliance << ',' << e.phase_fm_reliance << ','
            << e.phase_cross_reliance << ',' << e.in_check << ',' << e.root_move_count
            << ',' << e.previous_pv_length << ',' << e.previous_bestmove_changed
            << ',' << e.pv_move_changed << ',' << e.final_bestmove_changed
            << ',' << e.stopped << ',' << e.signal_compute_ns << ',' << e.log_write_ns << '\n';
    }
    return bool(out);
}

}  // namespace YaneuraOu::Search::NnueAspirationLog

#endif
#endif
