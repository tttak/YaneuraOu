#pragma once

#if defined(ENABLE_ROOT_TIME_RISK_BUDGET)

#include "../../search.h"
#include "root_time_risk_model_generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <vector>

#ifndef NNUE_ROOT_TIME_RISK_TOP_PERCENT
#define NNUE_ROOT_TIME_RISK_TOP_PERCENT 1
#endif
#ifndef NNUE_ROOT_TIME_EXTRA_PERCENT
#define NNUE_ROOT_TIME_EXTRA_PERCENT 10
#endif

#if NNUE_ROOT_TIME_RISK_TOP_PERCENT != 1 && NNUE_ROOT_TIME_RISK_TOP_PERCENT != 5
#error "NNUE_ROOT_TIME_RISK_TOP_PERCENT must be 1 or 5"
#endif
#if NNUE_ROOT_TIME_EXTRA_PERCENT != 5 && NNUE_ROOT_TIME_EXTRA_PERCENT != 10 \
    && NNUE_ROOT_TIME_EXTRA_PERCENT != 20
#error "NNUE_ROOT_TIME_EXTRA_PERCENT must be 5, 10, or 20"
#endif

namespace YaneuraOu::Search::RootTimeRiskBudget {

struct Context {
    std::uint64_t original_limit = 0;
    std::uint64_t extended_limit = 0;
    bool time_mode = false;
    std::uint64_t previous_total_nodes = 0;
    std::uint64_t previous_effort = 0;
    std::array<int, 2> previous_scores{};
    int score_count = 0;
    std::array<std::uint16_t, 3> previous_moves{};
    int move_count = 0;
    bool triggered = false;
    int trigger_depth = 0;
    int trigger_score = 0;
    float trigger_logit = 0.0f;
    std::uint16_t trigger_move = 0;
    std::uint16_t next_move = 0;
    bool saw_next_iteration = false;
    std::uint64_t nodes_at_trigger = 0;
    std::uint64_t elapsed_at_trigger_ms = 0;
};

struct Decision {
    bool trigger = false;
    float logit = 0.0f;
    std::uint64_t extended_limit = 0;
};

struct Row {
    int trigger_depth = 0;
    int trigger_score = 0;
    float trigger_logit = 0.0f;
    std::uint64_t original_limit = 0;
    std::uint64_t extended_limit = 0;
    std::uint64_t nodes_at_trigger = 0;
    std::uint64_t final_nodes = 0;
    std::uint64_t added_nodes = 0;
    std::uint16_t trigger_move = 0;
    std::uint16_t next_move = 0;
    std::uint16_t final_move = 0;
    int saw_next_iteration = 0;
    int post_trigger_changed = 0;
    int trigger_matches_final = 0;
    int next_matches_final = 0;
    int helped = 0;
    int harmed = 0;
    int depth_6_8 = 0;
    int depth_9_plus = 0;
    int balanced_score = 0;
    int time_mode = 0;
    std::uint64_t elapsed_at_trigger_ms = 0;
    std::uint64_t final_elapsed_ms = 0;
    std::uint64_t added_time_ms = 0;
};

struct Counters {
    std::uint64_t roots = 0;
    std::uint64_t eligible_iterations = 0;
    std::uint64_t threshold_events = 0;
    std::uint64_t triggered_roots = 0;
    std::uint64_t changed_after_trigger = 0;
    std::uint64_t next_observed = 0;
    std::uint64_t helped = 0;
    std::uint64_t harmed = 0;
    std::uint64_t total_added_nodes = 0;
    std::uint64_t total_added_time_ms = 0;
};

inline std::mutex Mutex;
inline Counters Totals;
inline std::vector<Row> Rows;

inline Context BeginRoot(const std::uint64_t node_limit,
                         const std::uint64_t movetime_limit) {
    Context context{};
    if (node_limit)
        context.original_limit = node_limit;
#if defined(ENABLE_ROOT_TIME_RISK_BUDGET_MOVETIME)
    else if (movetime_limit) {
        context.original_limit = movetime_limit;
        context.time_mode = true;
    }
#else
    (void) movetime_limit;
#endif
    return context;
}

inline float ScaledScore(const int value) {
    return static_cast<float>(std::clamp(value, -30000, 30000)) / 1000.0f;
}

inline std::uint16_t FirstMove(const RootMove& root_move) {
    return static_cast<std::uint16_t>(root_move.pv.empty()
      ? 0 : root_move.pv[0].to_move16().to_u16());
}

inline float Threshold() {
#if NNUE_ROOT_TIME_RISK_TOP_PERCENT == 1
    return Model::Top1Logit;
#else
    return Model::Top5Logit;
#endif
}

inline Decision OnCompletedIteration(Context& context, const int depth,
                                     const RootMoves& root_moves,
                                     const std::uint64_t total_nodes,
                                     const std::uint64_t elapsed_ms) {
    Decision decision{};
    if (root_moves.empty())
        return decision;

    const auto& current = root_moves[0];
    const int score = static_cast<int>(current.score);
    const std::uint16_t move = FirstMove(current);
    const std::uint64_t iteration_nodes =
      std::max<std::uint64_t>(1, total_nodes - context.previous_total_nodes);
    const std::uint64_t effort_delta = current.effort >= context.previous_effort
      ? current.effort - context.previous_effort : 0;

    if (context.triggered && !context.saw_next_iteration) {
        context.saw_next_iteration = true;
        context.next_move = move;
    }

    if (!context.triggered && context.original_limit && depth >= 3
        && std::abs(score) < 30000) {
        const float velocity = context.score_count >= 1
          ? ScaledScore(score - context.previous_scores[0]) : 0.0f;
        const float old_velocity = context.score_count >= 2
          ? ScaledScore(context.previous_scores[0] - context.previous_scores[1]) : 0.0f;
        int streak = 1;
        for (int i = context.move_count - 1;
             i >= 0 && i >= context.move_count - 3; --i) {
            if (context.previous_moves[static_cast<std::size_t>(i)] != move)
                break;
            ++streak;
        }
        const float signed_squared = static_cast<float>(current.meanSquaredScore);
        const std::array<float, 15> features{{
          depth / 20.0f,
          1.0f,
          ScaledScore(score),
          static_cast<float>(std::log1p(static_cast<double>(iteration_nodes)) / 15.0),
          static_cast<float>(std::log1p(static_cast<double>(total_nodes)) / 15.0),
          static_cast<float>(effort_delta) / static_cast<float>(iteration_nodes),
          ScaledScore(score - static_cast<int>(current.averageScore)),
          velocity,
          velocity - old_velocity,
          streak / 4.0f,
          velocity * old_velocity < 0.0f ? 1.0f : 0.0f,
          context.move_count > 0
              && context.previous_moves[static_cast<std::size_t>(context.move_count - 1)] != move
            ? 1.0f : 0.0f,
          std::copysign(static_cast<float>(std::log1p(std::abs(signed_squared)) / 20.0),
                        signed_squared),
          static_cast<float>(current.pv.size()) / 32.0f,
          static_cast<float>(current.selDepth) / 32.0f,
        }};
        decision.logit = Model::Logit(features);
        {
            std::lock_guard<std::mutex> lock(Mutex);
            ++Totals.eligible_iterations;
            if (decision.logit >= Threshold())
                ++Totals.threshold_events;
        }
        if (decision.logit >= Threshold()) {
            context.triggered = true;
            context.trigger_depth = depth;
            context.trigger_score = score;
            context.trigger_logit = decision.logit;
            context.trigger_move = move;
            context.nodes_at_trigger = total_nodes;
            context.elapsed_at_trigger_ms = elapsed_ms;
            context.extended_limit =
              (context.original_limit * (100 + NNUE_ROOT_TIME_EXTRA_PERCENT) + 99) / 100;
            decision.trigger = true;
            decision.extended_limit = context.extended_limit;
        }
    }

    context.previous_total_nodes = total_nodes;
    context.previous_effort = current.effort;
    if (context.score_count == 0) {
        context.previous_scores[0] = score;
        context.score_count = 1;
    } else {
        context.previous_scores[1] = context.previous_scores[0];
        context.previous_scores[0] = score;
        context.score_count = 2;
    }
    if (context.move_count < 3)
        context.previous_moves[static_cast<std::size_t>(context.move_count++)] = move;
    else {
        context.previous_moves[0] = context.previous_moves[1];
        context.previous_moves[1] = context.previous_moves[2];
        context.previous_moves[2] = move;
    }
    return decision;
}

inline void Finish(const Context& context, const RootMoves& root_moves,
                   const std::uint64_t final_nodes,
                   const std::uint64_t final_elapsed_ms) {
    std::lock_guard<std::mutex> lock(Mutex);
    ++Totals.roots;
    if (!context.triggered || root_moves.empty())
        return;
    const auto final_move = FirstMove(root_moves[0]);
    const std::uint64_t added_nodes = !context.time_mode && final_nodes > context.original_limit
      ? final_nodes - context.original_limit : 0;
    const std::uint64_t added_time_ms = context.time_mode
      && final_elapsed_ms > context.original_limit
      ? final_elapsed_ms - context.original_limit : 0;
    const bool changed = context.saw_next_iteration
      && context.next_move != context.trigger_move;
    const bool trigger_final = context.trigger_move == final_move;
    const bool next_final = context.saw_next_iteration && context.next_move == final_move;
    const bool helped = context.saw_next_iteration && !trigger_final && next_final;
    const bool harmed = context.saw_next_iteration && trigger_final && !next_final;
    ++Totals.triggered_roots;
    Totals.total_added_nodes += added_nodes;
    Totals.total_added_time_ms += added_time_ms;
    if (context.saw_next_iteration) {
        ++Totals.next_observed;
        Totals.changed_after_trigger += changed;
        Totals.helped += helped;
        Totals.harmed += harmed;
    }
    Rows.push_back({context.trigger_depth, context.trigger_score, context.trigger_logit,
                    context.original_limit, context.extended_limit,
                    context.nodes_at_trigger, final_nodes, added_nodes,
                    context.trigger_move, context.next_move, final_move,
                    context.saw_next_iteration, changed, trigger_final, next_final,
                    helped, harmed,
                    context.trigger_depth >= 6 && context.trigger_depth <= 8,
                    context.trigger_depth >= 9,
                    std::abs(context.trigger_score) < 300,
                    context.time_mode, context.elapsed_at_trigger_ms,
                    final_elapsed_ms, added_time_ms});
}

inline void Reset() {
    std::lock_guard<std::mutex> lock(Mutex);
    Totals = {};
    Rows.clear();
}

inline void Report(std::ostream& out) {
    std::lock_guard<std::mutex> lock(Mutex);
    const auto pct = [](const std::uint64_t n, const std::uint64_t d) {
        return d ? 100.0 * static_cast<double>(n) / static_cast<double>(d) : 0.0;
    };
    out << "root time risk budget counters begin\n"
        << "top_percent=" << NNUE_ROOT_TIME_RISK_TOP_PERCENT
        << " extra_percent=" << NNUE_ROOT_TIME_EXTRA_PERCENT
        << " threshold_logit=" << Threshold() << '\n'
        << "roots=" << Totals.roots
        << " eligible_iterations=" << Totals.eligible_iterations
        << " threshold_events=" << Totals.threshold_events
        << " threshold_event_pct=" << pct(Totals.threshold_events, Totals.eligible_iterations)
        << '\n'
        << "triggered_roots=" << Totals.triggered_roots
        << " trigger_pct=" << pct(Totals.triggered_roots, Totals.roots)
        << " total_added_nodes=" << Totals.total_added_nodes
        << " mean_added_nodes="
        << (Totals.triggered_roots ? Totals.total_added_nodes / Totals.triggered_roots : 0)
        << " total_added_time_ms=" << Totals.total_added_time_ms
        << " mean_added_time_ms="
        << (Totals.triggered_roots ? Totals.total_added_time_ms / Totals.triggered_roots : 0)
        << '\n'
        << "next_observed=" << Totals.next_observed
        << " changed_after_trigger=" << Totals.changed_after_trigger
        << " changed_pct=" << pct(Totals.changed_after_trigger, Totals.next_observed)
        << " helped=" << Totals.helped
        << " helped_pct=" << pct(Totals.helped, Totals.next_observed)
        << " harmed=" << Totals.harmed
        << " harmed_pct=" << pct(Totals.harmed, Totals.next_observed) << '\n'
        << "root time risk budget counters end\n";
}

inline bool WriteCsv(const char* path) {
    std::lock_guard<std::mutex> lock(Mutex);
    std::ofstream out(path);
    if (!out)
        return false;
    out << "trigger_depth,trigger_score,trigger_logit,original_limit,extended_limit,"
           "nodes_at_trigger,final_nodes,added_nodes,trigger_move,next_move,final_move,"
           "saw_next_iteration,post_trigger_changed,trigger_matches_final,"
           "next_matches_final,helped,harmed,depth_6_8,depth_9_plus,balanced_score,"
           "time_mode,elapsed_at_trigger_ms,final_elapsed_ms,added_time_ms\n";
    out << std::setprecision(9);
    for (const auto& row : Rows)
        out << row.trigger_depth << ',' << row.trigger_score << ',' << row.trigger_logit << ','
            << row.original_limit << ',' << row.extended_limit << ','
            << row.nodes_at_trigger << ',' << row.final_nodes << ',' << row.added_nodes << ','
            << row.trigger_move << ',' << row.next_move << ',' << row.final_move << ','
            << row.saw_next_iteration << ',' << row.post_trigger_changed << ','
            << row.trigger_matches_final << ',' << row.next_matches_final << ','
            << row.helped << ',' << row.harmed << ',' << row.depth_6_8 << ','
            << row.depth_9_plus << ',' << row.balanced_score << ',' << row.time_mode << ','
            << row.elapsed_at_trigger_ms << ',' << row.final_elapsed_ms << ','
            << row.added_time_ms << '\n';
    return bool(out);
}

}  // namespace YaneuraOu::Search::RootTimeRiskBudget

#endif
