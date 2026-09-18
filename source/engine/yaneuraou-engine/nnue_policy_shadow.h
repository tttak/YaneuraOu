// Experiment 51: shadow-only policy ordering diagnostics.
#ifndef YANEURAOU_NNUE_POLICY_SHADOW_H_INCLUDED
#define YANEURAOU_NNUE_POLICY_SHADOW_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_POLICY_SHADOW)

#include "../../eval/nnue/nnue_policy_probe.h"
#include "../../eval/nnue/nnue_signal.h"
#include "../../movegen.h"
#include "../../position.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace YaneuraOu::Search::NnuePolicyShadow {

enum Cohort : std::size_t {
    All, TtPresent, TtAbsent, Quiet, Capture, Drop, Check, Promotion,
    LmrEligible, TtAbsentQuiet, TtAbsentLmrEligible,
    HistoryLow, HistoryMedium, HistoryHigh, CohortCount
};
constexpr std::array<const char*, CohortCount> kCohortNames = {
    "all", "tt-present", "tt-absent", "quiet", "capture", "drop", "check",
    "promotion", "lmr-eligible", "tt-absent-quiet", "tt-absent-lmr-eligible",
    "history-low", "history-medium", "history-high"};
constexpr std::size_t kRankBins = 256;

struct AtomicRankStats {
    std::atomic<std::uint64_t> count{0}, existing_sum{0}, policy_sum{0};
    std::atomic<std::uint64_t> policy_top1{0}, policy_top3{0}, policy_top5{0};
    std::atomic<std::uint64_t> existing_top1{0}, existing_top3{0}, existing_top5{0};
    std::atomic<std::uint64_t> improved{0}, worsened{0}, equal{0};
    std::array<std::atomic<std::uint64_t>, kRankBins> existing_hist{};
    std::array<std::atomic<std::uint64_t>, kRankBins> policy_hist{};
};

struct GlobalStats {
    std::array<std::array<AtomicRankStats, CohortCount>, 2> best{};
    std::array<std::array<AtomicRankStats, CohortCount>, 2> cutoff{};
    std::atomic<std::uint64_t> valid_nodes{0}, legal_moves{0};
    std::atomic<std::uint64_t> query_evals{0};
    std::array<std::atomic<std::uint64_t>, 2> query_projection_ns{};
    std::atomic<std::uint64_t> feature_generation_ns{0};
    std::array<std::atomic<std::uint64_t>, 2> scoring_ns{};
};
inline GlobalStats g_stats;

struct RawRow {
    std::uint64_t key = 0;
    std::uint16_t move = 0;
    int existing_rank = 0, policy16_rank = 0, policy32_rank = 0, history = 0;
    float logit16 = 0.0f, logit32 = 0.0f;
    bool tt = false, capture = false, quiet = false, drop = false, check = false;
    bool promotion = false, lmr_eligible = false, lmr_research = false;
    bool best = false, cutoff = false;
};
inline std::mutex g_raw_mutex;
inline std::vector<RawRow> g_raw_rows;
constexpr std::size_t kRawRowLimit = 500000;

inline std::array<int, 20> Features(const Position& position, const Move move) {
    const Color us = position.side_to_move();
    const Color them = ~us;
    const Square to = move.to_sq();
    const Square normalized_to = us == BLACK ? to : Inv(to);
    const Square normalized_from = move.is_drop()
        ? SQ_ZERO : (us == BLACK ? move.from_sq() : Inv(move.from_sq()));
    const Square own_king = us == BLACK
        ? position.square<KING>(us) : Inv(position.square<KING>(us));
    const Square enemy_king = us == BLACK
        ? position.square<KING>(them) : Inv(position.square<KING>(them));
    return {
        move.is_drop() ? 81 + static_cast<int>(move.move_dropped_piece())
                       : static_cast<int>(move.from_sq()),
        static_cast<int>(to),
        static_cast<int>(raw_type_of(position.moved_piece_before(move))),
        move.is_promote() ? 1 : 0,
        move.is_drop() ? 1 : 0,
        position.capture(move) ? static_cast<int>(type_of(position.piece_on(to))) : 0,
        static_cast<int>(type_of(position.moved_piece_after(move))),
        position.capture(move) ? 1 : 0,
        position.gives_check(move) ? 1 : 0,
        move.is_drop() ? 17 : int(file_of(normalized_to)) - int(file_of(normalized_from)) + 8,
        move.is_drop() ? 17 : int(rank_of(normalized_to)) - int(rank_of(normalized_from)) + 8,
        int(file_of(normalized_to)), int(rank_of(normalized_to)),
        int(file_of(normalized_to)) - int(file_of(own_king)) + 8,
        int(rank_of(normalized_to)) - int(rank_of(own_king)) + 8,
        int(file_of(normalized_to)) - int(file_of(enemy_king)) + 8,
        int(rank_of(normalized_to)) - int(rank_of(enemy_king)) + 8,
        dist(normalized_to, enemy_king),
        std::min(int(position.board_effect[us].effect(to)), 3),
        std::min(int(position.board_effect[them].effect(to)), 3),
    };
}

struct MoveRecord {
    Move move = Move::none();
    float logit16 = 0.0f, logit32 = 0.0f;
    int rank16 = 0, rank32 = 0, existing_rank = 0, history = 0;
    bool tt = false, capture = false, quiet = false, drop = false, check = false;
    bool promotion = false, lmr_eligible = false, lmr_research = false;
};

inline std::size_t RankBin(const int rank) {
    return static_cast<std::size_t>(std::clamp(rank, 1, int(kRankBins)) - 1);
}

inline void Add(AtomicRankStats& stats, const int existing, const int policy) {
    stats.count.fetch_add(1, std::memory_order_relaxed);
    stats.existing_sum.fetch_add(existing, std::memory_order_relaxed);
    stats.policy_sum.fetch_add(policy, std::memory_order_relaxed);
    stats.existing_top1.fetch_add(existing <= 1, std::memory_order_relaxed);
    stats.existing_top3.fetch_add(existing <= 3, std::memory_order_relaxed);
    stats.existing_top5.fetch_add(existing <= 5, std::memory_order_relaxed);
    stats.policy_top1.fetch_add(policy <= 1, std::memory_order_relaxed);
    stats.policy_top3.fetch_add(policy <= 3, std::memory_order_relaxed);
    stats.policy_top5.fetch_add(policy <= 5, std::memory_order_relaxed);
    stats.improved.fetch_add(policy < existing, std::memory_order_relaxed);
    stats.worsened.fetch_add(policy > existing, std::memory_order_relaxed);
    stats.equal.fetch_add(policy == existing, std::memory_order_relaxed);
    stats.existing_hist[RankBin(existing)].fetch_add(1, std::memory_order_relaxed);
    stats.policy_hist[RankBin(policy)].fetch_add(1, std::memory_order_relaxed);
}

inline void AddCohorts(std::array<AtomicRankStats, CohortCount>& stats,
                       const MoveRecord& move, const bool tt_present, const int policy_rank) {
    const auto add = [&](const Cohort cohort) { Add(stats[cohort], move.existing_rank, policy_rank); };
    add(All); add(tt_present ? TtPresent : TtAbsent);
    if (move.quiet) add(Quiet);
    if (move.capture) add(Capture);
    if (move.drop) add(Drop);
    if (move.check) add(Check);
    if (move.promotion) add(Promotion);
    if (move.lmr_eligible) add(LmrEligible);
    if (!tt_present && move.quiet) add(TtAbsentQuiet);
    if (!tt_present && move.lmr_eligible) add(TtAbsentLmrEligible);
    add(move.history < -4000 ? HistoryLow : move.history > 4000 ? HistoryHigh : HistoryMedium);
}

class NodeObservation {
public:
    NodeObservation(const Position& position, const Move tt_move,
                    const Eval::NNUE::NnueSignalEvalAccess& access)
        : key_(position.key()), tt_move_(tt_move), sampled_((key_ & 255) == 0) {
        if (!access.signal.valid
            || (access.source != Eval::NNUE::NnueSignalEvalSource::FreshNetwork
                && access.source != Eval::NNUE::NnueSignalEvalSource::AccumulatorCached))
            return;
        const auto feature_started = std::chrono::steady_clock::now();
        MoveList<LEGAL_ALL> legal(position);
        records_.reserve(legal.size());
        for (const auto& ext : legal) {
            const Move move = ext;
            MoveRecord record;
            record.move = move;
            record.tt = move == tt_move;
            record.capture = position.capture(move);
            record.quiet = !record.capture;
            record.drop = move.is_drop();
            record.check = position.gives_check(move);
            record.promotion = move.is_promote();
            const auto feature = Features(position, move);
            const auto score16_started = std::chrono::steady_clock::now();
            record.logit16 = Eval::NNUE::PolicyProbe::Score<16>(
                access.signal.policy_query16, feature);
            const auto score16_ended = std::chrono::steady_clock::now();
            record.logit32 = Eval::NNUE::PolicyProbe::Score<32>(
                access.signal.policy_query32, feature);
            const auto score32_ended = std::chrono::steady_clock::now();
            score16_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                score16_ended - score16_started).count();
            score32_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                score32_ended - score16_ended).count();
            records_.push_back(record);
        }
        const auto feature_ended = std::chrono::steady_clock::now();
        total_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            feature_ended - feature_started).count();
        Rank(&MoveRecord::logit16, &MoveRecord::rank16);
        Rank(&MoveRecord::logit32, &MoveRecord::rank32);
        valid_ = true;
        g_stats.valid_nodes.fetch_add(1, std::memory_order_relaxed);
        g_stats.legal_moves.fetch_add(records_.size(), std::memory_order_relaxed);
        if (access.source == Eval::NNUE::NnueSignalEvalSource::FreshNetwork) {
            g_stats.query_evals.fetch_add(1, std::memory_order_relaxed);
            g_stats.query_projection_ns[0].fetch_add(
                access.signal.policy_query16_projection_ns, std::memory_order_relaxed);
            g_stats.query_projection_ns[1].fetch_add(
                access.signal.policy_query32_projection_ns, std::memory_order_relaxed);
        }
        g_stats.feature_generation_ns.fetch_add(
            total_ns_ - score16_ns_ - score32_ns_, std::memory_order_relaxed);
        g_stats.scoring_ns[0].fetch_add(score16_ns_, std::memory_order_relaxed);
        g_stats.scoring_ns[1].fetch_add(score32_ns_, std::memory_order_relaxed);
    }

    void ObserveMove(const Move move, const int existing_rank, const int history,
                     const bool lmr_eligible) {
        if (auto* record = Find(move)) {
            record->existing_rank = existing_rank;
            record->history = history;
            record->lmr_eligible = lmr_eligible;
        }
    }

    void MarkLmrResearch(const Move move) {
        if (auto* record = Find(move))
            record->lmr_research = true;
    }

    void RecordResult(const Move best_move, const bool cutoff) {
        if (!valid_ || best_move == Move::none())
            return;
        auto* record = Find(best_move);
        if (!record || record->existing_rank <= 0)
            return;
        for (int width = 0; width < 2; ++width) {
            auto& destination = cutoff ? g_stats.cutoff[width] : g_stats.best[width];
            AddCohorts(destination, *record, tt_move_ != Move::none(),
                       width == 0 ? record->rank16 : record->rank32);
        }
        if (sampled_)
            SaveRaw(best_move, cutoff);
    }

private:
    using LogitMember = float MoveRecord::*;
    using RankMember = int MoveRecord::*;
    void Rank(const LogitMember logit, const RankMember rank) {
        std::vector<std::size_t> order(records_.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](const auto a, const auto b) {
            return records_[a].*logit > records_[b].*logit;
        });
        for (std::size_t i = 0; i < order.size(); ++i)
            records_[order[i]].*rank = static_cast<int>(i + 1);
    }
    MoveRecord* Find(const Move move) {
        const auto it = std::find_if(records_.begin(), records_.end(),
            [&](const MoveRecord& record) { return record.move == move; });
        return it == records_.end() ? nullptr : &*it;
    }
    void SaveRaw(const Move best_move, const bool cutoff) {
        std::lock_guard<std::mutex> lock(g_raw_mutex);
        if (g_raw_rows.size() >= kRawRowLimit) return;
        for (const auto& move : records_) {
            if (g_raw_rows.size() >= kRawRowLimit) break;
            g_raw_rows.push_back({key_, move.move.to_u16(), move.existing_rank,
                move.rank16, move.rank32, move.history, move.logit16, move.logit32,
                move.tt, move.capture, move.quiet, move.drop, move.check,
                move.promotion, move.lmr_eligible, move.lmr_research,
                move.move == best_move, cutoff && move.move == best_move});
        }
    }
    std::uint64_t key_ = 0;
    Move tt_move_ = Move::none();
    bool sampled_ = false, valid_ = false;
    std::int64_t total_ns_ = 0, score16_ns_ = 0, score32_ns_ = 0;
    std::vector<MoveRecord> records_;
};

inline double Percent(const std::uint64_t value, const std::uint64_t count) {
    return count ? 100.0 * static_cast<double>(value) / count : 0.0;
}
inline int Median(const std::array<std::atomic<std::uint64_t>, kRankBins>& hist,
                  const std::uint64_t count) {
    if (!count) return 0;
    const auto target = (count + 1) / 2;
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < hist.size(); ++i)
        if ((sum += hist[i].load(std::memory_order_relaxed)) >= target)
            return static_cast<int>(i + 1);
    return static_cast<int>(kRankBins);
}

inline void Print(std::ostream& out, const char* outcome, const int width,
                  const char* cohort, const AtomicRankStats& stats) {
    const auto n = stats.count.load(std::memory_order_relaxed);
    if (!n) return;
    const auto es = stats.existing_sum.load(std::memory_order_relaxed);
    const auto ps = stats.policy_sum.load(std::memory_order_relaxed);
    out << outcome << ',' << width << ',' << cohort << ',' << n << ','
        << double(es) / n << ',' << Median(stats.existing_hist, n) << ','
        << double(ps) / n << ',' << Median(stats.policy_hist, n) << ','
        << Percent(stats.existing_top1.load(), n) << ','
        << Percent(stats.existing_top3.load(), n) << ','
        << Percent(stats.existing_top5.load(), n) << ','
        << Percent(stats.policy_top1.load(), n) << ','
        << Percent(stats.policy_top3.load(), n) << ','
        << Percent(stats.policy_top5.load(), n) << ','
        << Percent(stats.improved.load(), n) << ','
        << Percent(stats.worsened.load(), n) << ','
        << Percent(stats.equal.load(), n) << '\n';
}

inline void Report(std::ostream& out) {
    out << "outcome,width,cohort,count,existing_mean,existing_median,policy_mean,policy_median,"
           "existing_top1_pct,existing_top3_pct,existing_top5_pct,policy_top1_pct,"
           "policy_top3_pct,policy_top5_pct,improved_pct,worsened_pct,equal_pct\n";
    for (int width = 0; width < 2; ++width)
        for (std::size_t cohort = 0; cohort < CohortCount; ++cohort) {
            Print(out, "best", width ? 32 : 16, kCohortNames[cohort],
                  g_stats.best[width][cohort]);
            Print(out, "cutoff", width ? 32 : 16, kCohortNames[cohort],
                  g_stats.cutoff[width][cohort]);
        }
    const auto nodes = g_stats.valid_nodes.load();
    const auto moves = g_stats.legal_moves.load();
    const auto query_evals = g_stats.query_evals.load();
    out << "# timing valid_nodes=" << nodes << " legal_moves=" << moves
        << " query_evals=" << query_evals
        << " p16_query_ns/eval=" << (query_evals ? double(g_stats.query_projection_ns[0].load()) / query_evals : 0.0)
        << " p32_query_ns/eval=" << (query_evals ? double(g_stats.query_projection_ns[1].load()) / query_evals : 0.0)
        << " feature_ns/move=" << (moves ? double(g_stats.feature_generation_ns.load()) / moves : 0.0)
        << " p16_score_ns/move=" << (moves ? double(g_stats.scoring_ns[0].load()) / moves : 0.0)
        << " p32_score_ns/move=" << (moves ? double(g_stats.scoring_ns[1].load()) / moves : 0.0)
        << '\n';
}

inline void RawReport(std::ostream& out) {
    out << "position_key,move_raw,existing_rank,policy16_rank,policy32_rank,history_score,"
           "logit16,logit32,tt_move,capture,quiet,drop,check,promotion,lmr_eligible,"
           "lmr_research,best,cutoff\n";
    std::lock_guard<std::mutex> lock(g_raw_mutex);
    out << std::setprecision(9);
    for (const auto& r : g_raw_rows)
        out << r.key << ',' << r.move << ',' << r.existing_rank << ',' << r.policy16_rank
            << ',' << r.policy32_rank << ',' << r.history << ',' << r.logit16 << ','
            << r.logit32 << ',' << r.tt << ',' << r.capture << ',' << r.quiet << ','
            << r.drop << ',' << r.check << ',' << r.promotion << ',' << r.lmr_eligible
            << ',' << r.lmr_research << ',' << r.best << ',' << r.cutoff << '\n';
}

inline void ResetRankStats(AtomicRankStats& stats) {
    stats.count = 0; stats.existing_sum = 0; stats.policy_sum = 0;
    stats.policy_top1 = 0; stats.policy_top3 = 0; stats.policy_top5 = 0;
    stats.existing_top1 = 0; stats.existing_top3 = 0; stats.existing_top5 = 0;
    stats.improved = 0; stats.worsened = 0; stats.equal = 0;
    for (auto& value : stats.existing_hist) value = 0;
    for (auto& value : stats.policy_hist) value = 0;
}

inline void Reset() {
    for (int width = 0; width < 2; ++width)
        for (std::size_t cohort = 0; cohort < CohortCount; ++cohort) {
            ResetRankStats(g_stats.best[width][cohort]);
            ResetRankStats(g_stats.cutoff[width][cohort]);
        }
    g_stats.valid_nodes = 0; g_stats.legal_moves = 0; g_stats.query_evals = 0;
    g_stats.query_projection_ns[0] = 0; g_stats.query_projection_ns[1] = 0;
    g_stats.feature_generation_ns = 0;
    g_stats.scoring_ns[0] = 0; g_stats.scoring_ns[1] = 0;
    std::lock_guard<std::mutex> lock(g_raw_mutex);
    g_raw_rows.clear();
}

}  // namespace YaneuraOu::Search::NnuePolicyShadow

#endif
#endif
