#ifndef YANEURAOU_NNUE_HAO_RISK_LOGGER_H_INCLUDED
#define YANEURAOU_NNUE_HAO_RISK_LOGGER_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL)

#include "../../eval/nnue/nnue_signal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <vector>

namespace YaneuraOu::Search::NnueHaoRiskLog {

using Snapshot = Eval::NNUE::NnueSignalSnapshot;
constexpr std::size_t kKinds = Snapshot::HaoRiskCount;
constexpr std::size_t kQ = 256;
constexpr std::size_t kOther = 3;
constexpr std::size_t kBuckets = 12;
constexpr std::size_t kPlyGroups = 5;

inline constexpr std::array<const char*, kKinds> kNames = {
  "fc1-only-bucket", "context-only", "context+fc1-bucket", "residual-positive-bucket"};

struct NodeOutcome {
    std::uint64_t count = 0, abs_error_sum = 0, fail_high = 0, fail_low = 0;
};
struct LmrOutcome {
    std::uint64_t count = 0, reduced_fail_high = 0, research = 0, cutoff = 0;
};
struct Shadow {
    std::uint64_t samples = 0, completed = 0, wrong = 0;
    std::int64_t boundary_delta_sum = 0, static_delta_sum = 0;
};
struct Correlation {
    std::uint64_t count = 0;
    long double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
};
struct PerSignal {
    std::array<std::uint64_t, kQ> fresh{};
    std::array<NodeOutcome, kQ> node{};
    std::array<LmrOutcome, kQ> lmr{};
    std::array<LmrOutcome, kQ> uncovered{};
    std::array<std::array<LmrOutcome, kQ>, kBuckets> bucket{};
    std::array<std::array<LmrOutcome, kQ>, kPlyGroups> search_ply{};
    std::array<std::array<LmrOutcome, kQ>, kPlyGroups> context_game_ply{};
    std::array<Correlation, kOther> pearson{};
    std::array<std::array<std::uint64_t, kQ * kQ>, kOther> rank_joint{};
    std::array<Shadow, kQ> rfp{};
    std::array<Shadow, kQ> futility{};
};
struct Stats {
    std::array<PerSignal, kKinds> signal{};
    std::uint64_t fresh_compute_ns = 0;
    std::uint64_t fresh_compute_count = 0;
    std::uint64_t futility_sequence = 0;
};

// Tiny Threads=1 per-root counters.  Keeping these separate avoids clearing
// and snapshotting the large rank-joint diagnostic matrices after every root.
// The phase-3 runner fixes Threads=1; these counters are not for multithreaded
// aggregate reporting.
struct RootCompactStats {
    NodeOutcome node{};
    LmrOutcome lmr{};
    LmrOutcome uncovered{};
    Shadow rfp{};
    Shadow futility{};
    std::uint64_t fresh = 0;
};
inline RootCompactStats g_root_compact{};
inline void ResetRootCompact() { g_root_compact = RootCompactStats{}; }

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
    return ply < 32 ? 0 : ply < 64 ? 1 : ply < 96 ? 2 : ply < 128 ? 3 : 4;
}
inline std::uint8_t OtherRank(const Snapshot& s, const std::size_t kind) {
    if (kind == 0)
        return static_cast<std::uint8_t>(std::clamp<int>(
          int(std::log2(std::max(0, s.router_margin) + 1.0) * 16.0), 0, 255));
    if (kind == 1)
        return static_cast<std::uint8_t>(std::clamp(s.lca_abs_delta_sum * 255 / 4064, 0, 255));
    return static_cast<std::uint8_t>(std::min<int>(s.cross_abs_max * 2, 255));
}
inline double OtherValue(const Snapshot& s, const std::size_t kind) {
    return kind == 0 ? s.router_margin : kind == 1 ? s.lca_abs_delta_sum : s.cross_abs_max;
}
inline void Add(Correlation& v, const double x, const double y) {
    ++v.count; v.sx += x; v.sy += y; v.sxx += x*x; v.syy += y*y; v.sxy += x*y;
}
inline void Add(LmrOutcome& d, const LmrOutcome& s) {
    d.count += s.count; d.reduced_fail_high += s.reduced_fail_high;
    d.research += s.research; d.cutoff += s.cutoff;
}
inline void Add(NodeOutcome& d, const NodeOutcome& s) {
    d.count += s.count; d.abs_error_sum += s.abs_error_sum;
    d.fail_high += s.fail_high; d.fail_low += s.fail_low;
}
inline void Add(Shadow& d, const Shadow& s) {
    d.samples += s.samples; d.completed += s.completed; d.wrong += s.wrong;
    d.boundary_delta_sum += s.boundary_delta_sum; d.static_delta_sum += s.static_delta_sum;
}
inline void Add(Correlation& d, const Correlation& s) {
    d.count += s.count; d.sx += s.sx; d.sy += s.sy; d.sxx += s.sxx;
    d.syy += s.syy; d.sxy += s.sxy;
}

inline void RecordFresh(const Snapshot& s) {
    auto& stats = Local();
    ++g_root_compact.fresh;
    ++stats.fresh_compute_count;
    stats.fresh_compute_ns += s.hao_risk_compute_ns;
    for (std::size_t kind = 0; kind < kKinds; ++kind)
        ++stats.signal[kind].fresh[s.hao_risk_q8[kind]];
}
inline void RecordNode(const Snapshot& s, const std::uint32_t error,
                       const int result, const int alpha, const int beta) {
    auto& stats = Local();
    ++g_root_compact.node.count;
    g_root_compact.node.abs_error_sum += error;
    g_root_compact.node.fail_high += result >= beta;
    g_root_compact.node.fail_low += result <= alpha;
    for (std::size_t kind = 0; kind < kKinds; ++kind) {
        auto& v = stats.signal[kind].node[s.hao_risk_q8[kind]];
        ++v.count; v.abs_error_sum += error;
        v.fail_high += result >= beta; v.fail_low += result <= alpha;
    }
}
inline void RecordLmr(const Snapshot& s, const int search_ply,
                      const bool reduced_fh, const bool researched,
                      const bool cutoff, const bool router, const bool lca,
                      const bool cross) {
    auto add = [&](LmrOutcome& v) {
        ++v.count; v.reduced_fail_high += reduced_fh;
        v.research += researched; v.cutoff += cutoff;
    };
    auto& stats = Local();
    ++g_root_compact.lmr.count;
    g_root_compact.lmr.reduced_fail_high += reduced_fh;
    g_root_compact.lmr.research += researched;
    g_root_compact.lmr.cutoff += cutoff;
    if (!router && !lca && !cross) {
        ++g_root_compact.uncovered.count;
        g_root_compact.uncovered.reduced_fail_high += reduced_fh;
        g_root_compact.uncovered.research += researched;
        g_root_compact.uncovered.cutoff += cutoff;
    }
    const auto bucket = static_cast<std::size_t>(std::clamp(s.selected_bucket, 0, 11));
    for (std::size_t kind = 0; kind < kKinds; ++kind) {
        auto& p = stats.signal[kind];
        const auto q = s.hao_risk_q8[kind];
        add(p.lmr[q]); add(p.bucket[bucket][q]);
        add(p.search_ply[PlyGroup(search_ply)][q]);
        add(p.context_game_ply[PlyGroup(s.hao_context_game_ply)][q]);
        if (!router && !lca && !cross) add(p.uncovered[q]);
        for (std::size_t other = 0; other < kOther; ++other) {
            Add(p.pearson[other], q, OtherValue(s, other));
            ++p.rank_joint[other][std::size_t(q) * kQ + OtherRank(s, other)];
        }
    }
}

using QValues = std::array<std::uint8_t, kKinds>;
inline QValues Values(const Snapshot& s) {
    QValues result{};
    std::copy_n(s.hao_risk_q8, kKinds, result.begin());
    return result;
}
inline void RecordRfpSelected(const QValues& q) {
    auto& stats = Local();
    ++g_root_compact.rfp.samples;
    for (std::size_t kind = 0; kind < kKinds; ++kind)
        ++stats.signal[kind].rfp[q[kind]].samples;
}
inline void RecordRfpOutcome(const QValues& q, const int result,
                             const int beta, const int static_eval) {
    auto& stats = Local();
    ++g_root_compact.rfp.completed;
    g_root_compact.rfp.wrong += result < beta;
    g_root_compact.rfp.boundary_delta_sum += std::int64_t(result) - beta;
    g_root_compact.rfp.static_delta_sum += std::int64_t(result) - static_eval;
    for (std::size_t kind = 0; kind < kKinds; ++kind) {
        auto& v = stats.signal[kind].rfp[q[kind]];
        ++v.completed; v.wrong += result < beta;
        v.boundary_delta_sum += std::int64_t(result) - beta;
        v.static_delta_sum += std::int64_t(result) - static_eval;
    }
}
inline bool SelectFutilitySample(const Snapshot& s, QValues& q) {
    auto& stats = Local();
    q = Values(s);
    constexpr std::uint64_t kMask = 255;
    const bool selected = (stats.futility_sequence++ & kMask) == 0;
    if (selected)
        ++g_root_compact.futility.samples;
    if (selected)
        for (std::size_t kind = 0; kind < kKinds; ++kind)
            ++stats.signal[kind].futility[q[kind]].samples;
    return selected;
}
inline void RecordFutilityOutcome(const QValues& q, const int result,
                                  const int alpha, const int static_eval) {
    auto& stats = Local();
    ++g_root_compact.futility.completed;
    g_root_compact.futility.wrong += result > alpha;
    g_root_compact.futility.boundary_delta_sum += std::int64_t(result) - alpha;
    g_root_compact.futility.static_delta_sum += std::int64_t(result) - static_eval;
    for (std::size_t kind = 0; kind < kKinds; ++kind) {
        auto& v = stats.signal[kind].futility[q[kind]];
        ++v.completed; v.wrong += result > alpha;
        v.boundary_delta_sum += std::int64_t(result) - alpha;
        v.static_delta_sum += std::int64_t(result) - static_eval;
    }
}

inline void Reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto* stats : g_stats) { stats->~Stats(); ::new (stats) Stats(); }
}
inline std::unique_ptr<Stats> SnapshotStats() {
    auto out = std::make_unique<Stats>();
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto* src : g_stats) {
        out->fresh_compute_ns += src->fresh_compute_ns;
        out->fresh_compute_count += src->fresh_compute_count;
        for (std::size_t kind = 0; kind < kKinds; ++kind) {
            auto& d = out->signal[kind]; const auto& s = src->signal[kind];
            for (std::size_t q = 0; q < kQ; ++q) {
                d.fresh[q] += s.fresh[q]; Add(d.node[q], s.node[q]);
                Add(d.lmr[q], s.lmr[q]); Add(d.uncovered[q], s.uncovered[q]);
                Add(d.rfp[q], s.rfp[q]); Add(d.futility[q], s.futility[q]);
                for (std::size_t b = 0; b < kBuckets; ++b) Add(d.bucket[b][q], s.bucket[b][q]);
                for (std::size_t p = 0; p < kPlyGroups; ++p) {
                    Add(d.search_ply[p][q], s.search_ply[p][q]);
                    Add(d.context_game_ply[p][q], s.context_game_ply[p][q]);
                }
            }
            for (std::size_t other = 0; other < kOther; ++other) {
                Add(d.pearson[other], s.pearson[other]);
                for (std::size_t i = 0; i < kQ*kQ; ++i)
                    d.rank_joint[other][i] += s.rank_joint[other][i];
            }
        }
    }
    return out;
}
inline double Percent(const std::uint64_t n, const std::uint64_t d) {
    return d ? 100.0 * double(n) / double(d) : 0.0;
}
inline double Pearson(const Correlation& v) {
    if (v.count < 2) return 0;
    const long double n=v.count, num=n*v.sxy-v.sx*v.sy;
    const long double dx=n*v.sxx-v.sx*v.sx, dy=n*v.syy-v.sy*v.sy;
    return dx > 0 && dy > 0 ? double(num/std::sqrt(dx*dy)) : 0;
}
inline double Spearman(const std::array<std::uint64_t,kQ*kQ>& joint) {
    std::array<std::uint64_t,kQ> mx{},my{}; std::uint64_t total=0;
    for(std::size_t x=0;x<kQ;++x) for(std::size_t y=0;y<kQ;++y){auto n=joint[x*kQ+y];mx[x]+=n;my[y]+=n;total+=n;}
    if(total<2)return 0;
    std::array<double,kQ> rx{},ry{}; std::uint64_t before=0;
    for(std::size_t i=0;i<kQ;++i){rx[i]=before+(mx[i]+1.0)*.5;before+=mx[i];}
    before=0; for(std::size_t i=0;i<kQ;++i){ry[i]=before+(my[i]+1.0)*.5;before+=my[i];}
    Correlation c{};
    for(std::size_t x=0;x<kQ;++x) for(std::size_t y=0;y<kQ;++y){auto n=joint[x*kQ+y];
      c.count+=n;c.sx+=n*rx[x];c.sy+=n*ry[y];c.sxx+=n*rx[x]*rx[x];c.syy+=n*ry[y]*ry[y];c.sxy+=n*rx[x]*ry[y];}
    return Pearson(c);
}
template<class T> inline T Tail(const std::array<T,kQ>& a, const std::size_t threshold){T r{};for(std::size_t q=threshold;q<kQ;++q)Add(r,a[q]);return r;}
template<class T> inline T Range(const std::array<T,kQ>& a, const std::size_t lo,const std::size_t hi){T r{};for(std::size_t q=lo;q<hi;++q)Add(r,a[q]);return r;}
// Compact per-root output for experiments which attach an external label to
// the root position.  Every Hao signal kind observes the same node/move set,
// so kind 0 is sufficient for the all-q totals.  This is diagnostic-only and
// does not affect search decisions.
inline void ReportRootSummary(std::ostream& out) {
    const auto& node = g_root_compact.node;
    const auto& lmr = g_root_compact.lmr;
    const auto& uncovered = g_root_compact.uncovered;
    const auto& rfp = g_root_compact.rfp;
    const auto& futility = g_root_compact.futility;
    out << "NNUE_ROOT_DIAG"
        << " fresh=" << g_root_compact.fresh
        << " node=" << node.count
        << " node_abs_error_sum=" << node.abs_error_sum
        << " node_fail_high=" << node.fail_high
        << " node_fail_low=" << node.fail_low
        << " lmr=" << lmr.count
        << " lmr_reduced_fail_high=" << lmr.reduced_fail_high
        << " lmr_research=" << lmr.research
        << " lmr_cutoff=" << lmr.cutoff
        << " uncovered_lmr=" << uncovered.count
        << " uncovered_lmr_research=" << uncovered.research
        << " rfp_samples=" << rfp.samples
        << " rfp_completed=" << rfp.completed
        << " rfp_wrong=" << rfp.wrong
        << " futility_samples=" << futility.samples
        << " futility_completed=" << futility.completed
        << " futility_wrong=" << futility.wrong
        << '\n';
}
inline std::size_t Threshold(const std::array<LmrOutcome,kQ>& a,const unsigned bp){std::uint64_t n=0;for(auto&v:a)n+=v.count;auto want=(n*bp+9999)/10000,got=std::uint64_t(0);for(std::size_t q=kQ;q-->0;){got+=a[q].count;if(got>=want)return q;}return 0;}
inline void PrintLmr(std::ostream& o,const LmrOutcome& v){o<<" n="<<v.count<<" FH="<<Percent(v.reduced_fail_high,v.count)<<"% re-search="<<Percent(v.research,v.count)<<"% cutoff="<<Percent(v.cutoff,v.count)<<'%';}
inline void PrintNode(std::ostream& o,const NodeOutcome& v){o<<" n="<<v.count<<" mean|search-static|="<<(v.count?double(v.abs_error_sum)/v.count:0.0)<<" node-FH="<<Percent(v.fail_high,v.count)<<"% node-FL="<<Percent(v.fail_low,v.count)<<'%';}

inline void Report(std::ostream& out) {
    const auto storage=SnapshotStats(); const auto& stats=*storage;
    out<<"[Hao static-vs-depth9 search-risk diagnostic]\n"
       <<"  offline context static score: Hao HalfKP; runtime substitute: current production NNUE static score (distribution shift)\n"
       <<"  q8: round(sigmoid(logit)*255), four frozen 10M probes\n"
       <<"  head compute mean ns/fresh="<<(stats.fresh_compute_count?double(stats.fresh_compute_ns)/stats.fresh_compute_count:0.0)<<"\n";
    constexpr std::array<unsigned,5> tails={50,100,200,500,1000};
    constexpr std::array<const char*,kOther> other_names={"router_margin","lca_abs_delta_sum","cross_abs_max"};
    for(std::size_t kind=0;kind<kKinds;++kind){const auto&p=stats.signal[kind];
      out<<"[Hao risk "<<kNames[kind]<<"]\n";
      const auto all_lmr=Range(p.lmr,0,kQ);out<<"  all:";PrintLmr(out,all_lmr);out<<'\n';
      for(auto bp:tails){auto threshold=Threshold(p.lmr,bp);auto l=Tail(p.lmr,threshold);auto u=Tail(p.uncovered,threshold);auto n=Tail(p.node,threshold);auto r=Tail(p.rfp,threshold);auto f=Tail(p.futility,threshold);
        out<<"  top "<<double(bp)/100.0<<"% q>="<<threshold;PrintLmr(out,l);
        out<<" mean|search-static|="<<(n.count?double(n.abs_error_sum)/n.count:0.0)<<" node-FH="<<Percent(n.fail_high,n.count)<<"% node-FL="<<Percent(n.fail_low,n.count)<<"%\n    uncovered:";PrintLmr(out,u);
        out<<"\n    RFP shadow n="<<r.completed<<" wrong="<<Percent(r.wrong,r.completed)<<"% futility shadow n="<<f.completed<<" wrong="<<Percent(f.wrong,f.completed)<<"%\n";}
      const auto q50=Threshold(p.lmr,5000),q10=Threshold(p.lmr,1000);
      auto bottom=Range(p.lmr,0,q50);auto middle=Range(p.lmr,q50,q10);
      out<<"  bottom/middle: bottom50";PrintLmr(out,bottom);out<<" middle40";PrintLmr(out,middle);out<<'\n';
      auto bottom_node=Range(p.node,0,q50);auto middle_node=Range(p.node,q50,q10);
      out<<"  bottom/middle nodes: bottom50";PrintNode(out,bottom_node);out<<" middle40";PrintNode(out,middle_node);out<<'\n';
      for(std::size_t other=0;other<kOther;++other)out<<"  vs "<<other_names[other]<<" Pearson="<<Pearson(p.pearson[other])<<" binned-Spearman="<<Spearman(p.rank_joint[other])<<'\n';
      out<<"  top10 by bucket:";for(std::size_t b=0;b<kBuckets;++b){auto t=Threshold(p.bucket[b],1000);auto v=Tail(p.bucket[b],t);out<<" B"<<std::setw(2)<<std::setfill('0')<<b<<std::setfill(' ')<<'='<<Percent(v.research,v.count)<<"%(n="<<v.count<<')';}out<<'\n';
      out<<"  top10 by search-ply:";for(std::size_t x=0;x<kPlyGroups;++x){auto t=Threshold(p.search_ply[x],1000);auto v=Tail(p.search_ply[x],t);out<<' '<<x<<'='<<Percent(v.research,v.count)<<"%(n="<<v.count<<')';}out<<" [0=0-31,...,4=128+]\n";
      out<<"  top10 by context-game-ply:";for(std::size_t x=0;x<kPlyGroups;++x){auto t=Threshold(p.context_game_ply[x],1000);auto v=Tail(p.context_game_ply[x],t);out<<' '<<x<<'='<<Percent(v.research,v.count)<<"%(n="<<v.count<<')';}out<<" [0=0-31,...,4=128+]\n";
    }
}

} // namespace YaneuraOu::Search::NnueHaoRiskLog
#endif
#endif
