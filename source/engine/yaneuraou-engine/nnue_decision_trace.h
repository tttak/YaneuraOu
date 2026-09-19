// Diagnostic-only, chunked decision trace for LMR/RFP/futility probes.
#ifndef YANEURAOU_NNUE_DECISION_TRACE_H_INCLUDED
#define YANEURAOU_NNUE_DECISION_TRACE_H_INCLUDED

#include "../../config.h"

#if defined(ENABLE_NNUE_DECISION_TRACE)

#include "../../position.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
#include "nnue_decision_risk_weights.h"
#endif

namespace YaneuraOu::Search::NnueDecisionTrace {

enum class Decision : std::uint8_t { Lmr = 1, Rfp = 2, Futility = 3 };
enum class NodeKind : std::uint8_t { Root = 0, Pv = 1, Cut = 2, All = 3, QSearch = 4 };

enum RecordFlag : std::uint16_t {
    Capture = 1U << 0,
    Quiet = 1U << 1,
    Check = 1U << 2,
    TtMove = 1U << 3,
    Applied = 1U << 4,
    Eligible = 1U << 5,
    ParentSignFlip = 1U << 6,
    AnySignFlip = 1U << 7,
    Oscillation2 = 1U << 8,
    Oscillation3 = 1U << 9,
    FailHigh = 1U << 10,
    FailLow = 1U << 11,
    ShadowSample = 1U << 12,
};

#pragma pack(push, 1)
struct ChunkHeader {
    char magic[8];
    std::uint16_t version;
    std::uint16_t header_size;
    std::uint16_t record_size;
    std::uint16_t record_type;
    std::uint32_t chunk_index;
    std::uint32_t record_count;
    std::uint64_t payload_checksum;
    std::uint64_t first_root_id;
    std::uint64_t last_root_id;
    std::uint8_t reserved[16];
};

struct Record {
    PackedSfen packed_sfen;
    std::uint64_t root_id;
    std::uint64_t root_hash;
    std::uint64_t node_id;
    std::uint64_t parent_node_id;
    std::uint8_t decision;
    std::uint8_t label;
    std::uint8_t side_to_move;
    std::uint8_t node_kind;
    std::uint8_t in_check;
    std::uint8_t improving;
    std::int8_t bucket;
    std::uint8_t risk_q8;
    std::uint16_t flags;
    std::uint16_t move16;
    std::int16_t ply;
    std::int16_t depth;
    std::int16_t move_count;
    std::int16_t reduction;
    std::int32_t static_eval;
    std::int32_t alpha;
    std::int32_t beta;
    std::int32_t material;
    std::int32_t margin;
    std::int32_t history;
    std::int32_t reduced_result;
    std::int32_t full_result;
    std::int32_t parent_abs_delta;
    std::int32_t grandparent_abs_delta;
    std::int32_t great_grandparent_abs_delta;
    float sampling_probability;
    float sampling_weight;
};
#pragma pack(pop)

static_assert(sizeof(PackedSfen) == 32, "trace format requires 32-byte PackedSfen");
static_assert(sizeof(ChunkHeader) == 64, "unexpected decision trace chunk header");
static_assert(sizeof(Record) == 136, "unexpected decision trace record size");

struct Context {
    PackedSfen packed_sfen{};
    std::uint64_t root_id = 0;
    std::uint64_t root_hash = 0;
    std::uint64_t node_id = 0;
    std::uint64_t parent_node_id = 0;
    int ply = 0;
    int depth = 0;
    int alpha = 0;
    int beta = 0;
    int material = 0;
    int static_eval = 0;
    int parent_abs_delta = 0;
    int grandparent_abs_delta = 0;
    int great_grandparent_abs_delta = 0;
    int side_to_move = 0;
    int bucket = -1;
    NodeKind node_kind = NodeKind::All;
    bool in_check = false;
    bool improving = false;
    bool has_position = false;
    bool parent_sign_flip = false;
    bool any_sign_flip = false;
    bool oscillation2 = false;
    bool oscillation3 = false;
};

struct Stats {
    std::array<std::uint64_t, 4> candidates{};
    std::array<std::uint64_t, 4> positives{};
    std::array<std::uint64_t, 4> written{};
    std::uint64_t chunks = 0;
    std::uint64_t write_errors = 0;
};

inline std::mutex g_mutex;
inline std::vector<Record> g_buffer;
inline std::string g_prefix;
inline std::uint32_t g_chunk_index = 0;
inline std::size_t g_chunk_records = 16384;
inline std::atomic<bool> g_active{false};
inline Stats g_stats;
inline std::atomic<std::uint64_t> g_next_node_id{1};
inline std::atomic<std::uint64_t> g_next_root_id{1};
inline std::atomic<std::uint64_t> g_generation{1};
#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
inline std::atomic<std::uint64_t> g_risk_calls{0};
inline std::atomic<std::uint64_t> g_risk_nanoseconds{0};
#endif

struct ThreadState {
    std::uint64_t generation = 0;
    std::array<std::uint64_t, 512> node_by_ply{};
    std::uint64_t root_id = 0;
    std::uint64_t root_hash = 0;
};
inline thread_local ThreadState g_thread;

inline std::uint64_t Fnv1a(const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t value = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        value ^= bytes[i];
        value *= 1099511628211ULL;
    }
    return value;
}

inline bool Active() { return g_active.load(std::memory_order_relaxed); }

inline Context BeginNode(const int ply, const int depth, const int alpha,
                         const int beta, const int material, const bool pv,
                         const bool cut, const bool root, const bool qsearch,
                         const bool in_check) {
    Context result;
    if (!Active())
        return result;
    const auto generation = g_generation.load(std::memory_order_relaxed);
    if (g_thread.generation != generation) {
        g_thread = ThreadState{};
        g_thread.generation = generation;
    }
    result.node_id = g_next_node_id.fetch_add(1, std::memory_order_relaxed);
    result.parent_node_id = ply > 0 && ply < int(g_thread.node_by_ply.size())
                          ? g_thread.node_by_ply[std::size_t(ply - 1)] : 0;
    if (ply >= 0 && ply < int(g_thread.node_by_ply.size()))
        g_thread.node_by_ply[std::size_t(ply)] = result.node_id;
    result.root_id = g_thread.root_id;
    result.root_hash = g_thread.root_hash;
    result.ply = ply;
    result.depth = depth;
    result.alpha = alpha;
    result.beta = beta;
    result.material = material;
    result.in_check = in_check;
    result.node_kind = qsearch ? NodeKind::QSearch
                     : root ? NodeKind::Root
                     : pv ? NodeKind::Pv
                     : cut ? NodeKind::Cut : NodeKind::All;
    return result;
}

inline void CapturePosition(Context& context, Position& pos, const int bucket) {
    if (!Active())
        return;
    pos.sfen_pack(context.packed_sfen);
    context.has_position = true;
    context.side_to_move = int(pos.side_to_move());
    context.bucket = bucket;
    if (context.ply == 0) {
        const auto hash = Fnv1a(&context.packed_sfen, sizeof(context.packed_sfen));
        if (g_thread.root_id == 0 || g_thread.root_hash != hash) {
            g_thread.root_id = g_next_root_id.fetch_add(1, std::memory_order_relaxed);
            g_thread.root_hash = hash;
        }
        context.root_id = g_thread.root_id;
        context.root_hash = g_thread.root_hash;
    } else {
        context.root_id = g_thread.root_id;
        context.root_hash = g_thread.root_hash;
    }
}

inline std::string ChunkName(const std::uint32_t index, const bool temporary) {
    std::ostringstream name;
    name << g_prefix << '_' << std::setfill('0') << std::setw(6) << index << ".bin";
    if (temporary)
        name << ".tmp";
    return name.str();
}

inline bool FlushLocked() {
    if (g_buffer.empty())
        return true;
    const auto temporary = ChunkName(g_chunk_index, true);
    const auto complete = ChunkName(g_chunk_index, false);
    {
        std::ifstream existing(temporary, std::ios::binary);
        std::ifstream final_existing(complete, std::ios::binary);
        if (existing || final_existing) {
            ++g_stats.write_errors;
            return false;
        }
    }
    ChunkHeader header{};
    std::memcpy(header.magic, "NNTRC53", 7);
    header.version = 1;
    header.header_size = sizeof(header);
    header.record_size = sizeof(Record);
    header.record_type = 0;
    header.chunk_index = g_chunk_index;
    header.record_count = static_cast<std::uint32_t>(g_buffer.size());
    header.payload_checksum = Fnv1a(g_buffer.data(), g_buffer.size() * sizeof(Record));
    header.first_root_id = g_buffer.front().root_id;
    header.last_root_id = g_buffer.back().root_id;
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        ++g_stats.write_errors;
        return false;
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(g_buffer.data()),
                 std::streamsize(g_buffer.size() * sizeof(Record)));
    output.flush();
    const bool okay = bool(output);
    output.close();
    if (!okay || std::rename(temporary.c_str(), complete.c_str()) != 0) {
        ++g_stats.write_errors;
        return false;
    }
    ++g_chunk_index;
    ++g_stats.chunks;
    g_buffer.clear();
    return true;
}

inline bool Start(const std::string& prefix, const std::size_t chunk_records) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_active.load(std::memory_order_relaxed) || prefix.empty())
        return false;
    g_prefix = prefix;
    g_chunk_records = std::max<std::size_t>(1, chunk_records);
    g_chunk_index = 0;
    g_buffer.clear();
    g_buffer.reserve(g_chunk_records);
    g_stats = Stats{};
#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
    g_risk_calls.store(0, std::memory_order_relaxed);
    g_risk_nanoseconds.store(0, std::memory_order_relaxed);
#endif
    g_next_node_id = 1;
    g_next_root_id = 1;
    g_generation.fetch_add(1, std::memory_order_relaxed);
    g_active.store(true, std::memory_order_release);
    return true;
}

inline bool Stop() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_active.load(std::memory_order_acquire))
        return false;
    const bool result = FlushLocked();
    g_active.store(false, std::memory_order_release);
    return result;
}

inline std::uint64_t SampleHash(const Record& record) {
    std::uint64_t value = record.root_hash ^ (record.node_id * 0x9e3779b97f4a7c15ULL);
    value ^= std::uint64_t(record.move16) << 17;
    value ^= std::uint64_t(record.decision) << 57;
    value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27; value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline bool Write(Record record) {
    if (!Active() || !record.root_id || !record.node_id)
        return false;
    const auto kind = std::size_t(record.decision);
    const unsigned negative_shift = record.decision == std::uint8_t(Decision::Lmr) ? 4U : 0U;
    const bool keep = record.label || negative_shift == 0
                   || (SampleHash(record) & ((std::uint64_t(1) << negative_shift) - 1)) == 0;
    record.sampling_probability = record.label ? 1.0f
      : 1.0f / float(std::uint64_t(1) << negative_shift);
    record.sampling_weight = 1.0f / record.sampling_probability;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_stats.candidates[kind];
    g_stats.positives[kind] += record.label != 0;
    if (!keep)
        return false;
    ++g_stats.written[kind];
    g_buffer.push_back(record);
    if (g_buffer.size() >= g_chunk_records)
        return FlushLocked();
    return true;
}

inline Record BaseRecord(const Context& context, const Decision decision,
                         const bool label) {
    Record record{};
    record.packed_sfen = context.packed_sfen;
    record.root_id = context.root_id;
    record.root_hash = context.root_hash;
    record.node_id = context.node_id;
    record.parent_node_id = context.parent_node_id;
    record.decision = std::uint8_t(decision);
    record.label = label;
    record.side_to_move = std::uint8_t(context.side_to_move);
    record.node_kind = std::uint8_t(context.node_kind);
    record.in_check = context.in_check;
    record.improving = context.improving;
    record.bucket = std::int8_t(context.bucket);
    record.ply = std::int16_t(std::clamp(context.ply, -32768, 32767));
    record.depth = std::int16_t(std::clamp(context.depth, -32768, 32767));
    record.static_eval = context.static_eval;
    record.alpha = context.alpha;
    record.beta = context.beta;
    record.material = context.material;
    record.parent_abs_delta = context.parent_abs_delta;
    record.grandparent_abs_delta = context.grandparent_abs_delta;
    record.great_grandparent_abs_delta = context.great_grandparent_abs_delta;
    if (context.parent_sign_flip) record.flags |= ParentSignFlip;
    if (context.any_sign_flip) record.flags |= AnySignFlip;
    if (context.oscillation2) record.flags |= Oscillation2;
    if (context.oscillation3) record.flags |= Oscillation3;
    return record;
}

#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
inline std::array<float, NnueDecisionRiskWeights::Dimensions> RiskFeatures(
  const Context& context, const int move_count, const int reduction,
  const int history, const bool capture, const bool check, const bool tt_move) {
    std::array<float, NnueDecisionRiskWeights::Dimensions> x{};
    const auto clipped = [](const int value, const int lo, const int hi) {
        return std::clamp(value, lo, hi);
    };
    x[0] = clipped(context.depth, -2, 32) / 16.0f;
    x[1] = clipped(context.ply, 0, 256) / 128.0f;
    x[2] = clipped(context.static_eval, -4000, 4000) / 2000.0f;
    x[3] = clipped(context.alpha, -4000, 4000) / 2000.0f;
    x[4] = clipped(context.beta, -4000, 4000) / 2000.0f;
    x[5] = clipped(context.static_eval - context.alpha, -4000, 4000) / 2000.0f;
    x[6] = clipped(context.static_eval - context.beta, -4000, 4000) / 2000.0f;
    x[7] = clipped(std::abs(context.material), 0, 8000) / 4000.0f;
    x[8] = context.improving ? 1.0f : 0.0f;
    x[9] = context.in_check ? 1.0f : 0.0f;
    x[10] = clipped(move_count, 0, 64) / 32.0f;
    x[11] = clipped(reduction, -4, 16) / 8.0f;
    x[12] = clipped(history, -20000, 20000) / 10000.0f;
    x[13] = clipped(context.parent_abs_delta, 0, 4000) / 1000.0f;
    x[14] = clipped(context.grandparent_abs_delta, 0, 4000) / 1000.0f;
    x[15] = clipped(context.great_grandparent_abs_delta, 0, 4000) / 1000.0f;
    x[16] = capture ? 1.0f : 0.0f;
    x[17] = check ? 1.0f : 0.0f;
    x[18] = tt_move ? 1.0f : 0.0f;
    x[19] = context.oscillation2 ? 1.0f : 0.0f;
    x[20] = context.oscillation3 ? 1.0f : 0.0f;
    const auto node = std::clamp(int(context.node_kind), 0, 4);
    x[21 + node] = 1.0f;
    if (context.bucket >= 0 && context.bucket < 12)
        x[26 + context.bucket] = 1.0f;
    return x;
}

inline float Dot(const std::array<float, NnueDecisionRiskWeights::Dimensions>& x,
                 const std::array<float, NnueDecisionRiskWeights::Dimensions>& weight,
                 const float bias) {
    float value = bias;
    for (std::size_t index = 0; index < x.size(); ++index)
        value += x[index] * weight[index];
    return value;
}

inline float RiskLogit(const Context& context, const Decision decision,
                       const int move_count = 0, const int reduction = 0,
                       const int history = 0, const bool capture = false,
                       const bool check = false, const bool tt_move = false) {
    const auto begin = std::chrono::steady_clock::now();
    const auto x = RiskFeatures(context, move_count, reduction, history,
                                capture, check, tt_move);
    const float value = decision == Decision::Lmr
      ? Dot(x, NnueDecisionRiskWeights::LMRWeight, NnueDecisionRiskWeights::LMRBias)
      : Dot(x, NnueDecisionRiskWeights::RFPWeight, NnueDecisionRiskWeights::RFPBias);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - begin).count();
    g_risk_calls.fetch_add(1, std::memory_order_relaxed);
    g_risk_nanoseconds.fetch_add(std::uint64_t(elapsed), std::memory_order_relaxed);
    return value;
}

inline std::uint8_t RiskQ8(const Context& context, const Decision decision,
                           const int move_count = 0, const int reduction = 0,
                           const int history = 0, const bool capture = false,
                           const bool check = false, const bool tt_move = false) {
    const float logit = RiskLogit(context, decision, move_count, reduction,
                                  history, capture, check, tt_move);
    const float probability = 1.0f / (1.0f + std::exp(-logit));
    return std::uint8_t(std::clamp(std::lround(probability * 255.0f), 0L, 255L));
}

inline void RiskSelftest(std::ostream& output) {
    float max_lmr = 0.0f, max_rfp = 0.0f, max_probability = 0.0f;
    int q8_mismatches = 0;
    std::array<float, 4> actual_lmr{}, actual_rfp{};
    for (std::size_t row = 0; row < NnueDecisionRiskWeights::SelftestInput.size(); ++row) {
        actual_lmr[row] = Dot(
          NnueDecisionRiskWeights::SelftestInput[row], NnueDecisionRiskWeights::LMRWeight,
          NnueDecisionRiskWeights::LMRBias);
        actual_rfp[row] = Dot(
          NnueDecisionRiskWeights::SelftestInput[row], NnueDecisionRiskWeights::RFPWeight,
          NnueDecisionRiskWeights::RFPBias);
        max_lmr = std::max(max_lmr, std::abs(actual_lmr[row]
          - NnueDecisionRiskWeights::LMRExpectedLogit[row]));
        max_rfp = std::max(max_rfp, std::abs(actual_rfp[row]
          - NnueDecisionRiskWeights::RFPExpectedLogit[row]));
        const float lmr_probability = 1.0f / (1.0f + std::exp(-actual_lmr[row]));
        const float rfp_probability = 1.0f / (1.0f + std::exp(-actual_rfp[row]));
        max_probability = std::max(max_probability, std::abs(lmr_probability
          - NnueDecisionRiskWeights::LMRExpectedProbability[row]));
        max_probability = std::max(max_probability, std::abs(rfp_probability
          - NnueDecisionRiskWeights::RFPExpectedProbability[row]));
        q8_mismatches += std::lround(lmr_probability * 255.0f)
          != NnueDecisionRiskWeights::LMRExpectedQ8[row];
        q8_mismatches += std::lround(rfp_probability * 255.0f)
          != NnueDecisionRiskWeights::RFPExpectedQ8[row];
    }
    int rank_mismatches = 0;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i + 1; j < 4; ++j) {
            rank_mismatches += (actual_lmr[i] < actual_lmr[j])
              != (NnueDecisionRiskWeights::LMRExpectedLogit[i]
                  < NnueDecisionRiskWeights::LMRExpectedLogit[j]);
            rank_mismatches += (actual_rfp[i] < actual_rfp[j])
              != (NnueDecisionRiskWeights::RFPExpectedLogit[i]
                  < NnueDecisionRiskWeights::RFPExpectedLogit[j]);
        }
    output << std::scientific << std::setprecision(9)
           << "decision risk selftest max_logit_diff_lmr=" << max_lmr
           << " max_logit_diff_rfp=" << max_rfp
           << " max_probability_diff=" << max_probability
           << " q8_mismatches=" << q8_mismatches
           << " rank_pair_mismatches=" << rank_mismatches << std::defaultfloat << '\n';
}
#endif

inline void Report(std::ostream& output) {
    std::lock_guard<std::mutex> lock(g_mutex);
    output << "NNUE decision trace active="
           << g_active.load(std::memory_order_relaxed)
           << " chunks=" << g_stats.chunks
           << " buffered=" << g_buffer.size()
           << " write_errors=" << g_stats.write_errors << '\n';
    for (std::size_t kind = 1; kind <= 3; ++kind)
        output << "  type=" << kind << " candidates=" << g_stats.candidates[kind]
               << " positive=" << g_stats.positives[kind]
               << " written=" << g_stats.written[kind] << '\n';
#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
    const auto calls = g_risk_calls.load(std::memory_order_relaxed);
    const auto nanoseconds = g_risk_nanoseconds.load(std::memory_order_relaxed);
    output << "  decision_risk calls=" << calls << " nanoseconds=" << nanoseconds
           << " ns_per_decision=" << (calls ? double(nanoseconds) / calls : 0.0) << '\n';
#endif
}

}  // namespace YaneuraOu::Search::NnueDecisionTrace

#endif  // ENABLE_NNUE_DECISION_TRACE
#endif  // YANEURAOU_NNUE_DECISION_TRACE_H_INCLUDED
