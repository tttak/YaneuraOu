#pragma once

#if defined(ENABLE_ROOT_MOVE_HISTORY_DIAGNOSTIC)

#include "../../position.h"
#include "../../search.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace YaneuraOu::Search::RootMoveHistoryLog {

constexpr std::size_t MaxMoves = 8;

struct RootContext {
    std::uint64_t root_id = 0;
    std::uint64_t packed_sfen_hash = 0;
    std::uint64_t previous_total_nodes = 0;
};

struct Row {
    std::uint64_t root_id = 0;
    std::uint64_t packed_sfen_hash = 0;
    int depth = 0;
    int rank = 0;
    std::uint16_t move16 = 0;
    int score = 0;
    int previous_score = 0;
    int average_score = 0;
    int mean_squared_score = 0;
    std::uint64_t effort_cumulative = 0;
    std::uint64_t iteration_nodes = 0;
    std::uint64_t total_nodes = 0;
    int sel_depth = 0;
    int pv_length = 0;
    std::string pv;
};

inline std::mutex Mutex;
inline std::vector<Row> Rows;
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
    return {++NextRootId, HashPackedSfen(packed), 0};
}

inline void Reset() {
    std::lock_guard<std::mutex> lock(Mutex);
    Rows.clear();
    NextRootId = 0;
}

inline std::string PvString(const std::vector<Move>& pv) {
    std::ostringstream out;
    for (std::size_t i = 0; i < pv.size(); ++i) {
        if (i)
            out << ';';
        out << pv[i].to_move16().to_u16();
    }
    return out.str();
}

inline void RecordDepth(RootContext& context, const int depth,
                        const RootMoves& root_moves,
                        const std::uint64_t total_nodes) {
    const auto iteration_nodes = total_nodes - context.previous_total_nodes;
    context.previous_total_nodes = total_nodes;
    std::lock_guard<std::mutex> lock(Mutex);
    const auto count = std::min(MaxMoves, root_moves.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& move = root_moves[i];
        Rows.push_back({context.root_id,
                        context.packed_sfen_hash,
                        depth,
                        static_cast<int>(i + 1),
                        static_cast<std::uint16_t>(
                          move.pv.empty() ? 0 : move.pv[0].to_move16().to_u16()),
                        move.score,
                        move.previousScore,
                        move.averageScore,
                        move.meanSquaredScore,
                        move.effort,
                        iteration_nodes,
                        total_nodes,
                        move.selDepth,
                        static_cast<int>(move.pv.size()),
                        PvString(move.pv)});
    }
}

inline void Report(std::ostream& output) {
    std::lock_guard<std::mutex> lock(Mutex);
    output << "root move history rows=" << Rows.size()
           << " roots=" << NextRootId << '\n';
}

inline bool WriteCsv(const char* path) {
    std::lock_guard<std::mutex> lock(Mutex);
    std::ofstream out(path);
    if (!out)
        return false;
    out << "root_id,packed_sfen_hash,depth,rank,move16,score,previous_score,"
           "average_score,mean_squared_score,effort_cumulative,iteration_nodes,"
           "total_nodes,sel_depth,pv_length,pv\n";
    for (const auto& row : Rows) {
        out << row.root_id << ',' << row.packed_sfen_hash << ',' << row.depth << ','
            << row.rank << ',' << row.move16 << ',' << row.score << ','
            << row.previous_score << ',' << row.average_score << ','
            << row.mean_squared_score << ',' << row.effort_cumulative << ','
            << row.iteration_nodes << ',' << row.total_nodes << ','
            << row.sel_depth << ',' << row.pv_length << ','
            << std::quoted(row.pv) << '\n';
    }
    return bool(out);
}

}  // namespace YaneuraOu::Search::RootMoveHistoryLog

#endif
