#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace YaneuraOu {
class Position;

namespace NnueThreatLazy {

constexpr std::size_t Width = 128;

struct Statistics {
  std::uint64_t moves = 0;
  std::uint64_t generated_dirty_rows = 0;
  std::uint64_t evaluate_requests = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t lazy_updates = 0;
  std::uint64_t refreshes = 0;
  std::uint64_t dirty_states_applied = 0;
  std::uint64_t dirty_rows_applied = 0;
  std::uint64_t max_chain = 0;
};

// Experiment 83 only.  The pseudo accumulator never contributes to the
// returned NNUE score; these entry points measure infrastructure cost only.
void initialize();
void on_dirty_published(std::size_t rows);
void on_evaluate(const Position& pos);
void reset_all_statistics();
Statistics aggregate_statistics();
void print_statistics(std::ostream& output);
void self_test(std::ostream& output, std::uint64_t games, int max_ply);

}  // namespace NnueThreatLazy
}  // namespace YaneuraOu
