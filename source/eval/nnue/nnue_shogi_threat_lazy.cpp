#include "../../config.h"

#if defined(ENABLE_NNUE_SHOGI_THREAT_SPARSE_PROTOTYPE)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#include "../../movegen.h"
#include "../../position.h"
#include "nnue_shogi_threat_lazy.h"
#include "nnue_shogi_threat_sparse.h"

namespace YaneuraOu::NnueThreatLazy {
namespace {

using namespace NnueShogiThreatSparse;
using Accumulator = std::array<std::array<std::int32_t, Width>, COLOR_NB>;

constexpr std::size_t CacheSize = 512;
constexpr std::size_t CacheMask = CacheSize - 1;
constexpr std::size_t MaxContexts = 256;
static_assert((CacheSize & CacheMask) == 0, "cache must be power of two");
#if defined(ENABLE_NNUE_SHOGI_THREAT_LAZY_STATS)
constexpr bool CollectRuntimeStats = true;
#else
constexpr bool CollectRuntimeStats = false;
#endif

struct CacheSlot {
  const StateInfo* state = nullptr;
  Key64 key = 0;
  alignas(32) Accumulator accumulator{};
};

struct Context {
  std::array<CacheSlot, CacheSize> cache{};
  std::vector<std::uint32_t> compact_scratch;
  Statistics stats{};
  bool registered = false;

  Context() { compact_scratch.reserve(1024); }
};

std::once_flag table_once;
std::unique_ptr<std::int16_t[]> folded_table;
std::array<std::atomic<Context*>, MaxContexts> contexts{};
std::atomic<std::size_t> context_count{0};

inline std::int16_t pseudo_weight(const std::uint16_t index,
                                  const std::size_t channel) {
  std::uint32_t x = std::uint32_t(index) * 0x9e3779b9u
                  ^ std::uint32_t(channel + 1) * 0x85ebca6bu;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return std::int16_t(int(x & 63u) - 32);
}

Context& context() {
  thread_local Context value;
  if (!value.registered) {
    const auto slot = context_count.fetch_add(1, std::memory_order_relaxed);
    if (slot < contexts.size())
      contexts[slot].store(&value, std::memory_order_release);
    value.registered = true;
  }
  return value;
}

inline Key64 state_key(const StateInfo* state) {
  return static_cast<Key64>(state->key());
}

inline std::size_t cache_index(const StateInfo* state, const Key64) {
  // Search reuses one StateInfo object per ply.  Indexing by the object
  // identity therefore keeps all ancestors on the active path resident,
  // while a sibling naturally replaces the previous value for the same ply.
  // Mixing the position key here made every sibling choose an unrelated slot
  // and caused avoidable collisions/refreshes even though the pointer and key
  // are still both checked for correctness below.
  std::uint64_t x = std::uint64_t(reinterpret_cast<std::uintptr_t>(state) >> 6);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  return std::size_t(x) & CacheMask;
}

CacheSlot* find_slot(Context& ctx, const StateInfo* state) {
  const auto key = state_key(state);
  auto& slot = ctx.cache[cache_index(state, key)];
  return slot.state == state && slot.key == key ? &slot : nullptr;
}

CacheSlot& destination_slot(Context& ctx, const StateInfo* state) {
  return ctx.cache[cache_index(state, state_key(state))];
}

inline void update_row(const std::uint16_t index,
                       std::array<std::int32_t, Width>& accumulator,
                       const bool add) {
  const auto* row = folded_table.get() + std::size_t(index) * Width;
#if defined(USE_AVX2)
  for (std::size_t channel = 0; channel < Width; channel += 16) {
    const __m256i weights = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(row + channel));
    const __m256i low = _mm256_cvtepi16_epi32(
        _mm256_castsi256_si128(weights));
    const __m256i high = _mm256_cvtepi16_epi32(
        _mm256_extracti128_si256(weights, 1));
    auto a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        accumulator.data() + channel));
    auto a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        accumulator.data() + channel + 8));
    a0 = add ? _mm256_add_epi32(a0, low) : _mm256_sub_epi32(a0, low);
    a1 = add ? _mm256_add_epi32(a1, high) : _mm256_sub_epi32(a1, high);
    _mm256_store_si256(reinterpret_cast<__m256i*>(
        accumulator.data() + channel), a0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(
        accumulator.data() + channel + 8), a1);
  }
#else
  for (std::size_t channel = 0; channel < Width; ++channel)
    accumulator[channel] += add ? row[channel] : -row[channel];
#endif
}

inline void update_compact_batch(const std::uint32_t* compact,
                                 const std::size_t count,
                                 const Color perspective,
                                 std::array<std::int32_t, Width>& accumulator) {
#if defined(USE_AVX2)
  for (std::size_t base = 0; base < Width; base += 64) {
    __m256i acc[8];
    for (std::size_t lane = 0; lane < 8; ++lane)
      acc[lane] = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          accumulator.data() + base + lane * 8));
    for (std::size_t i = 0; i < count; ++i) {
      const auto packed = compact[i];
      const auto index = std::uint16_t(
          perspective == BLACK ? packed & 0x1fffu
                               : (packed >> 13) & 0x1fffu);
      const bool add = bool((packed >> 26) & 1u);
      const auto* row = folded_table.get() + std::size_t(index) * Width + base;
      for (std::size_t block = 0; block < 4; ++block) {
        const __m256i weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row + block * 16));
        const __m256i low = _mm256_cvtepi16_epi32(
            _mm256_castsi256_si128(weights));
        const __m256i high = _mm256_cvtepi16_epi32(
            _mm256_extracti128_si256(weights, 1));
        acc[block * 2] = add
            ? _mm256_add_epi32(acc[block * 2], low)
            : _mm256_sub_epi32(acc[block * 2], low);
        acc[block * 2 + 1] = add
            ? _mm256_add_epi32(acc[block * 2 + 1], high)
            : _mm256_sub_epi32(acc[block * 2 + 1], high);
      }
    }
    for (std::size_t lane = 0; lane < 8; ++lane)
      _mm256_store_si256(reinterpret_cast<__m256i*>(
          accumulator.data() + base + lane * 8), acc[lane]);
  }
#else
  for (std::size_t i = 0; i < count; ++i) {
    const auto packed = compact[i];
    const auto index = std::uint16_t(
        perspective == BLACK ? packed & 0x1fffu
                             : (packed >> 13) & 0x1fffu);
    update_row(index, accumulator, bool((packed >> 26) & 1u));
  }
#endif
}

void refresh(const Position& pos, Accumulator& accumulator) {
  std::array<Edge, MaxRelations> edges{};
  for (const Color perspective : {BLACK, WHITE}) {
    accumulator[perspective].fill(0);
    const auto count = std::min(generate_edges(
        pos, perspective, edges.data(), edges.size()), edges.size());
    for (std::size_t i = 0; i < count; ++i)
      update_row(edges[i].index, accumulator[perspective], true);
  }
}

struct DecodedDirty {
  Square from;
  Square to;
  Piece attacker;
  Piece target;
  bool add;
};

inline DecodedDirty decode(const StateInfo::ThreatDirtyRecord record) {
  const auto packed = record.packed;
  return {Square(packed & 0x7fu), Square((packed >> 7) & 0x7fu),
          Piece((packed >> 14) & 0x1fu), Piece((packed >> 19) & 0x1fu),
          bool((packed >> 24) & 1u)};
}

const Accumulator& ensure(const Position& pos) {
  initialize();
  auto& ctx = context();
  auto& stats = ctx.stats;
  if constexpr (CollectRuntimeStats)
    ++stats.evaluate_requests;
  const StateInfo* current = pos.state();
  if (auto* cached = find_slot(ctx, current)) {
    if constexpr (CollectRuntimeStats)
      ++stats.cache_hits;
    return cached->accumulator;
  }

  std::array<const StateInfo*, MAX_PLY + 8> chain{};
  std::size_t chain_size = 0;
  const StateInfo* cursor = current;
  CacheSlot* ancestor = nullptr;
  while (cursor && chain_size < chain.size()) {
    if ((ancestor = find_slot(ctx, cursor)))
      break;
    chain[chain_size++] = cursor;
    cursor = cursor->previous;
  }
  if constexpr (CollectRuntimeStats)
    stats.max_chain = std::max(stats.max_chain,
                               std::uint64_t(chain_size));

  alignas(32) Accumulator work{};
  bool can_update = ancestor != nullptr;
  if (can_update)
    work = ancestor->accumulator;
  for (std::size_t i = 0; can_update && i < chain_size; ++i) {
    const auto& dirty = chain[chain_size - 1 - i]->threatDirty;
    if (dirty.full_refresh || dirty.overflow)
      can_update = false;
  }

  if (!can_update) {
    refresh(pos, work);
    if constexpr (CollectRuntimeStats)
      ++stats.refreshes;
  } else {
    ctx.compact_scratch.clear();
    for (std::size_t i = 0; i < chain_size; ++i) {
      const auto& dirty = chain[chain_size - 1 - i]->threatDirty;
      if constexpr (CollectRuntimeStats) {
        ++stats.dirty_states_applied;
        stats.dirty_rows_applied += dirty.count;
      }
      for (std::size_t j = 0; j < dirty.count; ++j) {
        const auto edge = decode(dirty.records[j]);
        std::uint16_t black, white;
        feature_indices(edge.attacker, edge.target, edge.from, edge.to,
                        black, white);
        ctx.compact_scratch.push_back(
            std::uint32_t(black) | (std::uint32_t(white) << 13)
            | (std::uint32_t(edge.add) << 26));
      }
    }
    for (const Color perspective : {BLACK, WHITE})
      update_compact_batch(ctx.compact_scratch.data(),
                           ctx.compact_scratch.size(), perspective,
                           work[perspective]);
    if constexpr (CollectRuntimeStats)
      ++stats.lazy_updates;
  }

  auto& destination = destination_slot(ctx, current);
  destination.state = current;
  destination.key = state_key(current);
  destination.accumulator = work;
  return destination.accumulator;
}

void clear_context(Context& ctx) {
  for (auto& slot : ctx.cache) {
    slot.state = nullptr;
    slot.key = 0;
  }
  ctx.compact_scratch.clear();
  ctx.stats = {};
}

}  // namespace

void initialize() {
  std::call_once(table_once, [] {
    folded_table = std::make_unique<std::int16_t[]>(FeatureCount * Width);
    for (std::size_t index = 0; index < FeatureCount; ++index)
      for (std::size_t channel = 0; channel < Width; ++channel)
        folded_table[index * Width + channel] =
            pseudo_weight(std::uint16_t(index), channel);
  });
}

void on_dirty_published(const std::size_t rows) {
  if constexpr (CollectRuntimeStats) {
    auto& stats = context().stats;
    ++stats.moves;
    stats.generated_dirty_rows += rows;
  }
}

void on_evaluate(const Position& pos) { (void)ensure(pos); }

void reset_all_statistics() {
  const auto count = std::min(context_count.load(std::memory_order_acquire),
                              contexts.size());
  for (std::size_t i = 0; i < count; ++i)
    if (auto* ctx = contexts[i].load(std::memory_order_acquire))
      clear_context(*ctx);
}

Statistics aggregate_statistics() {
  Statistics total{};
  const auto count = std::min(context_count.load(std::memory_order_acquire),
                              contexts.size());
  for (std::size_t i = 0; i < count; ++i) {
    const auto* ctx = contexts[i].load(std::memory_order_acquire);
    if (!ctx) continue;
    total.moves += ctx->stats.moves;
    total.generated_dirty_rows += ctx->stats.generated_dirty_rows;
    total.evaluate_requests += ctx->stats.evaluate_requests;
    total.cache_hits += ctx->stats.cache_hits;
    total.lazy_updates += ctx->stats.lazy_updates;
    total.refreshes += ctx->stats.refreshes;
    total.dirty_states_applied += ctx->stats.dirty_states_applied;
    total.dirty_rows_applied += ctx->stats.dirty_rows_applied;
    total.max_chain = std::max(total.max_chain, ctx->stats.max_chain);
  }
  return total;
}

void print_statistics(std::ostream& output) {
  const auto s = aggregate_statistics();
  const auto percent = [](const std::uint64_t n, const std::uint64_t d) {
    return d ? 100.0 * double(n) / double(d) : 0.0;
  };
  const auto skipped = s.generated_dirty_rows > s.dirty_rows_applied
      ? s.generated_dirty_rows - s.dirty_rows_applied : 0;
  output << "[Shogi Threat Lazy Accumulator]" << std::endl
         << "moves=" << s.moves
         << " generated_dirty_rows=" << s.generated_dirty_rows << std::endl
         << "evaluate_requests=" << s.evaluate_requests
         << " cache_hits=" << s.cache_hits
         << " lazy_updates=" << s.lazy_updates
         << " refreshes=" << s.refreshes
         << " update_rate=" << std::fixed << std::setprecision(3)
         << percent(s.lazy_updates, s.evaluate_requests) << '%' << std::endl
         << "dirty_states_applied=" << s.dirty_states_applied
         << " dirty_rows_applied=" << s.dirty_rows_applied
         << " avg_rows/eval="
         << (s.evaluate_requests ? double(s.dirty_rows_applied)
                                  / s.evaluate_requests : 0.0)
         << " max_chain=" << s.max_chain << std::endl
         << "skipped_dirty_rows=" << skipped
         << " skipped_row_update_rate="
         << percent(skipped, s.generated_dirty_rows) << '%' << std::endl;
}

void self_test(std::ostream& output, std::uint64_t games, const int max_ply) {
  initialize();
  clear_context(context());
  Position pos;
  StateInfo root;
  std::vector<StateInfo> states(static_cast<std::size_t>(max_ply));
  PRNG prng(UINT64_C(0x83a2f17ec7c0ffee));
  std::uint64_t moves = 0, evaluated = 0, skipped_positions = 0;
  std::uint64_t mismatches = 0, lazy_ns = 0, max_skip_streak = 0;
  std::uint64_t skip_streak = 0;
  std::uint64_t drops = 0, captures = 0, promotions = 0;

  for (std::uint64_t game = 0; game < games; ++game) {
    pos.set_hirate(&root);
    (void)ensure(pos);
    for (int ply = 0; ply < max_ply; ++ply) {
      MoveList<LEGAL_ALL> legal(pos);
      if (legal.size() == 0) break;
      const Move move = legal.begin()[prng.rand(legal.size())];
      drops += move.is_drop();
      captures += pos.piece_on(move.to_sq()) != NO_PIECE;
      promotions += move.is_promote();
      pos.do_move(move, states[ply]);
      ++moves;

      // Deliberately leave long gaps as well as random short gaps, so an
      // evaluated descendant frequently applies several StateInfo deltas.
      const bool should_evaluate = (ply % 31 == 0)
          || (prng.rand(4) == 0) || (skip_streak >= 23);
      if (!should_evaluate) {
        ++skipped_positions;
        ++skip_streak;
        max_skip_streak = std::max(max_skip_streak, skip_streak);
        continue;
      }
      skip_streak = 0;
      const auto started = std::chrono::steady_clock::now();
      const auto& lazy = ensure(pos);
      lazy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count();
      Accumulator fresh{};
      refresh(pos, fresh);
      ++evaluated;
      if (lazy != fresh) {
        ++mismatches;
        output << "threat_lazy_mismatch game=" << game
               << " ply=" << ply << " move=" << move
               << " sfen=" << pos.sfen() << std::endl;
        return;
      }
    }
  }

  output << "[Shogi Threat Lazy Correctness]" << std::endl
         << "games=" << games << " moves=" << moves
         << " evaluated=" << evaluated
         << " unevaluated=" << skipped_positions
         << " mismatches=" << mismatches << std::endl
         << "coverage drop=" << drops << " capture=" << captures
         << " promotion=" << promotions
         << " max_skipped_ply_chain=" << max_skip_streak << std::endl
         << "mean_lazy_ensure_ns="
         << (evaluated ? double(lazy_ns) / evaluated : 0.0) << std::endl;
  print_statistics(output);
}

}  // namespace YaneuraOu::NnueThreatLazy

#endif
