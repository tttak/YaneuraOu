// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "nnue_architecture.h"
#include "nnue_signal.h"

namespace YaneuraOu {
namespace Eval::NNUE {

#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
#if !defined(NNUE_HALFKAHM2_SIMPLE)
#error "NNUE_SIMPLE_ACCUMULATOR_STACK is Simple HalfKA_HM2-only"
#endif

using SimpleMainAccumulator = std::int16_t
    [2][kRefreshTriggers.size()][kTransformedFeatureDimensions];

// Experiment 136 Phase A intentionally contains only the data consumed by the
// HalfKA_HM2 transformer.  In particular, the generic Complex FM factor sums
// are not carried into the per-worker stack.
struct alignas(64) SimpleAccumulatorPayload {
  SimpleMainAccumulator accumulation{};
};
#endif

#if defined(NNUE_SIMPLE_PAIR64_ANY)
// The relation machinery is shared by the 64d and 32d LocalPair variants;
// only the compact accumulator/transform width differs.
inline constexpr IndexType kSimplePairDimensions =
#if defined(NNUE_SIMPLE_LOCALPAIR32_R5)
    32;
#else
    64;
#endif
#endif

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
  // The large payload is owned by SimpleAccumulatorStack.  Keeping the same
  // indexing syntax minimizes changes to the proven SIMD kernels.
  std::int16_t (*accumulation)[kRefreshTriggers.size()]
                              [kTransformedFeatureDimensions] = nullptr;
#else
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
#endif

#if defined(NNUE_SIMPLE_PAIR64_ANY)
  // Experiment 121: PP3WidePL has an independent compact accumulator.  It is
  // materialized together with the ordinary FT accumulator, so the existing
  // computed_accumulation flag is the single validity bit for both paths.
  alignas(64) std::int16_t
      pp3wide64_accumulation[2][kSimplePairDimensions];
#endif


#if !defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
  // 因子計算用 (FM項)
  struct FactorGroup {
      std::int64_t sum_v[32];   // Σv
      std::int64_t sum_v2[32];  // Σv^2
  };

  struct FactorPart {
      FactorGroup halfka;   // HalfKA (12672 ～ 203670)
      FactorGroup ksdg;     // KSDG3 (0 ～ 12671)
  } factors[2];             // [手番]
#endif

  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
  // Captured at the destination state.  It lets Phase A replay an arbitrary
  // HalfKA_HM2 dirty suffix without consulting the leaf Position's king
  // squares for older transitions.
  Square stack_king_square[COLOR_NB] = {SQ_NONE, SQ_NONE};
  bool* stack_computed = nullptr;
#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
  // Diagnostic provenance only.  Production score validity lives in the
  // EvalHash entry and is deliberately independent from FT validity.
  bool stack_score_valid = false;
  Value stack_cached_score = VALUE_ZERO;
#endif
#endif
#if defined(EVAL_HASH_VERIFY_HITS)
  // Diagnostic-only provenance.  This is deliberately absent from normal
  // builds so it cannot alter production StateInfo size/cache behavior.
  // 0=uncomputed, 1=incremental, 2=scratch, 3=Finny, 4=null-copy.
  std::uint8_t debug_accumulator_source = 0;
  bool debug_was_null_move = false;
  std::uint32_t debug_move_raw = 0;
  std::int32_t debug_game_ply = 0;
  std::uint32_t debug_children_before_materialization = 0;
  std::uint32_t debug_last_child_move_raw = 0;
#endif
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
  // Experiment 102: independent scalar accumulator.  It is observation-only
  // and absent from production/default builds.
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
  double kp_progress_accumulation[2] = {0.0, 0.0};
#else
  std::int32_t kp_progress_accumulation[2] = {0, 0};
#endif
  bool computed_kp_progress = false;
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
  // Valid only when computed_score was produced by a full Network evaluation.
  // Kept diagnostic-only so production StateInfo size and cache behavior do not change.
  NnueSignalSnapshot nnue_signal{};
#endif
#if defined(USE_NNUE_ROUTER_LMR)
  // Minimal cached Router payload for the logger-free LMR match candidate.
  NnueRouterLmrSignal nnue_router_lmr_signal{};
#endif
};

#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)

// Non-search callers (Position::set(), test commands) do not own a Worker.
// Give them compact thread-local backing keyed by the lightweight Accumulator
// address.  Search states are rebound to the Worker stack before use.
inline SimpleAccumulatorPayload& EnsureSimpleFallbackPayload(Accumulator& accumulator) {
  struct Slot {
    const Accumulator* owner = nullptr;
    SimpleAccumulatorPayload payload{};
  };
  static thread_local std::array<Slot, MAX_PLY + 32> slots{};

  if (accumulator.accumulation)
    for (auto& slot : slots)
      if (slot.payload.accumulation == accumulator.accumulation)
        return slot.payload;

  for (auto& slot : slots)
    if (slot.owner == &accumulator) {
      accumulator.accumulation = slot.payload.accumulation;
      return slot.payload;
    }

  for (auto& slot : slots)
    if (!slot.owner) {
      slot.owner = &accumulator;
      accumulator.accumulation = slot.payload.accumulation;
      return slot.payload;
    }

  // Test code can recycle StateInfo addresses.  A deterministic address slot
  // is safe because a non-search scratch caller never retains more than
  // MAX_PLY live states; the Worker path does not use this pool.
  auto& slot = slots[(reinterpret_cast<std::uintptr_t>(&accumulator) >> 6)
                     % slots.size()];
  slot.owner = &accumulator;
  accumulator.accumulation = slot.payload.accumulation;
  return slot.payload;
}

inline void EnsureSimpleAccumulatorStorage(Accumulator& accumulator) {
  if (!accumulator.accumulation)
    (void) EnsureSimpleFallbackPayload(accumulator);
}

class SimpleAccumulatorStack {
 public:
  struct alignas(64) Entry {
    SimpleAccumulatorPayload payload{};
    const void* state = nullptr;
    bool computed = false;
    Square king_square[COLOR_NB] = {SQ_NONE, SQ_NONE};
  };

  void reset(Accumulator& root, const void* state) {
    depth_ = 0;
    auto& entry = entries_[0];
    if (root.accumulation)
      std::memcpy(entry.payload.accumulation, root.accumulation,
                  sizeof(entry.payload.accumulation));
    entry.state = state;
    entry.computed = root.computed_accumulation;
    entry.king_square[BLACK] = root.stack_king_square[BLACK];
    entry.king_square[WHITE] = root.stack_king_square[WHITE];
    bind(root, entry);
  }

  void push(Accumulator& destination, const void* state) {
    assert(depth_ < MAX_PLY);
    auto& entry = entries_[++depth_];
    entry.state = state;
    entry.computed = false;
    entry.king_square[BLACK] = destination.stack_king_square[BLACK];
    entry.king_square[WHITE] = destination.stack_king_square[WHITE];
    destination.computed_accumulation = false;
    destination.computed_score = false;
#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
    destination.stack_score_valid = false;
    destination.stack_cached_score = VALUE_ZERO;
#endif
    bind(destination, entry);
  }

  void pop(const void* destinationState) {
    assert(depth_ > 0);
    assert(entries_[depth_].state == destinationState);
    --depth_;
  }

  std::size_t depth() const { return depth_; }
  const Entry& current() const { return entries_[depth_]; }
  static constexpr std::size_t capacity() { return MAX_PLY + 1; }

 private:
  static void bind(Accumulator& accumulator, Entry& entry) {
    accumulator.accumulation = entry.payload.accumulation;
    accumulator.stack_computed = &entry.computed;
    accumulator.stack_king_square[BLACK] = entry.king_square[BLACK];
    accumulator.stack_king_square[WHITE] = entry.king_square[WHITE];
  }

  std::array<Entry, MAX_PLY + 1> entries_{};
  std::size_t depth_ = 0;
};

#endif

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
