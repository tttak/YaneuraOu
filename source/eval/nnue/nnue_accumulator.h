// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_architecture.h"
#include "nnue_signal.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];

  // 因子計算用 (FM項)
  struct FactorGroup {
      std::int64_t sum_v[32];   // Σv
      std::int64_t sum_v2[32];  // Σv^2
  };

  struct FactorPart {
      FactorGroup halfka;   // HalfKA (12672 ～ 203670)
      FactorGroup ksdg;     // KSDG3 (0 ～ 12671)
  } factors[2];             // [手番]

  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
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

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
