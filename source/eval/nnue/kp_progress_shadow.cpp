#include "../../config.h"

#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)

#include "kp_progress_shadow.h"

#include "nnue_architecture.h"
#include "features/index_list.h"
#include "../../position.h"
#include "../../movegen.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <vector>

namespace YaneuraOu::Eval::NNUE::NnueKpProgressShadow {
namespace {

constexpr std::uint32_t Magic = 0x3150474bu;  // "KGP1" little endian
constexpr std::uint32_t Version = 1;
constexpr double LowThreshold = -1.0986122886681098;
constexpr double MidThreshold = -0.20067069546215138;
#ifndef KP_PROGRESS_SHADOW_TABLE_BITS
#define KP_PROGRESS_SHADOW_TABLE_BITS 32
#endif
#ifndef KP_PROGRESS_SHADOW_SCALE
#define KP_PROGRESS_SHADOW_SCALE 4096
#endif
constexpr int RuntimeScale = KP_PROGRESS_SHADOW_SCALE;

std::vector<float> weights;
#if KP_PROGRESS_SHADOW_TABLE_BITS == 16
std::vector<std::int16_t> runtime_weights;
#elif KP_PROGRESS_SHADOW_TABLE_BITS == 8
std::vector<std::int8_t> runtime_weights;
#endif
float bias = 0.0f;
#if KP_PROGRESS_SHADOW_TABLE_BITS != 32
std::int32_t runtime_bias = 0;
#endif
bool is_loaded = false;
volatile int route_sink = 0;

int route(double logit) {
  return logit < LowThreshold ? 0 : (logit < MidThreshold ? 1 : 2);
}

int route_q(std::int64_t twice_logit_q, int scale) {
  const auto low = static_cast<std::int64_t>(std::llround(2.0 * scale * LowThreshold));
  const auto mid = static_cast<std::int64_t>(std::llround(2.0 * scale * MidThreshold));
  return twice_logit_q < low ? 0 : (twice_logit_q < mid ? 1 : 2);
}

template <typename T>
bool read_exact(std::istream& in, T* dst, std::size_t count = 1) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(sizeof(T) * count));
  return bool(in);
}

void active_indices(const Position& pos, Features::IndexList active[2]) {
  RawFeatures::AppendActiveIndices(
      pos, Features::TriggerEvent::kFriendKingMoved, active);
}

double sum_float(const Features::IndexList& list) {
  double result = 0.0;
  for (const auto index : list)
    result += weights[index];
  return result;
}

#if KP_PROGRESS_SHADOW_TABLE_BITS != 32
std::int32_t sum_runtime(const Features::IndexList& list) {
  std::int32_t result = 0;
  for (const auto index : list) result += runtime_weights[index];
  return result;
}
#endif

std::int64_t sum_q(const Features::IndexList& list, int scale) {
  std::int64_t result = 0;
  for (const auto index : list)
    result += static_cast<std::int32_t>(std::lround(weights[index] * scale));
  return result;
}

void refresh(const Position& pos) {
  auto& acc = pos.state()->accumulator;
  Features::IndexList active[2];
  active_indices(pos, active);
  for (int c = 0; c < COLOR_NB; ++c) {
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
    acc.kp_progress_accumulation[c] = sum_float(active[c]);
#else
    acc.kp_progress_accumulation[c] = sum_runtime(active[c]);
#endif
  }
  acc.computed_kp_progress = true;
}

void ensure(const Position& pos) {
  auto& acc = pos.state()->accumulator;
  if (acc.computed_kp_progress) return;
  const auto* previous = pos.state()->previous;
  if (!previous || !previous->accumulator.computed_kp_progress) {
    refresh(pos);
    return;
  }

  // Standalone diagnostic fallback.  The hot path uses the feature
  // transformer's already-produced active/changed lists and never reaches
  // this duplicate extraction path.
  refresh(pos);
}

ProgressShadowResult from_sums(double black, double white) {
  const double logit = static_cast<double>(bias) + 0.5 * (black + white);
  return {logit, route(logit)};
}

bool is_b08(const Position& pos) {
  constexpr int friend_band[9] = {0,0,0,3,3,3,6,6,6};
  constexpr int enemy_band[9] = {0,0,0,1,1,1,2,2,2};
  const Color stm = pos.side_to_move();
  const Square fk = pos.square<KING>(stm);
  const Square ek = pos.square<KING>(~stm);
  const int fr = stm == BLACK ? rank_of(fk) : rank_of(Inv(fk));
  const int er = stm == BLACK ? rank_of(Inv(ek)) : rank_of(ek);
  return friend_band[fr] + enemy_band[er] == 8;
}

}  // namespace

bool load(const std::string& path, std::ostream* error) {
  std::ifstream in(path, std::ios::binary);
  std::uint32_t magic = 0, version = 0, dimensions = 0;
  float file_bias = 0.0f;
  if (!in || !read_exact(in, &magic) || !read_exact(in, &version)
      || !read_exact(in, &dimensions) || !read_exact(in, &file_bias)
      || magic != Magic || version != Version
      || dimensions != RawFeatures::kDimensions) {
    if (error) *error << "kp_progress load failed: " << path << '\n';
    return false;
  }
  std::vector<float> incoming(dimensions);
  if (!read_exact(in, incoming.data(), incoming.size())) {
    if (error) *error << "kp_progress truncated: " << path << '\n';
    return false;
  }
  weights = std::move(incoming);
  bias = file_bias;
#if KP_PROGRESS_SHADOW_TABLE_BITS == 16
  runtime_weights.resize(weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i)
    runtime_weights[i] = static_cast<std::int16_t>(std::clamp<long>(
        std::lround(weights[i] * RuntimeScale), -32768, 32767));
  runtime_bias = static_cast<std::int32_t>(std::lround(bias * RuntimeScale));
#elif KP_PROGRESS_SHADOW_TABLE_BITS == 8
  runtime_weights.resize(weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i)
    runtime_weights[i] = static_cast<std::int8_t>(std::clamp<long>(
        std::lround(weights[i] * RuntimeScale), -128, 127));
  runtime_bias = static_cast<std::int32_t>(std::lround(bias * RuntimeScale));
#endif
  is_loaded = true;
  return true;
}

bool load_from_environment(std::ostream* error) {
  if (is_loaded) return true;
  const char* path = std::getenv("KP_PROGRESS_SHADOW_FILE");
  return path && *path && load(path, error);
}

bool enabled() { return is_loaded || load_from_environment(nullptr); }

std::int32_t runtime_weight(IndexType index) {
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
  return static_cast<std::int32_t>(std::lround(weights[index] * RuntimeScale));
#else
  return runtime_weights[index];
#endif
}

ProgressShadowResult fresh(const Position& pos) {
  if (!enabled()) return {};
  Features::IndexList active[2];
  active_indices(pos, active);
  return from_sums(sum_float(active[BLACK]), sum_float(active[WHITE]));
}

ProgressShadowResult incremental(const Position& pos) {
  if (!enabled()) return {};
  ensure(pos);
  const auto& acc = pos.state()->accumulator;
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
  return from_sums(acc.kp_progress_accumulation[BLACK],
                   acc.kp_progress_accumulation[WHITE]);
#else
  const std::int64_t twice = 2LL * runtime_bias
      + acc.kp_progress_accumulation[BLACK]
      + acc.kp_progress_accumulation[WHITE];
  return {static_cast<double>(twice) / (2.0 * RuntimeScale),
          route_q(twice, RuntimeScale)};
#endif
}

ProgressShadowResult fixed_fresh(const Position& pos, int scale) {
  if (!enabled()) return {};
  Features::IndexList active[2];
  active_indices(pos, active);
  const auto twice = 2LL * static_cast<std::int64_t>(std::llround(bias * scale))
                   + sum_q(active[BLACK], scale) + sum_q(active[WHITE], scale);
  return {static_cast<double>(twice) / (2.0 * scale), route_q(twice, scale)};
}

static ProgressShadowResult runtime_fresh(const Position& pos) {
  Features::IndexList active[2];
  active_indices(pos, active);
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
  return from_sums(sum_float(active[BLACK]), sum_float(active[WHITE]));
#else
  const std::int64_t twice = 2LL * runtime_bias
      + sum_runtime(active[BLACK]) + sum_runtime(active[WHITE]);
  return {static_cast<double>(twice) / (2.0 * RuntimeScale),
          route_q(twice, RuntimeScale)};
#endif
}

void refresh_from_active(const Position& pos,
                         const Features::IndexList active[2]) {
  if (!enabled() || !is_b08(pos)) return;
  auto& acc = pos.state()->accumulator;
  for (int c = 0; c < COLOR_NB; ++c) {
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
    acc.kp_progress_accumulation[c] = sum_float(active[c]);
#else
    acc.kp_progress_accumulation[c] = sum_runtime(active[c]);
#endif
  }
  acc.computed_kp_progress = true;
}

void update_from_changed(const Position& pos,
                         const Features::IndexList removed[2],
                         const Features::IndexList added[2],
                         const bool reset[2]) {
  if (!enabled() || !is_b08(pos)) return;
  auto& acc = pos.state()->accumulator;
  const auto* previous = pos.state()->previous;
  Features::IndexList active[2];
  bool need_active = !previous || !previous->accumulator.computed_kp_progress
                  || reset[BLACK] || reset[WHITE];
  if (need_active) active_indices(pos, active);
  for (int c = 0; c < COLOR_NB; ++c) {
    if (!previous || !previous->accumulator.computed_kp_progress || reset[c]) {
      #if KP_PROGRESS_SHADOW_TABLE_BITS == 32
      acc.kp_progress_accumulation[c] = sum_float(active[c]);
      #else
      acc.kp_progress_accumulation[c] = sum_runtime(active[c]);
      #endif
    } else {
      auto value = previous->accumulator.kp_progress_accumulation[c];
#if KP_PROGRESS_SHADOW_TABLE_BITS == 32
      for (const auto index : removed[c]) value -= weights[index];
      for (const auto index : added[c]) value += weights[index];
#else
      for (const auto index : removed[c]) value -= runtime_weights[index];
      for (const auto index : added[c]) value += runtime_weights[index];
#endif
      acc.kp_progress_accumulation[c] = value;
    }
  }
  acc.computed_kp_progress = true;
}

void colocated_refresh(const Position& pos, const std::int32_t sums[2]) {
  if (!enabled() || !is_b08(pos)) return;
  auto& acc=pos.state()->accumulator;
  acc.kp_progress_accumulation[BLACK]=sums[BLACK];
  acc.kp_progress_accumulation[WHITE]=sums[WHITE];
  acc.computed_kp_progress=true;
}

void colocated_update(const Position& pos,
                      const std::int32_t removed_sums[2],
                      const std::int32_t added_sums[2],
                      const bool reset[2]) {
  if (!enabled() || !is_b08(pos)) return;
  auto& acc=pos.state()->accumulator;
  const auto* prev=pos.state()->previous;
  for(int c=0;c<COLOR_NB;++c) {
    if (!prev || !prev->accumulator.computed_kp_progress || reset[c])
      acc.kp_progress_accumulation[c]=added_sums[c];
    else
      acc.kp_progress_accumulation[c]=prev->accumulator.kp_progress_accumulation[c]
          - removed_sums[c] + added_sums[c];
  }
  acc.computed_kp_progress=true;
}

void observe(const Position& pos) {
  if (!enabled()) return;
  const auto result = incremental(pos);
  route_sink ^= result.bucket3;
}

void test_position(const Position& pos, const std::string& path,
                   std::uint64_t repeats, std::ostream& out) {
  if (!path.empty() && !load(path, &out)) return;
  const auto f = fresh(pos);
  const auto i = incremental(pos);
  Features::IndexList active[2]; active_indices(pos, active);
  out << std::setprecision(17)
      << "kp_progress float logit " << f.logit << " bucket " << f.bucket3
      << " incremental_logit " << i.logit << " incremental_bucket " << i.bucket3
      << " black_count " << active[BLACK].size()
      << " white_count " << active[WHITE].size();
  for (const int scale : {256, 512, 1024, 2048, 4096}) {
    const auto q = fixed_fresh(pos, scale);
    out << " q" << scale << "_logit " << q.logit
        << " q" << scale << "_bucket " << q.bucket3;
  }
  out << '\n';

  if (repeats == 0)
    return;

  using clock = std::chrono::steady_clock;
  double checksum = 0.0;
  auto begin = clock::now();
  for (std::uint64_t n = 0; n < repeats; ++n) checksum += fresh(pos).logit;
  auto middle = clock::now();
  int route_checksum = 0;
  for (std::uint64_t n = 0; n < repeats; ++n)
    route_checksum += route(f.logit);
  auto end = clock::now();
  out << "kp_progress bench fresh_ns "
      << std::chrono::duration<double, std::nano>(middle - begin).count() / repeats
      << " selection_ns "
      << std::chrono::duration<double, std::nano>(end - middle).count() / repeats
      << " checksum " << checksum << ' ' << route_checksum << '\n';
}

void self_test(const std::string& path, std::uint64_t games, int max_ply,
               std::ostream& out) {
  if (!path.empty() && !load(path, &out)) return;
  std::mt19937_64 random(0x1025a11ULL);
  std::uint64_t moves = 0, mismatch = 0;
  double max_diff = 0.0;
  std::array<std::uint64_t, 3> distribution{};
  std::chrono::nanoseconds update_time{};
  for (std::uint64_t game = 0; game < games; ++game) {
    Position pos;
    std::vector<StateInfo> states(static_cast<std::size_t>(max_ply) + 1);
    pos.set_hirate(&states[0]);
    if (is_b08(pos)) {
      Features::IndexList active[2]; active_indices(pos, active);
      refresh_from_active(pos, active);
    }
    for (int ply = 0; ply < max_ply; ++ply) {
      MoveList<LEGAL> legal(pos);
      if (legal.size() == 0) break;
      const Move move = legal.begin()[random() % legal.size()];
      pos.do_move(move, states[static_cast<std::size_t>(ply) + 1]);
      Features::IndexList removed[2], added[2];
      bool reset[2] = {false, false};
      RawFeatures::AppendChangedIndices(
          pos, Features::TriggerEvent::kFriendKingMoved,
          removed, added, reset);
      const auto begin = std::chrono::steady_clock::now();
      update_from_changed(pos, removed, added, reset);
      const auto end = std::chrono::steady_clock::now();
      if (!is_b08(pos)) {
        ++moves;
        if (pos.is_mated()) break;
        continue;
      }
      const auto inc = incremental(pos);
      // Compare against the exact table contract used by the runtime.  This
      // matters for int8, where the table is saturated after quantization.
      const auto ref = runtime_fresh(pos);
      update_time += end - begin;
      const double diff = std::abs(inc.logit - ref.logit);
      max_diff = std::max(max_diff, diff);
      mismatch += diff > 1e-6 || inc.bucket3 != ref.bucket3;
      ++distribution[inc.bucket3];
      ++moves;
      if (pos.is_mated()) break;
    }
  }
  out << std::setprecision(12)
      << "kp_progress selftest games " << games << " moves " << moves
      << " mismatch " << mismatch << " max_abs_diff " << max_diff
      << " incremental_ns_per_move "
      << (moves ? double(update_time.count()) / moves : 0.0)
      << " low " << distribution[0] << " mid " << distribution[1]
      << " high " << distribution[2] << '\n';
}

}  // namespace YaneuraOu::Eval::NNUE::NnueKpProgressShadow

#endif
