// NNUE評価関数に関するUSI拡張コマンド

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../engine.h"
#include "../../extra/all.h"
#include "../../evaluate.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(ENABLE_NNUE_TRACE) || defined(ENABLE_NNUE_BENCH)
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <system_error>
#endif

namespace YaneuraOu {
namespace Eval::NNUE {

namespace {

struct MoveAccuracyRecord {
  PackedSfen sfen;
  s16 score;
  u16 move;
  u16 game_ply;
  s8 game_result;
  u8 padding;
};

static_assert(sizeof(MoveAccuracyRecord) == 40,
              "sfenpack record must be exactly 40 bytes");

class NullStreamBuffer : public std::streambuf {
 protected:
  int_type overflow(int_type character) override {
    return traits_type::not_eof(character);
  }
};

class ScopedCoutRedirect {
 public:
  ScopedCoutRedirect() : original_buffer_(std::cout.rdbuf(&null_buffer_)) {}
  ~ScopedCoutRedirect() { std::cout.rdbuf(original_buffer_); }

 private:
  NullStreamBuffer null_buffer_;
  std::streambuf* original_buffer_;
};

class ScopedMoveAccuracyEngineState {
 public:
  ScopedMoveAccuracyEngineState(IEngine& engine, std::string& best_move)
      : engine_(engine),
        original_position_(engine.sfen()),
        original_threads_(engine.get_options()["Threads"]),
        original_multi_pv_(engine.get_options()["MultiPV"]),
        original_generate_all_legal_moves_(
            engine.get_options()["GenerateAllLegalMoves"]),
        original_own_book_(engine.get_options()["USI_OwnBook"]),
        original_draw_value_black_(engine.get_options()["DrawValueBlack"]),
        original_draw_value_white_(engine.get_options()["DrawValueWhite"]),
        original_entering_king_rule_(
            static_cast<std::string>(engine.get_options()["EnteringKingRule"])),
        original_bestmove_callback_(engine.get_on_bestmove()) {
    auto& options = engine_.get_options();
    options.set_option_if_exists("Threads", "1");
    options.set_option_if_exists("MultiPV", "1");
    options.set_option_if_exists("GenerateAllLegalMoves", "false");
    options.set_option_if_exists("USI_OwnBook", "false");
    options.set_option_if_exists("DrawValueBlack", "0");
    options.set_option_if_exists("DrawValueWhite", "0");
    options.set_option_if_exists("EnteringKingRule", EKR_STRINGS[EKR_27_POINT]);

    engine_.set_on_bestmove(
        [&best_move](std::string_view move, std::string_view) {
          best_move.assign(move.data(), move.size());
        });
  }

  ~ScopedMoveAccuracyEngineState() {
    engine_.wait_for_search_finished();
    engine_.set_on_bestmove(std::move(original_bestmove_callback_));

    auto& options = engine_.get_options();
    options.set_option_if_exists("Threads", std::to_string(original_threads_));
    options.set_option_if_exists("MultiPV", std::to_string(original_multi_pv_));
    options.set_option_if_exists("GenerateAllLegalMoves",
                                 original_generate_all_legal_moves_ ? "true" : "false");
    options.set_option_if_exists("USI_OwnBook", original_own_book_ ? "true" : "false");
    options.set_option_if_exists("DrawValueBlack",
                                 std::to_string(original_draw_value_black_));
    options.set_option_if_exists("DrawValueWhite",
                                 std::to_string(original_draw_value_white_));
    options.set_option_if_exists("EnteringKingRule", original_entering_king_rule_);
    engine_.set_position(original_position_, {});
  }

  bool is_configured() const {
    const auto& options = engine_.get_options();
    return static_cast<int64_t>(options["Threads"]) == 1
        && static_cast<int64_t>(options["MultiPV"]) == 1
        && static_cast<int64_t>(options["GenerateAllLegalMoves"]) == 0
        && static_cast<int64_t>(options["USI_OwnBook"]) == 0
        && static_cast<int64_t>(options["DrawValueBlack"]) == 0
        && static_cast<int64_t>(options["DrawValueWhite"]) == 0
        && static_cast<std::string>(options["EnteringKingRule"])
               == EKR_STRINGS[EKR_27_POINT];
  }

 private:
  IEngine& engine_;
  std::string original_position_;
  int64_t original_threads_;
  int64_t original_multi_pv_;
  bool original_generate_all_legal_moves_;
  bool original_own_book_;
  int64_t original_draw_value_black_;
  int64_t original_draw_value_white_;
  std::string original_entering_king_rule_;
  std::function<void(std::string_view, std::string_view)>
      original_bestmove_callback_;
};

void TestMoveAccuracy(IEngine& engine, std::istream& stream) {
  std::string file_name;
  stream >> file_name;
  if (file_name.empty()) {
    std::cout << "error: sfenpack file path is required" << std::endl;
    return;
  }

  std::ifstream input(file_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open sfenpack file: " << file_name << std::endl;
    return;
  }

  std::uint64_t total_records = 0;
  std::uint64_t tested_positions = 0;
  std::uint64_t correct_moves = 0;
  std::string error_message;
  std::string best_move_text;

  {
    ScopedCoutRedirect suppress_search_output;
    ScopedMoveAccuracyEngineState engine_state(engine, best_move_text);

    if (!engine_state.is_configured()) {
      error_message = "failed to configure engine for move accuracy measurement";
    } else {
      MoveAccuracyRecord packed_record;
      while (input.read(reinterpret_cast<char*>(&packed_record),
                        sizeof(packed_record))) {
        ++total_records;

        const int score = packed_record.score;
        if (30000 < std::abs(score) || packed_record.game_result == 0)
          continue;

        Position decoded_position;
        StateInfo decoded_state;
        if (decoded_position
                .set_from_packed_sfen(packed_record.sfen, &decoded_state, false)
                .is_not_ok())
          continue;

        if (MoveList<LEGAL>(decoded_position).size() == 0) {
          error_message = "no legal move in a tested sfenpack position";
          break;
        }

        engine.set_position(decoded_position.sfen(), {});
        best_move_text.clear();

        Search::LimitsType limits;
        limits.depth = 1;
        limits.startTime = now();
        engine.go(limits);
        engine.wait_for_search_finished();

        if (best_move_text.empty()) {
          error_message = "depth=1 search did not return a best move";
          break;
        }

        const Move16 best_move = Move16::from_string(best_move_text);
        if (best_move == Move16::none() || best_move == Move16::resign()
            || best_move == Move16::win()) {
          error_message = "depth=1 search returned no comparable best move: "
                        + best_move_text;
          break;
        }

        ++tested_positions;
        const u16 teacher_move = packed_record.move;
        if (best_move.to_u16() == teacher_move)
          ++correct_moves;
      }
    }
  }

  if (error_message.empty() && input.bad())
    error_message = "failed while reading sfenpack file";
  else if (error_message.empty() && input.gcount() != 0)
    error_message = "sfenpack file ends with an incomplete record";

  if (!error_message.empty()) {
    std::cout << "error: " << error_message << std::endl;
    return;
  }
  if (total_records == 0) {
    std::cout << "error: sfenpack file is empty" << std::endl;
    return;
  }
  if (tested_positions == 0) {
    std::cout << "error: no positions passed the sfenpack filters" << std::endl;
    return;
  }

  const double accuracy =
      100.0 * static_cast<double>(correct_moves)
      / static_cast<double>(tested_positions);
  std::ostringstream accuracy_text;
  accuracy_text << std::fixed << std::setprecision(3) << accuracy;

  std::cout << "tested positions = " << tested_positions << std::endl;
  std::cout << "correct moves    = " << correct_moves << std::endl;
  std::cout << "accuracy=" << accuracy_text.str() << "%" << std::endl;
}

// 主に差分計算に関するRawFeaturesのテスト
void TestFeatures(Position& pos) {
  const std::uint64_t num_games = 1000;
  StateInfo si;
  pos.set_hirate(&si);
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  int ply; // 初期局面からの手数

  PRNG prng(20171128);

  std::uint64_t num_moves = 0;
  std::vector<std::uint64_t> num_updates(kRefreshTriggers.size() + 1);
  std::vector<std::uint64_t> num_resets(kRefreshTriggers.size());
  constexpr IndexType kUnknown = -1;
  std::vector<IndexType> trigger_map(RawFeatures::kDimensions, kUnknown);
  auto make_index_sets = [&](const Position& pos) {
    std::vector<std::vector<std::set<IndexType>>> index_sets(
        kRefreshTriggers.size(), std::vector<std::set<IndexType>>(2));
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : COLOR) {
        for (const auto index : active_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT(index_sets[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          index_sets[i][perspective].insert(index);
          trigger_map[index] = i;
        }
      }
    }
    return index_sets;
  };
  auto update_index_sets = [&](const Position& pos, auto* index_sets) {
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : COLOR) {
        if (reset[perspective]) {
          (*index_sets)[i][perspective].clear();
          ++num_resets[i];
        } else {
          for (const auto index : removed_indices[perspective]) {
            ASSERT(index < RawFeatures::kDimensions);
            ASSERT((*index_sets)[i][perspective].count(index) == 1);
            ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
            (*index_sets)[i][perspective].erase(index);
            ++num_updates.back();
            ++num_updates[i];
            trigger_map[index] = i;
          }
        }
        for (const auto index : added_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT((*index_sets)[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          (*index_sets)[i][perspective].insert(index);
          ++num_updates.back();
          ++num_updates[i];
          trigger_map[index] = i;
        }
      }
    }
  };

  std::cout << "feature set: " << RawFeatures::GetName()
            << "[" << RawFeatures::kDimensions << "]" << std::endl;
  std::cout << "start testing with random games";

  for (std::uint64_t i = 0; i < num_games; ++i) {
    auto index_sets = make_index_sets(pos);
    for (ply = 0; ply < MAX_PLY; ++ply) {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

      // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[prng.rand(mg.size())];
      pos.do_move(m, state[ply]);

      ++num_moves;
      update_index_sets(pos, &index_sets);
      ASSERT(index_sets == make_index_sets(pos));
    }

    pos.set_hirate(&si);

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 100) == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves, "
            << num_updates.back() << " updates, "
            << (1.0 * num_updates.back() / num_moves)
            << " updates per move" << std::endl;
  std::size_t num_observed_indices = 0;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    const auto count = std::count(trigger_map.begin(), trigger_map.end(), i);
    num_observed_indices += count;
    std::cout << "TriggerEvent(" << static_cast<int>(kRefreshTriggers[i])
              << "): " << count << " features ("
              << (100.0 * count / RawFeatures::kDimensions) << "%), "
              << num_updates[i] << " updates ("
              << (1.0 * num_updates[i] / num_moves) << " per move), "
              << num_resets[i] << " resets ("
              << (100.0 * num_resets[i] / num_moves) << "%)"
              << std::endl;
  }
  std::cout << "observed " << num_observed_indices << " ("
            << (100.0 * num_observed_indices / RawFeatures::kDimensions)
            << "% of " << RawFeatures::kDimensions
            << ") features" << std::endl;
}

// NNUE Accumulatorの差分更新結果と全計算結果を比較するテスト
void TestAccumulator(Position& pos) {
  const std::uint64_t num_games = 1000;
  const int MAX_PLY = 256;

  StateInfo si;
  pos.set_hirate(&si);
  StateInfo state[MAX_PLY];

  PRNG prng(20171128);
  std::uint64_t num_moves = 0;

  auto print_state_failure = [&](const std::uint64_t game, const int ply,
                                 const char* reason, const Move* move) {
    std::cout << std::endl
              << "NNUE accumulator test failed" << std::endl
              << "  game              : " << (game + 1) << std::endl
              << "  ply               : " << (ply + 1) << std::endl
              << "  SFEN              : " << pos.sfen() << std::endl
              << "  move              : ";
    if (move)
      std::cout << *move;
    else
      std::cout << "(root)";
    std::cout << std::endl
              << "  reason            : " << reason << std::endl;
  };

  auto print_value_failure = [&](const std::uint64_t game, const int ply,
                                 const Move move, const Color perspective,
                                 const char* target, const std::size_t index,
                                 const std::int64_t incremental_value,
                                 const std::int64_t full_value,
                                 const std::size_t trigger_index,
                                 const bool is_main) {
    std::cout << std::endl
              << "NNUE accumulator test failed" << std::endl
              << "  game              : " << (game + 1) << std::endl
              << "  ply               : " << (ply + 1) << std::endl
              << "  SFEN              : " << pos.sfen() << std::endl
              << "  move              : " << move << std::endl
              << "  perspective       : "
              << (perspective == BLACK ? "BLACK" : "WHITE") << std::endl
              << "  target            : " << target << std::endl;
    if (is_main) {
      std::cout << "  trigger_index     : " << trigger_index << std::endl
                << "  TriggerEvent      : "
                << static_cast<int>(kRefreshTriggers[trigger_index]) << std::endl
                << "  dimension index   : " << index << std::endl;
    } else {
      std::cout << "  index             : " << index << std::endl;
    }
    std::cout << "  incremental value : " << incremental_value << std::endl
              << "  full refresh value: " << full_value << std::endl
              << "  difference        : "
              << (incremental_value - full_value) << std::endl;
  };

  std::cout << "start testing accumulator with random games";

  for (std::uint64_t game = 0; game < num_games; ++game) {
    if (!pos.state()->accumulator.computed_accumulation) {
      print_state_failure(game, -1, "root accumulator is not computed", nullptr);
      std::cout << "failed." << std::endl;
      return;
    }

    for (int ply = 0; ply < MAX_PLY; ++ply) {
      MoveList<LEGAL_ALL> mg(pos);
      if (mg.size() == 0)
        break;

      const Move move = mg.begin()[prng.rand(mg.size())];
      pos.do_move(move, state[ply]);
      ++num_moves;

      auto* const current = pos.state();
      if (current->accumulator.computed_accumulation) {
        print_state_failure(game, ply,
                            "current accumulator is already computed after do_move",
                            &move);
        std::cout << "failed." << std::endl;
        return;
      }
      if (current->previous == nullptr) {
        print_state_failure(game, ply, "current StateInfo has no previous state", &move);
        std::cout << "failed." << std::endl;
        return;
      }
      if (!current->previous->accumulator.computed_accumulation) {
        print_state_failure(game, ply, "previous accumulator is not computed", &move);
        std::cout << "failed." << std::endl;
        return;
      }

      ::YaneuraOu::Eval::evaluate_with_no_return(pos);
      if (!current->accumulator.computed_accumulation) {
        print_state_failure(game, ply, "incremental update did not compute accumulator",
                            &move);
        std::cout << "failed." << std::endl;
        return;
      }

      const Accumulator incremental = current->accumulator;

      // compute_eval()はComputeScore(pos, true)を呼び、同じ局面の
      // Accumulatorをfull refreshで再計算する。
      ::YaneuraOu::Eval::compute_eval(pos);
      if (!current->accumulator.computed_accumulation) {
        print_state_failure(game, ply, "full refresh did not compute accumulator", &move);
        std::cout << "failed." << std::endl;
        return;
      }

      const auto& full = current->accumulator;
      for (const Color perspective : {BLACK, WHITE}) {
        for (std::size_t trigger_index = 0;
             trigger_index < kRefreshTriggers.size(); ++trigger_index) {
          for (IndexType index = 0; index < kTransformedFeatureDimensions; ++index) {
            const std::int64_t incremental_value =
                incremental.accumulation[perspective][trigger_index][index];
            const std::int64_t full_value =
                full.accumulation[perspective][trigger_index][index];
            if (incremental_value != full_value) {
              print_value_failure(game, ply, move, perspective, "main accumulation",
                                  index, incremental_value, full_value,
                                  trigger_index, true);
              std::cout << "failed." << std::endl;
              return;
            }
          }
        }

        const auto& incremental_factors = incremental.factors[perspective];
        const auto& full_factors = full.factors[perspective];
        struct FactorComparison {
          const char* target;
          const std::int64_t* incremental_values;
          const std::int64_t* full_values;
        };
        const FactorComparison factor_comparisons[] = {
            {"halfka.sum_v", incremental_factors.halfka.sum_v,
             full_factors.halfka.sum_v},
            {"halfka.sum_v2", incremental_factors.halfka.sum_v2,
             full_factors.halfka.sum_v2},
            {"ksdg.sum_v", incremental_factors.ksdg.sum_v,
             full_factors.ksdg.sum_v},
            {"ksdg.sum_v2", incremental_factors.ksdg.sum_v2,
             full_factors.ksdg.sum_v2},
        };

        for (const auto& comparison : factor_comparisons) {
          for (std::size_t index = 0; index < 32; ++index) {
            const std::int64_t incremental_value = comparison.incremental_values[index];
            const std::int64_t full_value = comparison.full_values[index];
            if (incremental_value != full_value) {
              print_value_failure(game, ply, move, perspective, comparison.target,
                                  index, incremental_value, full_value, 0, false);
              std::cout << "failed." << std::endl;
              return;
            }
          }
        }
      }
    }

    pos.set_hirate(&si);
    if ((game % 100) == 0)
      std::cout << "." << std::flush;
  }

  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves" << std::endl;
}

#if defined(ENABLE_NNUE_BENCH)

constexpr std::uint64_t kNnueBenchSeed = 20171128;
constexpr std::uint64_t kNnueBenchWarmupGames = 8;
constexpr std::uint64_t kNnueBenchMeasuredGames = 64;
constexpr int kNnueBenchMaxPly = 128;

using NnueBenchClock = std::chrono::steady_clock;

struct NnueBenchTiming {
  std::uint64_t calls = 0;
  double nanoseconds = 0.0;
};

struct NnueBenchSummary {
  double median = 0.0;
  double mean = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
};

struct NnueBenchSamples {
  std::vector<double> ns_per_call;
  std::uint64_t calls_per_repeat = 0;
  double total_nanoseconds = 0.0;

  void Add(const NnueBenchTiming& timing) {
    if (timing.calls == 0)
      return;
    if (calls_per_repeat == 0)
      calls_per_repeat = timing.calls;
    ns_per_call.push_back(
        timing.nanoseconds / static_cast<double>(timing.calls));
    total_nanoseconds += timing.nanoseconds;
  }
};

bool ReadNnueBenchRepeatCount(std::istream& stream,
                              std::uint64_t& repeat_count) {
  repeat_count = 1;
  std::string token;
  if (!(stream >> token)) {
    stream.clear();
    return true;
  }

  if (!token.empty() && token.front() == '-') {
    std::cout << "error: benchmark repeat count must be a positive integer"
              << std::endl;
    return false;
  }

  std::uint64_t value = 0;
  const char* const begin = token.data();
  const char* const end = begin + token.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end || value == 0) {
    std::cout << "error: benchmark repeat count must be a positive integer"
              << std::endl;
    return false;
  }
  repeat_count = value;
  return true;
}

NnueBenchSummary SummarizeNnueBenchSamples(
    const NnueBenchSamples& samples) {
  NnueBenchSummary summary;
  if (samples.ns_per_call.empty())
    return summary;

  std::vector<double> sorted = samples.ns_per_call;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  summary.median = sorted.size() % 2 == 0
      ? (sorted[middle - 1] + sorted[middle]) / 2.0
      : sorted[middle];
  summary.minimum = sorted.front();
  summary.maximum = sorted.back();
  for (const double value : sorted)
    summary.mean += value;
  summary.mean /= static_cast<double>(sorted.size());
  return summary;
}

struct FtChangeStatistics {
  std::uint64_t perspective_samples = 0;
  std::uint64_t non_reset_samples = 0;
  std::uint64_t changed_non_reset_samples = 0;
  std::uint64_t reset_samples = 0;
  std::uint64_t one_removed_one_added = 0;
  std::uint64_t removed_features = 0;
  std::uint64_t added_features = 0;
  std::uint64_t halfka_removed = 0;
  std::uint64_t halfka_added = 0;
  std::uint64_t ksdg_removed = 0;
  std::uint64_t ksdg_added = 0;
};

enum class FtBenchOperation {
  IncrementalUpdate,
  ForcedRefresh,
  TransformOnly,
};

void MixNnueBenchChecksum(std::uint64_t& checksum, const std::int64_t value) {
  checksum ^= static_cast<std::uint64_t>(value);
  checksum *= UINT64_C(1099511628211);
}

void ChecksumAccumulator(const Position& pos, std::uint64_t& checksum) {
  const auto& accumulator = pos.state()->accumulator;
  for (const Color perspective : {BLACK, WHITE}) {
    for (std::size_t trigger = 0; trigger < kRefreshTriggers.size(); ++trigger)
      for (IndexType index = 0; index < kTransformedFeatureDimensions; ++index)
        MixNnueBenchChecksum(
            checksum, accumulator.accumulation[perspective][trigger][index]);

    const auto& factors = accumulator.factors[perspective];
    for (IndexType index = 0; index < 32; ++index) {
      MixNnueBenchChecksum(checksum, factors.halfka.sum_v[index]);
      MixNnueBenchChecksum(checksum, factors.halfka.sum_v2[index]);
      MixNnueBenchChecksum(checksum, factors.ksdg.sum_v[index]);
      MixNnueBenchChecksum(checksum, factors.ksdg.sum_v2[index]);
    }
  }
}

int NnueBenchMaterialBucket(const Position& pos) {
  // Keep this benchmark-only calculation identical to stack_index_for_nnue().
  constexpr int bucket_by_material[24] = {
      0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8,
      8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11};
  return bucket_by_material[std::min(
      (std::abs(pos.state()->materialValue) + 99) / 100, 23)];
}

void CollectFtChangeStatistics(const Position& pos,
                               FtChangeStatistics& statistics) {
  for (IndexType trigger = 0; trigger < kRefreshTriggers.size(); ++trigger) {
    Features::IndexList removed_indices[2], added_indices[2];
    bool reset[2];
    RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[trigger],
                                      removed_indices, added_indices, reset);

    for (const Color perspective : {BLACK, WHITE}) {
      ++statistics.perspective_samples;
      if (reset[perspective]) {
        ++statistics.reset_samples;
      } else {
        ++statistics.non_reset_samples;
        if (removed_indices[perspective].size() != 0
            || added_indices[perspective].size() != 0)
          ++statistics.changed_non_reset_samples;
        if (removed_indices[perspective].size() == 1
            && added_indices[perspective].size() == 1)
          ++statistics.one_removed_one_added;

        statistics.removed_features += removed_indices[perspective].size();
        if (trigger == 0) {
          for (const IndexType index : removed_indices[perspective]) {
            if (FeatureTransformer::BenchmarkIsHalfKaIndex(index))
              ++statistics.halfka_removed;
            else
              ++statistics.ksdg_removed;
          }
        }
      }

      statistics.added_features += added_indices[perspective].size();
      if (trigger == 0) {
        for (const IndexType index : added_indices[perspective]) {
          if (FeatureTransformer::BenchmarkIsHalfKaIndex(index))
            ++statistics.halfka_added;
          else
            ++statistics.ksdg_added;
        }
      }
    }
  }
}

NnueBenchTiming RunFtBenchmarkPass(const FtBenchOperation operation,
                                   const std::uint64_t num_games,
                                   FtChangeStatistics* const statistics,
                                   std::uint64_t& checksum) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  NnueBenchTiming timing;

  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> transformed{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> diff_transformed{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> abs_transformed{};

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);

    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;

      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      if (operation == FtBenchOperation::TransformOnly) {
        if (!feature_transformer->UpdateAccumulatorIfPossible(pos)) {
          std::cout << "error: benchmark could not prepare incremental accumulator"
                    << std::endl;
          return {};
        }
      }

      const auto begin = NnueBenchClock::now();
      if (operation == FtBenchOperation::IncrementalUpdate) {
        if (!feature_transformer->UpdateAccumulatorIfPossible(pos)) {
          std::cout << "error: benchmark incremental update was unavailable"
                    << std::endl;
          return {};
        }
      } else if (operation == FtBenchOperation::ForcedRefresh) {
        feature_transformer->BenchmarkRefreshAccumulator(pos);
      } else {
        feature_transformer->Transform(
            pos, transformed.data(), diff_transformed.data(),
            abs_transformed.data(), false, NnueBenchMaterialBucket(pos));
      }
      const auto end = NnueBenchClock::now();

      timing.nanoseconds +=
          std::chrono::duration<double, std::nano>(end - begin).count();
      ++timing.calls;

      if (statistics != nullptr)
        CollectFtChangeStatistics(pos, *statistics);

      if (operation == FtBenchOperation::TransformOnly) {
        for (const auto value : transformed)
          MixNnueBenchChecksum(checksum, value);
        for (const auto value : diff_transformed)
          MixNnueBenchChecksum(checksum, value);
        for (const auto value : abs_transformed)
          MixNnueBenchChecksum(checksum, value);
      } else {
        ChecksumAccumulator(pos, checksum);
      }
    }
  }

  return timing;
}

void PrintNnueBenchSamples(const char* const name,
                           const NnueBenchSamples& samples) {
  const NnueBenchSummary summary = SummarizeNnueBenchSamples(samples);
  const double median_calls_per_second = summary.median == 0.0
      ? 0.0
      : 1.0e9 / summary.median;

  std::cout << name << std::endl
            << "  repeats          : " << samples.ns_per_call.size() << std::endl
            << "  calls/repeat     : " << samples.calls_per_repeat << std::endl
            << "  total time       : " << std::fixed << std::setprecision(3)
            << samples.total_nanoseconds / 1.0e6 << " ms" << std::endl
            << "  median ns/call   : " << std::setprecision(1) << summary.median
            << std::endl
            << "  mean ns/call     : " << summary.mean << std::endl
            << "  min ns/call      : " << summary.minimum << std::endl
            << "  max ns/call      : " << summary.maximum << std::endl
            << "  median calls/sec : " << median_calls_per_second << std::endl;
}

void TestFeatureTransformerBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: FeatureTransformer]" << std::endl
            << "  seed           : " << kNnueBenchSeed << std::endl
            << "  warm-up games  : " << kNnueBenchWarmupGames << std::endl
            << "  measured games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game   : " << kNnueBenchMaxPly << std::endl
            << "  repeats        : " << repeat_count << std::endl;

  std::uint64_t checksum = UINT64_C(14695981039346656037);
  FtChangeStatistics statistics;
  NnueBenchSamples incremental;
  NnueBenchSamples refresh;
  NnueBenchSamples transform;

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    RunFtBenchmarkPass(FtBenchOperation::IncrementalUpdate,
                       kNnueBenchWarmupGames, nullptr, checksum);
    incremental.Add(RunFtBenchmarkPass(
        FtBenchOperation::IncrementalUpdate, kNnueBenchMeasuredGames,
        repeat == 0 ? &statistics : nullptr, checksum));

    RunFtBenchmarkPass(FtBenchOperation::ForcedRefresh,
                       kNnueBenchWarmupGames, nullptr, checksum);
    refresh.Add(RunFtBenchmarkPass(
        FtBenchOperation::ForcedRefresh, kNnueBenchMeasuredGames, nullptr,
        checksum));

    RunFtBenchmarkPass(FtBenchOperation::TransformOnly,
                       kNnueBenchWarmupGames, nullptr, checksum);
    transform.Add(RunFtBenchmarkPass(
        FtBenchOperation::TransformOnly, kNnueBenchMeasuredGames, nullptr,
        checksum));
  }

  PrintNnueBenchSamples("incremental accumulator update", incremental);
  PrintNnueBenchSamples("forced full refresh", refresh);
  PrintNnueBenchSamples("Transform (precomputed accumulator)", transform);

  const auto percentage = [](const std::uint64_t numerator,
                             const std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : 100.0 * static_cast<double>(numerator)
              / static_cast<double>(denominator);
  };

  std::cout << "[incremental feature statistics]" << std::endl
            << "  trigger/perspective samples : "
            << statistics.perspective_samples << std::endl
            << "  non-reset samples           : "
            << statistics.non_reset_samples << std::endl
            << "  changed non-reset samples   : "
            << statistics.changed_non_reset_samples << std::endl
            << "  reset samples               : "
            << statistics.reset_samples << std::endl
            << "  removed features            : "
            << statistics.removed_features << std::endl
            << "  added features              : "
            << statistics.added_features << std::endl
            << "  removed=1 && added=1        : "
            << statistics.one_removed_one_added << " / "
            << statistics.non_reset_samples << " ("
            << std::fixed << std::setprecision(2)
            << percentage(statistics.one_removed_one_added,
                          statistics.non_reset_samples)
            << "% of non-reset, "
            << percentage(statistics.one_removed_one_added,
                          statistics.changed_non_reset_samples)
            << "% of changed non-reset)" << std::endl
			<< "  Main fused path uses        : "
			<< statistics.one_removed_one_added << std::endl
            << "  HalfKA removed / added      : "
            << statistics.halfka_removed << " / " << statistics.halfka_added
            << std::endl
            << "  KSDG3 removed / added       : "
            << statistics.ksdg_removed << " / " << statistics.ksdg_added
            << std::endl
            << "  checksum                    : 0x" << std::hex << checksum
            << std::dec << std::endl;
}

struct alignas(kCacheLineSize) NetworkBenchCase {
  std::array<FeatureTransformer::OutputType,
             FeatureTransformer::kOutputDimensions> transformed;
  std::array<FeatureTransformer::OutputType, 128> diff_transformed;
  std::array<FeatureTransformer::OutputType, 128> abs_transformed;
  std::array<std::uint8_t, 384> router_input;
  int material_bucket = 0;
  int selected_bucket = 0;
};

int SelectNnueBenchBucket(const std::int32_t* const router_output) {
  int selected_bucket = 0;
  std::int32_t max_score = router_output[0];
  for (int bucket = 1; bucket < kLayerStacks; ++bucket) {
    if (router_output[bucket] > max_score) {
      max_score = router_output[bucket];
      selected_bucket = bucket;
    }
  }
  return selected_bucket;
}

void FillNnueBenchRouterInput(NetworkBenchCase& sample) {
  for (int index = 0; index < 128; ++index) {
    const int abs_value = sample.abs_transformed[index];
    sample.router_input[index] = static_cast<std::uint8_t>(
        std::clamp((abs_value - 64) * 2, 0, 127));
    sample.router_input[index + 128] = sample.diff_transformed[index];
    sample.router_input[index + 256] = sample.transformed[index];
  }
}

std::vector<NetworkBenchCase> MakeNnueNetworkBenchCorpus() {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  std::vector<NetworkBenchCase> corpus;
  corpus.reserve(kNnueBenchMeasuredGames * kNnueBenchMaxPly);

  alignas(kCacheLineSize) std::int32_t router_output[32];

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;

      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      NetworkBenchCase sample{};
      sample.material_bucket = NnueBenchMaterialBucket(pos);
      feature_transformer->Transform(
          pos, sample.transformed.data(), sample.diff_transformed.data(),
          sample.abs_transformed.data(), false, sample.material_bucket);
      FillNnueBenchRouterInput(sample);
      router->PropagatePrefix<12>(sample.router_input.data(), router_output);
      sample.selected_bucket = SelectNnueBenchBucket(router_output);
      corpus.emplace_back(std::move(sample));
    }
  }

  return corpus;
}

enum class NetworkBenchOperation {
  RouterOnly,
  SelectedNetworkOnly,
  RouterAndNetwork,
};

enum class NetworkBenchImplementation {
  Full,
  Prefix,
};

template<NetworkBenchImplementation Implementation>
NnueBenchTiming MeasureNnueNetworkCorpus(
    const std::vector<NetworkBenchCase>& corpus,
    const NetworkBenchOperation operation, std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t router_output[32];
  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    int selected_bucket = sample.selected_bucket;
    if (operation != NetworkBenchOperation::SelectedNetworkOnly) {
      if constexpr (Implementation == NetworkBenchImplementation::Prefix)
        router->PropagatePrefix<12>(sample.router_input.data(), router_output);
      else
        router->Propagate(sample.router_input.data(), router_output);
      selected_bucket = SelectNnueBenchBucket(router_output);
      MixNnueBenchChecksum(checksum, selected_bucket);
      MixNnueBenchChecksum(checksum, router_output[selected_bucket]);
    }

    if (operation != NetworkBenchOperation::RouterOnly) {
#if defined(SFNNwoPSQT)
      const auto output = network[selected_bucket]->Propagate<
          Implementation == NetworkBenchImplementation::Prefix>(
#else
      const auto output = network->Propagate<
          Implementation == NetworkBenchImplementation::Prefix>(
#endif
          sample.transformed.data(), sample.diff_transformed.data(),
          sample.abs_transformed.data(), sample.material_bucket,
          network_buffer);
      MixNnueBenchChecksum(checksum, output[0]);
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

void TestNetworkBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Router / Network]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl;

  const auto corpus = MakeNnueNetworkBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: NNUE network benchmark corpus is empty" << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size() << std::endl;

  std::uint64_t checksum = UINT64_C(14695981039346656037);
  NnueBenchSamples router_samples;
  NnueBenchSamples selected_network_samples;
  NnueBenchSamples combined_samples;

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
        corpus, NetworkBenchOperation::RouterOnly, checksum);
    router_samples.Add(
        MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
            corpus, NetworkBenchOperation::RouterOnly, checksum));

    MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
        corpus, NetworkBenchOperation::SelectedNetworkOnly, checksum);
    selected_network_samples.Add(
        MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
            corpus, NetworkBenchOperation::SelectedNetworkOnly, checksum));

    MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
        corpus, NetworkBenchOperation::RouterAndNetwork, checksum);
    combined_samples.Add(
        MeasureNnueNetworkCorpus<NetworkBenchImplementation::Prefix>(
            corpus, NetworkBenchOperation::RouterAndNetwork, checksum));
  }

  PrintNnueBenchSamples("Router (FC + argmax)", router_samples);
  PrintNnueBenchSamples("selected Network::Propagate",
                        selected_network_samples);
  PrintNnueBenchSamples("Router + selected Network::Propagate",
                        combined_samples);
  std::cout << "  checksum    : 0x" << std::hex << checksum << std::dec
            << std::endl;
}

template<NetworkBenchImplementation Implementation>
NnueBenchTiming MeasureNnueNetworkCorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus,
    const NetworkBenchOperation operation, std::uint64_t& checksum) {
  MeasureNnueNetworkCorpus<Implementation>(corpus, operation, checksum);
  return MeasureNnueNetworkCorpus<Implementation>(corpus, operation,
                                                   checksum);
}

void PrintNnueBenchComparison(const char* const name,
                              const NnueBenchSamples& full,
                              const NnueBenchSamples& prefix) {
  const NnueBenchSummary full_summary = SummarizeNnueBenchSamples(full);
  const NnueBenchSummary prefix_summary = SummarizeNnueBenchSamples(prefix);
  const double difference = prefix_summary.median - full_summary.median;
  const double improvement = full_summary.median == 0.0
      ? 0.0
      : (full_summary.median - prefix_summary.median)
            * 100.0 / full_summary.median;

  std::cout << name << std::endl
            << "  full" << std::endl
            << "    median ns/call : " << std::fixed << std::setprecision(1)
            << full_summary.median << std::endl
            << "    mean ns/call   : " << full_summary.mean << std::endl
            << "    min ns/call    : " << full_summary.minimum << std::endl
            << "    max ns/call    : " << full_summary.maximum << std::endl
            << "  prefix" << std::endl
            << "    median ns/call : " << prefix_summary.median << std::endl
            << "    mean ns/call   : " << prefix_summary.mean << std::endl
            << "    min ns/call    : " << prefix_summary.minimum << std::endl
            << "    max ns/call    : " << prefix_summary.maximum << std::endl
            << "  difference (prefix - full) : " << difference
            << " ns/call" << std::endl
            << "  improvement               : " << std::setprecision(2)
            << improvement << "%" << std::endl;
}

void TestNetworkBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Full / Prefix comparison]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : even=full,prefix odd=prefix,full"
            << std::endl;

  const auto corpus = MakeNnueNetworkBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: NNUE network benchmark corpus is empty" << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  constexpr std::array<NetworkBenchOperation, 3> operations = {
      NetworkBenchOperation::RouterOnly,
      NetworkBenchOperation::SelectedNetworkOnly,
      NetworkBenchOperation::RouterAndNetwork};
  constexpr std::array<const char*, 3> names = {
      "Router (FC + argmax)",
      "selected Network::Propagate",
      "Router + selected Network::Propagate"};

  std::array<NnueBenchSamples, 3> full_samples;
  std::array<NnueBenchSamples, 3> prefix_samples;
  std::uint64_t full_checksum = UINT64_C(14695981039346656037);
  std::uint64_t prefix_checksum = UINT64_C(14695981039346656037);

  for (std::size_t operation_index = 0;
       operation_index < operations.size(); ++operation_index) {
    const NetworkBenchOperation operation = operations[operation_index];
    for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
      if ((repeat & 1) == 0) {
        full_samples[operation_index].Add(
            MeasureNnueNetworkCorpusAfterWarmup<
                NetworkBenchImplementation::Full>(
                    corpus, operation, full_checksum));
        prefix_samples[operation_index].Add(
            MeasureNnueNetworkCorpusAfterWarmup<
                NetworkBenchImplementation::Prefix>(
                    corpus, operation, prefix_checksum));
      } else {
        prefix_samples[operation_index].Add(
            MeasureNnueNetworkCorpusAfterWarmup<
                NetworkBenchImplementation::Prefix>(
                    corpus, operation, prefix_checksum));
        full_samples[operation_index].Add(
            MeasureNnueNetworkCorpusAfterWarmup<
                NetworkBenchImplementation::Full>(
                    corpus, operation, full_checksum));
      }
    }
  }

  for (std::size_t operation_index = 0;
       operation_index < operations.size(); ++operation_index)
    PrintNnueBenchComparison(names[operation_index],
                             full_samples[operation_index],
                             prefix_samples[operation_index]);

  std::cout << "  full checksum   : 0x" << std::hex << full_checksum
            << std::endl
            << "  prefix checksum : 0x" << prefix_checksum << std::dec
            << std::endl
            << "  checksum match  : "
            << (full_checksum == prefix_checksum ? "yes" : "NO")
            << std::endl;
}

enum class NetworkStageBenchOperation {
  Phase,
  FcDiff,
  FcAbs,
  DiffRmsNorm,
  DiffQuantize,
  AbsSigmoidGate,
  AbsGateQuantize,
  AbsSquared,
  MainFc0,
  MainGate,
  MainSqrClippedRelu,
  MainClippedRelu,
  Lca,
  Cross,
  L2Assembly,
  Fc1,
  Ac1,
  Fc2,
  Blend,
};

constexpr std::array<NetworkStageBenchOperation, 19>
    kNetworkStageBenchOperations = {
        NetworkStageBenchOperation::Phase,
        NetworkStageBenchOperation::FcDiff,
        NetworkStageBenchOperation::FcAbs,
        NetworkStageBenchOperation::DiffRmsNorm,
        NetworkStageBenchOperation::DiffQuantize,
        NetworkStageBenchOperation::AbsSigmoidGate,
        NetworkStageBenchOperation::AbsGateQuantize,
        NetworkStageBenchOperation::AbsSquared,
        NetworkStageBenchOperation::MainFc0,
        NetworkStageBenchOperation::MainGate,
        NetworkStageBenchOperation::MainSqrClippedRelu,
        NetworkStageBenchOperation::MainClippedRelu,
        NetworkStageBenchOperation::Lca,
        NetworkStageBenchOperation::Cross,
        NetworkStageBenchOperation::L2Assembly,
        NetworkStageBenchOperation::Fc1,
        NetworkStageBenchOperation::Ac1,
        NetworkStageBenchOperation::Fc2,
        NetworkStageBenchOperation::Blend};

constexpr std::array<const char*, kNetworkStageBenchOperations.size()>
    kNetworkStageBenchNames = {
        "phase input + phase_proj + scales",
        "FM fc_diff",
        "FM fc_abs",
        "FM Diff RMSNorm",
        "FM Diff quantize + clamp",
        "FM Abs sigmoid + value gate",
        "FM Abs gated-value quantize + clamp",
        "FM Abs squared",
        "Main fc_0",
        "Main gate + clamp",
        "Main SqrClippedReLU",
        "Main ClippedReLU",
        "LCA",
        "Cross",
        "L2 input assembly",
        "fc_1",
        "ac_1",
        "fc_2",
        "bypass + blend + output"};

const Network& NnueBenchSelectedNetwork(const int selected_bucket) {
#if defined(SFNNwoPSQT)
  return *network[selected_bucket];
#else
  (void) selected_bucket;
  return *network;
#endif
}

struct alignas(kCacheLineSize) NetworkStageBenchCase {
  NetworkBenchCase input;
  Network::Buffer intermediate;
  float diff_sum_sq = 0.0f;
  float diff_inv_rms = 0.0f;
  std::array<std::int32_t, 32> abs_gated{};
  std::array<std::uint8_t, 32> diff_before_lca;
  std::array<std::int32_t, 32> main_before_gate;
  Network::BenchmarkPhaseScales phase_scales{};
  std::int32_t fc2_before_blend = 0;
  std::int32_t final_output = 0;
};

std::uint32_t NnueBenchFloatBits(const float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void MixNnueBenchFloatBits(std::uint64_t& checksum, const float value) {
  MixNnueBenchChecksum(checksum, NnueBenchFloatBits(value));
}

template<typename T>
void MixNnueBenchRange(std::uint64_t& checksum, const T* values,
                       const std::size_t count) {
  for (std::size_t index = 0; index < count; ++index)
    MixNnueBenchChecksum(checksum, values[index]);
}

void MixNnueBenchPhaseScales(
    std::uint64_t& checksum, const Network::BenchmarkPhaseScales& scales) {
  MixNnueBenchFloatBits(checksum, scales.main_sqr);
  MixNnueBenchFloatBits(checksum, scales.main_raw);
  MixNnueBenchFloatBits(checksum, scales.diff);
  MixNnueBenchFloatBits(checksum, scales.abs_raw);
  MixNnueBenchFloatBits(checksum, scales.abs_sqr);
  MixNnueBenchFloatBits(checksum, scales.cross);
}

struct NnueNetworkStageMismatch {
  std::uint64_t count = 0;
  std::int64_t max_abs_diff = 0;
  std::size_t first_sample = 0;
  std::size_t first_element = 0;
  std::int64_t first_captured = 0;
  std::int64_t first_recomputed = 0;
};

struct NnueNetworkFloatMismatch {
  std::uint64_t count = 0;
  float max_abs_diff = 0.0f;
  std::size_t first_sample = 0;
  std::size_t first_element = 0;
  float first_captured = 0.0f;
  float first_recomputed = 0.0f;
  std::uint32_t first_captured_bits = 0;
  std::uint32_t first_recomputed_bits = 0;
};

template<typename T>
void CompareNnueNetworkStageRange(
    const T* captured, const T* recomputed, const std::size_t count,
    const std::size_t sample_index, NnueNetworkStageMismatch& mismatch) {
  for (std::size_t element = 0; element < count; ++element) {
    const std::int64_t captured_value = captured[element];
    const std::int64_t recomputed_value = recomputed[element];
    if (captured_value == recomputed_value)
      continue;
    const std::int64_t difference =
        std::abs(captured_value - recomputed_value);
    if (mismatch.count == 0) {
      mismatch.first_sample = sample_index;
      mismatch.first_element = element;
      mismatch.first_captured = captured_value;
      mismatch.first_recomputed = recomputed_value;
    }
    ++mismatch.count;
    mismatch.max_abs_diff = std::max(mismatch.max_abs_diff, difference);
  }
}

void CompareNnueNetworkFloatRange(
    const float* captured, const float* recomputed, const std::size_t count,
    const std::size_t sample_index, NnueNetworkFloatMismatch& mismatch) {
  for (std::size_t element = 0; element < count; ++element) {
    const std::uint32_t captured_bits = NnueBenchFloatBits(captured[element]);
    const std::uint32_t recomputed_bits =
        NnueBenchFloatBits(recomputed[element]);
    if (captured_bits == recomputed_bits)
      continue;
    const float difference = std::abs(captured[element] - recomputed[element]);
    if (mismatch.count == 0) {
      mismatch.first_sample = sample_index;
      mismatch.first_element = element;
      mismatch.first_captured = captured[element];
      mismatch.first_recomputed = recomputed[element];
      mismatch.first_captured_bits = captured_bits;
      mismatch.first_recomputed_bits = recomputed_bits;
    }
    ++mismatch.count;
    mismatch.max_abs_diff = std::max(mismatch.max_abs_diff, difference);
  }
}

std::array<float, 6> NnueBenchPhaseScalesToArray(
    const Network::BenchmarkPhaseScales& scales) {
  return {scales.main_sqr, scales.main_raw, scales.diff,
          scales.abs_raw, scales.abs_sqr, scales.cross};
}

struct NnueBenchLcaDerivedDiagnostics {
  float dot_product = 0.0f;
  float attention_logit = 0.0f;
  float attention_score = 0.0f;
  std::array<float, 32> value_clamped{};
  std::array<float, 32> correction{};
  std::array<float, 32> output_before_narrow{};
  std::array<std::uint8_t, 32> output{};
};

NnueBenchLcaDerivedDiagnostics MakeNnueBenchLcaDerivedDiagnostics(
    const Network& selected_network, const std::uint8_t* diff_input,
    const std::int32_t* query_output, const std::int32_t* key_output,
    const std::int32_t* value_output) {
  NnueBenchLcaDerivedDiagnostics result;
  for (int j = 0; j < 32; ++j)
    result.dot_product +=
        (static_cast<float>(query_output[j]) / 8128.0f)
        * (static_cast<float>(key_output[j]) / 8128.0f);
  result.attention_logit =
      (result.dot_product * 0.17677f) / selected_network.lca_temp;
  result.attention_score =
      1.0f / (1.0f + std::exp(-result.attention_logit));

  for (int j = 0; j < 32; ++j) {
    const float current_diff = static_cast<float>(diff_input[j]) / 127.0f;
    const float value = static_cast<float>(value_output[j]) / 8128.0f;
    result.value_clamped[j] =
        std::max(0.0f, std::min(1.0f, value * 0.4f + 0.5f));
    const float final_diff =
        current_diff * (1.0f - result.attention_score)
        + result.value_clamped[j] * result.attention_score;
    result.correction[j] = final_diff - current_diff;
    result.output_before_narrow[j] = final_diff * 127.0f;
    result.output[j] =
        static_cast<std::uint8_t>(result.output_before_narrow[j]);
  }
  return result;
}

std::vector<NetworkStageBenchCase> MakeNnueNetworkStageBenchCorpus() {
  const auto input_corpus = MakeNnueNetworkBenchCorpus();
  std::vector<NetworkStageBenchCase> corpus;
  corpus.reserve(input_corpus.size());

  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];
  for (const auto& input : input_corpus) {
    NetworkStageBenchCase sample{};
    sample.input = input;
    const Network& selected_network =
        NnueBenchSelectedNetwork(input.selected_bucket);
    const auto output = selected_network.Propagate(
        input.transformed.data(), input.diff_transformed.data(),
        input.abs_transformed.data(), input.material_bucket, network_buffer);
    sample.final_output = output[0];
    sample.intermediate =
        *reinterpret_cast<const Network::Buffer*>(network_buffer);

    sample.phase_scales = selected_network.BenchmarkPhaseScalesFromOutput(
        sample.intermediate.phase_out);
    selected_network.BenchmarkDiffRmsNorm(
        sample.intermediate.diff_fc_out, &sample.diff_sum_sq,
        &sample.diff_inv_rms);
    selected_network.BenchmarkAbsSigmoidGate(
        sample.intermediate.abs_fc_out, sample.abs_gated.data());
    std::copy_n(sample.intermediate.fm_cat_uint8, 32,
                sample.diff_before_lca.begin());
    selected_network.BenchmarkMainFc0(
        input.transformed.data(), sample.main_before_gate.data());
    selected_network.BenchmarkFc2(sample.intermediate.ac_1_out,
                                  &sample.fc2_before_blend);
    corpus.emplace_back(std::move(sample));
  }
  return corpus;
}

void ComputeNnueNetworkStageValidationChecksums(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::array<std::uint64_t, kNetworkStageBenchOperations.size()>&
        captured_checksums,
    std::array<std::uint64_t, kNetworkStageBenchOperations.size()>&
        recomputed_checksums,
    std::array<NnueNetworkStageMismatch,
               kNetworkStageBenchOperations.size()>& mismatches,
    std::array<NnueNetworkFloatMismatch,
               kNetworkStageBenchOperations.size()>& float_mismatches,
    NnueNetworkStageMismatch& l2_padding_mismatch,
    std::uint64_t& normal_output_checksum,
    std::uint64_t& staged_output_checksum) {
  captured_checksums.fill(UINT64_C(14695981039346656037));
  recomputed_checksums.fill(UINT64_C(14695981039346656037));
  mismatches.fill(NnueNetworkStageMismatch{});
  float_mismatches.fill(NnueNetworkFloatMismatch{});
  l2_padding_mismatch = {};
  normal_output_checksum = staged_output_checksum =
      UINT64_C(14695981039346656037);

  alignas(kCacheLineSize) Network::Buffer work{};
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);

    MixNnueBenchRange(captured_checksums[0], sample.intermediate.phase_input,
                      384);
    MixNnueBenchRange(captured_checksums[0], sample.intermediate.phase_out, 6);
    MixNnueBenchPhaseScales(captured_checksums[0], sample.phase_scales);
    MixNnueBenchRange(captured_checksums[1], sample.intermediate.diff_fc_out,
                      64);
    MixNnueBenchRange(captured_checksums[2], sample.intermediate.abs_fc_out,
                      64);
    MixNnueBenchFloatBits(captured_checksums[3], sample.diff_sum_sq);
    MixNnueBenchFloatBits(captured_checksums[3], sample.diff_inv_rms);
    MixNnueBenchRange(captured_checksums[4], sample.diff_before_lca.data(), 32);
    MixNnueBenchRange(captured_checksums[5], sample.abs_gated.data(), 32);
    MixNnueBenchRange(captured_checksums[6], sample.intermediate.abs_ac_out,
                      32);
    MixNnueBenchRange(captured_checksums[7], sample.intermediate.abs_sqr_out,
                      32);
    MixNnueBenchRange(captured_checksums[8], sample.main_before_gate.data(),
                      32);
    MixNnueBenchRange(captured_checksums[9], sample.intermediate.fc_0_out, 32);
    MixNnueBenchRange(captured_checksums[10],
                      sample.intermediate.ac_sqr_0_out_temp,
                      32);
    MixNnueBenchRange(captured_checksums[11], sample.intermediate.ac_0_out, 32);
    MixNnueBenchRange(captured_checksums[12], sample.intermediate.diff_ac_out,
                      32);
    MixNnueBenchRange(captured_checksums[13], sample.intermediate.cross_cat,
                      32);
    MixNnueBenchRange(captured_checksums[13], sample.intermediate.cross_fc_out,
                      32);
    MixNnueBenchRange(captured_checksums[13], sample.intermediate.cross_feat,
                      32);
    MixNnueBenchRange(captured_checksums[14], sample.intermediate.l2_input,
                      190);
    MixNnueBenchRange(captured_checksums[15], sample.intermediate.fc_1_out,
                      kHidden2Dims);
    MixNnueBenchRange(captured_checksums[16], sample.intermediate.ac_1_out,
                      kHidden2Dims);
    MixNnueBenchChecksum(captured_checksums[17], sample.fc2_before_blend);
    MixNnueBenchChecksum(captured_checksums[18], sample.final_output);

    const auto scales = selected_network.BenchmarkPhase(
        sample.input.transformed.data(), sample.input.diff_transformed.data(),
        sample.input.abs_transformed.data(), sample.input.material_bucket,
        work.phase_input, work.phase_out);
    MixNnueBenchRange(recomputed_checksums[0], work.phase_input, 384);
    MixNnueBenchRange(recomputed_checksums[0], work.phase_out, 6);
    MixNnueBenchPhaseScales(recomputed_checksums[0], scales);

    selected_network.BenchmarkFcDiff(
        sample.input.diff_transformed.data(), work.diff_fc_out);
    MixNnueBenchRange(recomputed_checksums[1], work.diff_fc_out, 64);
    CompareNnueNetworkStageRange(
        sample.intermediate.diff_fc_out, work.diff_fc_out, 64, sample_index,
        mismatches[1]);
    selected_network.BenchmarkFcAbs(
        sample.input.abs_transformed.data(), work.abs_fc_out);
    MixNnueBenchRange(recomputed_checksums[2], work.abs_fc_out, 64);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_fc_out, work.abs_fc_out, 64, sample_index,
        mismatches[2]);

    float diff_sum_sq = 0.0f;
    float diff_inv_rms = 0.0f;
    selected_network.BenchmarkDiffRmsNorm(
        sample.intermediate.diff_fc_out, &diff_sum_sq, &diff_inv_rms);
    MixNnueBenchFloatBits(recomputed_checksums[3], diff_sum_sq);
    MixNnueBenchFloatBits(recomputed_checksums[3], diff_inv_rms);
    const float captured_diff_rms[2] = {
        sample.diff_sum_sq, sample.diff_inv_rms};
    const float recomputed_diff_rms[2] = {diff_sum_sq, diff_inv_rms};
    CompareNnueNetworkFloatRange(
        captured_diff_rms, recomputed_diff_rms, 2, sample_index,
        float_mismatches[3]);

    selected_network.BenchmarkDiffQuantize(
        sample.intermediate.diff_fc_out, sample.diff_inv_rms,
        work.diff_ac_out);
    MixNnueBenchRange(recomputed_checksums[4], work.diff_ac_out, 32);
    CompareNnueNetworkStageRange(
        sample.diff_before_lca.data(), work.diff_ac_out, 32, sample_index,
        mismatches[4]);

    std::int32_t abs_gated[32];
    selected_network.BenchmarkAbsSigmoidGate(
        sample.intermediate.abs_fc_out, abs_gated);
    MixNnueBenchRange(recomputed_checksums[5], abs_gated, 32);
    CompareNnueNetworkStageRange(
        sample.abs_gated.data(), abs_gated, 32, sample_index,
        mismatches[5]);

    selected_network.BenchmarkAbsGateQuantize(
        sample.abs_gated.data(), work.abs_ac_out);
    MixNnueBenchRange(recomputed_checksums[6], work.abs_ac_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_ac_out, work.abs_ac_out, 32, sample_index,
        mismatches[6]);

    selected_network.BenchmarkAbsSquared(
        sample.intermediate.abs_ac_out, work.abs_sqr_out);
    MixNnueBenchRange(recomputed_checksums[7], work.abs_sqr_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_sqr_out, work.abs_sqr_out, 32, sample_index,
        mismatches[7]);

    selected_network.BenchmarkMainFc0(sample.input.transformed.data(),
                                      work.fc_0_out);
    MixNnueBenchRange(recomputed_checksums[8], work.fc_0_out, 32);
    selected_network.BenchmarkMainGate(
        sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
        work.fc_0_out);
    MixNnueBenchRange(recomputed_checksums[9], work.fc_0_out, 32);
    selected_network.BenchmarkMainSqrClippedRelu(
        sample.intermediate.fc_0_out, work.ac_sqr_0_out_temp);
    MixNnueBenchRange(recomputed_checksums[10],
                      work.ac_sqr_0_out_temp, 32);
    selected_network.BenchmarkMainClippedRelu(
        sample.intermediate.fc_0_out, work.ac_0_out);
    MixNnueBenchRange(recomputed_checksums[11], work.ac_0_out, 32);

    selected_network.BenchmarkLca(
        sample.intermediate.ac_0_out, sample.diff_before_lca.data(),
        sample.intermediate.abs_ac_out, work.diff_ac_out,
        work.fm_cat_uint8, work.lca_q_out, work.lca_k_out, work.lca_v_out);
    MixNnueBenchRange(recomputed_checksums[12], work.diff_ac_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.diff_ac_out, work.diff_ac_out, 32, sample_index,
        mismatches[12]);

    selected_network.BenchmarkCross(
        sample.intermediate.ac_sqr_0_out_temp,
        sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
        sample.intermediate.abs_ac_out, work.cross_cat, work.cross_fc_out,
        work.cross_feat);
    MixNnueBenchRange(recomputed_checksums[13], work.cross_cat, 32);
    MixNnueBenchRange(recomputed_checksums[13], work.cross_fc_out, 32);
    MixNnueBenchRange(recomputed_checksums[13], work.cross_feat, 32);

    selected_network.BenchmarkL2Assembly(
        sample.intermediate.ac_sqr_0_out_temp,
        sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
        sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
        sample.intermediate.cross_feat, sample.phase_scales, work.l2_input);
    MixNnueBenchRange(recomputed_checksums[14], work.l2_input, 190);
    CompareNnueNetworkStageRange(
        sample.intermediate.l2_input, work.l2_input, 190, sample_index,
        mismatches[14]);
    CompareNnueNetworkStageRange(
        sample.intermediate.l2_input + 190, work.l2_input + 190, 2,
        sample_index, l2_padding_mismatch);

    selected_network.BenchmarkFc1(
        sample.intermediate.l2_input, work.fc_1_out);
    MixNnueBenchRange(recomputed_checksums[15], work.fc_1_out,
                      kHidden2Dims);
    CompareNnueNetworkStageRange(
        sample.intermediate.fc_1_out, work.fc_1_out, kHidden2Dims,
        sample_index, mismatches[15]);

    selected_network.BenchmarkAc1(
        sample.intermediate.fc_1_out, work.ac_1_out);
    MixNnueBenchRange(recomputed_checksums[16], work.ac_1_out,
                      kHidden2Dims);
    CompareNnueNetworkStageRange(
        sample.intermediate.ac_1_out, work.ac_1_out, kHidden2Dims,
        sample_index, mismatches[16]);

    selected_network.BenchmarkFc2(sample.intermediate.ac_1_out,
                                  work.fc_2_out);
    MixNnueBenchChecksum(recomputed_checksums[17], work.fc_2_out[0]);
    work.fc_2_out[0] = selected_network.BenchmarkBlend(
        sample.intermediate.fc_0_out[31], sample.fc2_before_blend);
    MixNnueBenchChecksum(recomputed_checksums[18], work.fc_2_out[0]);

    MixNnueBenchChecksum(normal_output_checksum, sample.final_output);
    MixNnueBenchChecksum(staged_output_checksum, work.fc_2_out[0]);
  }
}

struct NnueBenchIntegerDiagnostic {
  std::uint64_t captured_checksum = UINT64_C(14695981039346656037);
  std::uint64_t recomputed_checksum = UINT64_C(14695981039346656037);
  NnueNetworkStageMismatch mismatch{};
};

struct NnueBenchFloatDiagnostic {
  std::uint64_t captured_checksum = UINT64_C(14695981039346656037);
  std::uint64_t recomputed_checksum = UINT64_C(14695981039346656037);
  NnueNetworkFloatMismatch mismatch{};
};

template<typename T>
void AddNnueBenchIntegerDiagnostic(
    NnueBenchIntegerDiagnostic& diagnostic, const T* captured,
    const T* recomputed, const std::size_t count,
    const std::size_t sample_index) {
  MixNnueBenchRange(diagnostic.captured_checksum, captured, count);
  MixNnueBenchRange(diagnostic.recomputed_checksum, recomputed, count);
  CompareNnueNetworkStageRange(captured, recomputed, count, sample_index,
                               diagnostic.mismatch);
}

void AddNnueBenchFloatDiagnostic(
    NnueBenchFloatDiagnostic& diagnostic, const float* captured,
    const float* recomputed, const std::size_t count,
    const std::size_t sample_index) {
  for (std::size_t element = 0; element < count; ++element) {
    MixNnueBenchFloatBits(diagnostic.captured_checksum, captured[element]);
    MixNnueBenchFloatBits(diagnostic.recomputed_checksum,
                          recomputed[element]);
  }
  CompareNnueNetworkFloatRange(captured, recomputed, count, sample_index,
                               diagnostic.mismatch);
}

void PrintNnueBenchIntegerDiagnostic(
    const char* name, const NnueBenchIntegerDiagnostic& diagnostic) {
  std::cout << name << std::endl
            << "  captured checksum  : 0x" << std::hex
            << diagnostic.captured_checksum << std::endl
            << "  recomputed checksum: 0x"
            << diagnostic.recomputed_checksum << std::dec << std::endl
            << "  mismatch count     : " << diagnostic.mismatch.count
            << std::endl;
  if (diagnostic.mismatch.count != 0) {
    std::cout << "  first mismatch sample : "
              << diagnostic.mismatch.first_sample << std::endl
              << "  first mismatch element: "
              << diagnostic.mismatch.first_element << std::endl
              << "  captured value        : "
              << diagnostic.mismatch.first_captured << std::endl
              << "  recomputed value      : "
              << diagnostic.mismatch.first_recomputed << std::endl
              << "  max abs diff          : "
              << diagnostic.mismatch.max_abs_diff << std::endl;
  }
}

void PrintNnueBenchFloatDiagnostic(
    const char* name, const NnueBenchFloatDiagnostic& diagnostic) {
  std::cout << name << std::endl
            << "  captured checksum  : 0x" << std::hex
            << diagnostic.captured_checksum << std::endl
            << "  recomputed checksum: 0x"
            << diagnostic.recomputed_checksum << std::dec << std::endl
            << "  bit mismatch count : " << diagnostic.mismatch.count
            << std::endl;
  if (diagnostic.mismatch.count != 0) {
    std::cout << "  first mismatch sample : "
              << diagnostic.mismatch.first_sample << std::endl
              << "  first mismatch channel: "
              << diagnostic.mismatch.first_element << std::endl
              << "  captured value        : "
              << std::setprecision(9)
              << diagnostic.mismatch.first_captured << std::endl
              << "  recomputed value      : "
              << diagnostic.mismatch.first_recomputed << std::endl
              << "  captured bits         : 0x" << std::hex
              << diagnostic.mismatch.first_captured_bits << std::endl
              << "  recomputed bits       : 0x"
              << diagnostic.mismatch.first_recomputed_bits << std::dec
              << std::endl
              << "  max abs diff          : "
              << diagnostic.mismatch.max_abs_diff << std::endl;
  }
}

void DiagnoseNnueNetworkPhaseAndLca(
    const std::vector<NetworkStageBenchCase>& corpus) {
  NnueBenchIntegerDiagnostic phase_input;
  NnueBenchIntegerDiagnostic phase_output;
  NnueBenchFloatDiagnostic phase_scales;

  NnueBenchIntegerDiagnostic lca_fm_cat;
  NnueBenchIntegerDiagnostic lca_query;
  NnueBenchIntegerDiagnostic lca_key;
  NnueBenchIntegerDiagnostic lca_value;
  NnueBenchFloatDiagnostic lca_dot_product;
  NnueBenchFloatDiagnostic lca_attention_logit;
  NnueBenchFloatDiagnostic lca_attention_score;
  NnueBenchIntegerDiagnostic lca_diff_before_correction;
  NnueBenchFloatDiagnostic lca_value_clamped;
  NnueBenchFloatDiagnostic lca_correction;
  NnueBenchFloatDiagnostic lca_before_narrow;
  NnueBenchIntegerDiagnostic lca_output_from_captured_operands;
  NnueBenchIntegerDiagnostic lca_helper_vs_derived;
  NnueBenchIntegerDiagnostic lca_output;

  alignas(kCacheLineSize) Network::Buffer work{};
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);

    const auto recomputed_scales = selected_network.BenchmarkPhase(
        sample.input.transformed.data(), sample.input.diff_transformed.data(),
        sample.input.abs_transformed.data(), sample.input.material_bucket,
        work.phase_input, work.phase_out);
    AddNnueBenchIntegerDiagnostic(
        phase_input, sample.intermediate.phase_input, work.phase_input, 384,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        phase_output, sample.intermediate.phase_out, work.phase_out, 6,
        sample_index);
    const auto captured_scale_values =
        NnueBenchPhaseScalesToArray(sample.phase_scales);
    const auto recomputed_scale_values =
        NnueBenchPhaseScalesToArray(recomputed_scales);
    AddNnueBenchFloatDiagnostic(
        phase_scales, captured_scale_values.data(),
        recomputed_scale_values.data(), captured_scale_values.size(),
        sample_index);

    selected_network.BenchmarkLca(
        sample.intermediate.ac_0_out, sample.diff_before_lca.data(),
        sample.intermediate.abs_ac_out, work.diff_ac_out,
        work.fm_cat_uint8, work.lca_q_out, work.lca_k_out, work.lca_v_out);

    AddNnueBenchIntegerDiagnostic(
        lca_fm_cat, sample.intermediate.fm_cat_uint8, work.fm_cat_uint8, 64,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_query, sample.intermediate.lca_q_out, work.lca_q_out, 32,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_key, sample.intermediate.lca_k_out, work.lca_k_out, 32,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_value, sample.intermediate.lca_v_out, work.lca_v_out, 32,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_diff_before_correction, sample.intermediate.fm_cat_uint8,
        sample.diff_before_lca.data(), 32, sample_index);

    const auto captured_lca = MakeNnueBenchLcaDerivedDiagnostics(
        selected_network, sample.intermediate.fm_cat_uint8,
        sample.intermediate.lca_q_out, sample.intermediate.lca_k_out,
        sample.intermediate.lca_v_out);
    const auto recomputed_lca = MakeNnueBenchLcaDerivedDiagnostics(
        selected_network, sample.diff_before_lca.data(), work.lca_q_out,
        work.lca_k_out, work.lca_v_out);
    AddNnueBenchFloatDiagnostic(
        lca_dot_product, &captured_lca.dot_product,
        &recomputed_lca.dot_product, 1, sample_index);
    AddNnueBenchFloatDiagnostic(
        lca_attention_logit, &captured_lca.attention_logit,
        &recomputed_lca.attention_logit, 1, sample_index);
    AddNnueBenchFloatDiagnostic(
        lca_attention_score, &captured_lca.attention_score,
        &recomputed_lca.attention_score, 1, sample_index);
    AddNnueBenchFloatDiagnostic(
        lca_value_clamped, captured_lca.value_clamped.data(),
        recomputed_lca.value_clamped.data(), 32, sample_index);
    AddNnueBenchFloatDiagnostic(
        lca_correction, captured_lca.correction.data(),
        recomputed_lca.correction.data(), 32, sample_index);
    AddNnueBenchFloatDiagnostic(
        lca_before_narrow, captured_lca.output_before_narrow.data(),
        recomputed_lca.output_before_narrow.data(), 32, sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_output_from_captured_operands, sample.intermediate.diff_ac_out,
        captured_lca.output.data(), 32, sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_helper_vs_derived, work.diff_ac_out,
        recomputed_lca.output.data(), 32, sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_output, sample.intermediate.diff_ac_out, work.diff_ac_out, 32,
        sample_index);
  }

  std::cout << "[Phase intermediate diagnostics]" << std::endl;
  PrintNnueBenchIntegerDiagnostic("phase.input[384]", phase_input);
  PrintNnueBenchIntegerDiagnostic("phase.proj_output[6]", phase_output);
  PrintNnueBenchFloatDiagnostic("phase.channel_scales[6]", phase_scales);

  std::cout << "[LCA intermediate diagnostics]" << std::endl;
  PrintNnueBenchIntegerDiagnostic("lca.fm_cat[64]", lca_fm_cat);
  PrintNnueBenchIntegerDiagnostic("lca.query[32]", lca_query);
  PrintNnueBenchIntegerDiagnostic("lca.key[32]", lca_key);
  PrintNnueBenchIntegerDiagnostic("lca.value[32]", lca_value);
  PrintNnueBenchFloatDiagnostic("lca.dot_product", lca_dot_product);
  PrintNnueBenchFloatDiagnostic("lca.attention_logit", lca_attention_logit);
  PrintNnueBenchFloatDiagnostic("lca.attention_score", lca_attention_score);
  PrintNnueBenchIntegerDiagnostic(
      "lca.diff_before_correction[32]", lca_diff_before_correction);
  PrintNnueBenchFloatDiagnostic(
      "lca.value_clamped[32]", lca_value_clamped);
  PrintNnueBenchFloatDiagnostic("lca.correction[32]", lca_correction);
  PrintNnueBenchFloatDiagnostic(
      "lca.output_before_narrow[32]", lca_before_narrow);
  PrintNnueBenchIntegerDiagnostic(
      "lca.normal_output_vs_captured_operands[32]",
      lca_output_from_captured_operands);
  PrintNnueBenchIntegerDiagnostic(
      "lca.helper_output_vs_derived[32]", lca_helper_vs_derived);
  PrintNnueBenchIntegerDiagnostic("lca.diff_ac_out[32]", lca_output);
}

template<bool UseTiledFc1>
std::int32_t ComputeNnueNetworkStagedOutput(
    const NetworkStageBenchCase& sample, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(sample.input.selected_bucket);
  const auto scales = selected_network.BenchmarkPhase(
      sample.input.transformed.data(), sample.input.diff_transformed.data(),
      sample.input.abs_transformed.data(), sample.input.material_bucket,
      work.phase_input, work.phase_out);
  selected_network.BenchmarkFmAffine(
      sample.input.diff_transformed.data(),
      sample.input.abs_transformed.data(), work.diff_fc_out,
      work.abs_fc_out);
  selected_network.BenchmarkFmActivation(
      work.diff_fc_out, work.abs_fc_out, work.diff_ac_out, work.abs_ac_out,
      work.abs_sqr_out);
  selected_network.BenchmarkMain(
      sample.input.transformed.data(), work.diff_fc_out, work.fc_0_out,
      work.ac_sqr_0_out_temp, work.ac_0_out);
  selected_network.BenchmarkLca(
      work.ac_0_out, work.diff_ac_out, work.abs_ac_out, work.diff_ac_out,
      work.fm_cat_uint8, work.lca_q_out, work.lca_k_out, work.lca_v_out);
  selected_network.BenchmarkCross(
      work.ac_sqr_0_out_temp, work.ac_0_out, work.diff_ac_out,
      work.abs_ac_out, work.cross_cat, work.cross_fc_out, work.cross_feat);
  selected_network.BenchmarkL2Assembly(
      work.ac_sqr_0_out_temp, work.ac_0_out, work.diff_ac_out,
      work.abs_ac_out, work.abs_sqr_out, work.cross_feat, scales,
      work.l2_input);
#if defined(USE_AVX2) && !defined(USE_AVX512)
  if constexpr (UseTiledFc1)
    selected_network.BenchmarkFc1OutputTiled(work.l2_input, work.fc_1_out);
  else
#else
  static_assert(!UseTiledFc1);
#endif
    selected_network.BenchmarkFc1(work.l2_input, work.fc_1_out);
  selected_network.BenchmarkAc1(work.fc_1_out, work.ac_1_out);
  selected_network.BenchmarkFc2(work.ac_1_out, work.fc_2_out);
  return selected_network.BenchmarkBlend(work.fc_0_out[31],
                                         work.fc_2_out[0]);
}

void ValidateNnueNetworkStagedEndToEnd(
    const std::vector<NetworkStageBenchCase>& corpus) {
  std::uint64_t normal_checksum = UINT64_C(14695981039346656037);
  std::uint64_t staged_checksum = UINT64_C(14695981039346656037);
  NnueNetworkStageMismatch mismatch;
  alignas(kCacheLineSize) Network::Buffer work{};

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const std::int32_t staged_output =
        ComputeNnueNetworkStagedOutput<false>(sample, work);

    MixNnueBenchChecksum(normal_checksum, sample.final_output);
    MixNnueBenchChecksum(staged_checksum, staged_output);
    CompareNnueNetworkStageRange(
        &sample.final_output, &staged_output, 1, sample_index, mismatch);
  }

  std::cout << "[Staged end-to-end validation]" << std::endl
            << "  normal checksum : 0x" << std::hex << normal_checksum
            << std::endl
            << "  staged checksum : 0x" << staged_checksum << std::dec
            << std::endl
            << "  checksum match  : "
            << (normal_checksum == staged_checksum ? "yes" : "NO")
            << std::endl
            << "  mismatch count : " << mismatch.count << std::endl;
  if (mismatch.count != 0) {
    std::cout << "  first mismatch sample : " << mismatch.first_sample
              << std::endl
              << "  normal output         : " << mismatch.first_captured
              << std::endl
              << "  staged output         : " << mismatch.first_recomputed
              << std::endl
              << "  max abs diff          : " << mismatch.max_abs_diff
              << std::endl;
  }
}

template<NetworkStageBenchOperation Operation>
NnueBenchTiming MeasureNnueNetworkStageCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    const std::size_t index = static_cast<std::size_t>(timing.calls);
    std::uint64_t representative = 0;

    if constexpr (Operation == NetworkStageBenchOperation::Phase) {
      const auto scales = selected_network.BenchmarkPhase(
          sample.input.transformed.data(),
          sample.input.diff_transformed.data(),
          sample.input.abs_transformed.data(), sample.input.material_bucket,
          work.phase_input, work.phase_out);
      const float selected_scale = index % 2 == 0 ? scales.main_sqr
                                                  : scales.cross;
      representative = static_cast<std::uint32_t>(work.phase_out[index % 6])
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(selected_scale))
             << 32);
    } else if constexpr (Operation == NetworkStageBenchOperation::FcDiff) {
      selected_network.BenchmarkFcDiff(
          sample.input.diff_transformed.data(), work.diff_fc_out);
      representative = static_cast<std::uint32_t>(
                           work.diff_fc_out[index % 64])
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.diff_fc_out[(index + 33) % 64])) << 32);
    } else if constexpr (Operation == NetworkStageBenchOperation::FcAbs) {
      selected_network.BenchmarkFcAbs(
          sample.input.abs_transformed.data(), work.abs_fc_out);
      representative = static_cast<std::uint32_t>(
                           work.abs_fc_out[index % 64])
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.abs_fc_out[(index + 33) % 64])) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::DiffRmsNorm) {
      float sum_sq = 0.0f;
      float inv_rms = 0.0f;
      selected_network.BenchmarkDiffRmsNorm(
          sample.intermediate.diff_fc_out, &sum_sq, &inv_rms);
      representative = NnueBenchFloatBits(sum_sq)
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(inv_rms)) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::DiffQuantize) {
      selected_network.BenchmarkDiffQuantize(
          sample.intermediate.diff_fc_out, sample.diff_inv_rms,
          work.diff_ac_out);
      representative = work.diff_ac_out[index % 32];
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::AbsSigmoidGate) {
      selected_network.BenchmarkAbsSigmoidGate(
          sample.intermediate.abs_fc_out, work.lca_q_out);
      representative = static_cast<std::uint32_t>(work.lca_q_out[index % 32]);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::AbsGateQuantize) {
      selected_network.BenchmarkAbsGateQuantize(
          sample.abs_gated.data(), work.abs_ac_out);
      representative = work.abs_ac_out[index % 32];
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::AbsSquared) {
      selected_network.BenchmarkAbsSquared(
          sample.intermediate.abs_ac_out, work.abs_sqr_out);
      representative = work.abs_sqr_out[index % 32];
    } else if constexpr (Operation == NetworkStageBenchOperation::MainFc0) {
      selected_network.BenchmarkMainFc0(sample.input.transformed.data(),
                                        work.fc_0_out);
      representative = static_cast<std::uint32_t>(work.fc_0_out[index % 32])
          ^ (static_cast<std::uint64_t>(
                 static_cast<std::uint32_t>(work.fc_0_out[(index + 17) % 32]))
             << 32);
    } else if constexpr (Operation == NetworkStageBenchOperation::MainGate) {
      selected_network.BenchmarkMainGate(
          sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
          work.fc_0_out);
      representative = static_cast<std::uint32_t>(work.fc_0_out[index % 32])
          ^ (static_cast<std::uint64_t>(
                 static_cast<std::uint32_t>(work.fc_0_out[(index + 17) % 32]))
             << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::MainSqrClippedRelu) {
      selected_network.BenchmarkMainSqrClippedRelu(
          sample.intermediate.fc_0_out, work.ac_sqr_0_out_temp);
      representative = work.ac_sqr_0_out_temp[index % 32];
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::MainClippedRelu) {
      selected_network.BenchmarkMainClippedRelu(
          sample.intermediate.fc_0_out, work.ac_0_out);
      representative = work.ac_0_out[index % 32];
    } else if constexpr (Operation == NetworkStageBenchOperation::Lca) {
      selected_network.BenchmarkLca(
          sample.intermediate.ac_0_out, sample.diff_before_lca.data(),
          sample.intermediate.abs_ac_out, work.diff_ac_out,
          work.fm_cat_uint8, work.lca_q_out, work.lca_k_out,
          work.lca_v_out);
      representative = work.diff_ac_out[index % 32]
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.lca_q_out[index % 32])) << 8)
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.lca_k_out[index % 32])) << 24)
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.lca_v_out[index % 32])) << 40);
    } else if constexpr (Operation == NetworkStageBenchOperation::Cross) {
      selected_network.BenchmarkCross(
          sample.intermediate.ac_sqr_0_out_temp,
          sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
          sample.intermediate.abs_ac_out, work.cross_cat,
          work.cross_fc_out, work.cross_feat);
      representative = work.cross_cat[index % 32]
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.cross_fc_out[index % 32])) << 8)
          ^ (static_cast<std::uint64_t>(work.cross_feat[index % 32]) << 48);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::L2Assembly) {
      selected_network.BenchmarkL2Assembly(
          sample.intermediate.ac_sqr_0_out_temp,
          sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
          sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
          sample.intermediate.cross_feat, sample.phase_scales,
          work.l2_input);
      representative = work.l2_input[index % L2_INPUT_SIZE];
    } else if constexpr (Operation == NetworkStageBenchOperation::Fc1) {
      selected_network.BenchmarkFc1(
          sample.intermediate.l2_input, work.fc_1_out);
      representative = static_cast<std::uint32_t>(
          work.fc_1_out[index % kHidden2Dims]);
    } else if constexpr (Operation == NetworkStageBenchOperation::Ac1) {
      selected_network.BenchmarkAc1(
          sample.intermediate.fc_1_out, work.ac_1_out);
      representative = work.ac_1_out[index % kHidden2Dims];
    } else if constexpr (Operation == NetworkStageBenchOperation::Fc2) {
      selected_network.BenchmarkFc2(sample.intermediate.ac_1_out,
                                    work.fc_2_out);
      representative = work.fc_2_out[0];
    } else {
      const std::int32_t output = selected_network.BenchmarkBlend(
          sample.intermediate.fc_0_out[31], sample.fc2_before_blend);
      representative = output;
    }
    MixNnueBenchChecksum(checksum, static_cast<std::int64_t>(representative));
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureNnueNetworkStageByIndex(
    const std::vector<NetworkStageBenchCase>& corpus,
    const std::size_t stage_index, std::uint64_t& checksum) {
  switch (kNetworkStageBenchOperations[stage_index]) {
    case NetworkStageBenchOperation::Phase:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Phase>(
          corpus, checksum);
    case NetworkStageBenchOperation::FcDiff:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::FcDiff>(corpus, checksum);
    case NetworkStageBenchOperation::FcAbs:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::FcAbs>(corpus, checksum);
    case NetworkStageBenchOperation::DiffRmsNorm:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::DiffRmsNorm>(corpus, checksum);
    case NetworkStageBenchOperation::DiffQuantize:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::DiffQuantize>(corpus, checksum);
    case NetworkStageBenchOperation::AbsSigmoidGate:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::AbsSigmoidGate>(corpus, checksum);
    case NetworkStageBenchOperation::AbsGateQuantize:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::AbsGateQuantize>(corpus, checksum);
    case NetworkStageBenchOperation::AbsSquared:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::AbsSquared>(corpus, checksum);
    case NetworkStageBenchOperation::MainFc0:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainFc0>(corpus, checksum);
    case NetworkStageBenchOperation::MainGate:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainGate>(corpus, checksum);
    case NetworkStageBenchOperation::MainSqrClippedRelu:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainSqrClippedRelu>(corpus, checksum);
    case NetworkStageBenchOperation::MainClippedRelu:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainClippedRelu>(corpus, checksum);
    case NetworkStageBenchOperation::Lca:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Lca>(
          corpus, checksum);
    case NetworkStageBenchOperation::Cross:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Cross>(
          corpus, checksum);
    case NetworkStageBenchOperation::L2Assembly:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::L2Assembly>(corpus, checksum);
    case NetworkStageBenchOperation::Fc1:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Fc1>(
          corpus, checksum);
    case NetworkStageBenchOperation::Ac1:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Ac1>(
          corpus, checksum);
    case NetworkStageBenchOperation::Fc2:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Fc2>(
          corpus, checksum);
    case NetworkStageBenchOperation::Blend:
      return MeasureNnueNetworkStageCorpus<NetworkStageBenchOperation::Blend>(
          corpus, checksum);
  }
  return {};
}

NnueBenchTiming MeasureNnueNetworkStageAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    const std::size_t stage_index, std::uint64_t& checksum) {
  MeasureNnueNetworkStageByIndex(corpus, stage_index, checksum);
  return MeasureNnueNetworkStageByIndex(corpus, stage_index, checksum);
}

NnueBenchTiming MeasureNnueFullNetworkStageCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const auto output = NnueBenchSelectedNetwork(sample.input.selected_bucket)
        .Propagate(sample.input.transformed.data(),
                   sample.input.diff_transformed.data(),
                   sample.input.abs_transformed.data(),
                   sample.input.material_bucket, network_buffer);
    MixNnueBenchChecksum(checksum, output[0]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureNnueFullNetworkStageAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  MeasureNnueFullNetworkStageCorpus(corpus, checksum);
  return MeasureNnueFullNetworkStageCorpus(corpus, checksum);
}

void PrintNnueNetworkStageSamples(const char* const name,
                                  const NnueBenchSamples& samples,
                                  const double full_median) {
  const NnueBenchSummary summary = SummarizeNnueBenchSamples(samples);
  const double percentage = full_median == 0.0
      ? 0.0
      : summary.median * 100.0 / full_median;
  std::cout << name << std::endl
            << "  median ns/call : " << std::fixed << std::setprecision(1)
            << summary.median << std::endl
            << "  mean ns/call   : " << summary.mean << std::endl
            << "  min ns/call    : " << summary.minimum << std::endl
            << "  max ns/call    : " << summary.maximum << std::endl
            << "  share of full  : " << std::setprecision(2) << percentage
            << "%" << std::endl;
}

void TestNetworkStagesBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Network stages]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  stage order  : rotated by one stage per repeat" << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: NNUE network stage benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  std::array<std::uint64_t, kNetworkStageBenchOperations.size()>
      captured_checksums;
  std::array<std::uint64_t, kNetworkStageBenchOperations.size()>
      recomputed_checksums;
  std::array<NnueNetworkStageMismatch,
             kNetworkStageBenchOperations.size()> mismatches{};
  std::array<NnueNetworkFloatMismatch,
             kNetworkStageBenchOperations.size()> float_mismatches{};
  NnueNetworkStageMismatch l2_padding_mismatch{};
  std::uint64_t normal_output_checksum;
  std::uint64_t staged_output_checksum;
  ComputeNnueNetworkStageValidationChecksums(
      corpus, captured_checksums, recomputed_checksums, mismatches,
      float_mismatches, l2_padding_mismatch,
      normal_output_checksum, staged_output_checksum);

  std::array<NnueBenchSamples, kNetworkStageBenchOperations.size()>
      stage_samples;
  NnueBenchSamples full_samples;
  std::array<std::uint64_t, kNetworkStageBenchOperations.size()>
      stage_checksums;
  stage_checksums.fill(UINT64_C(14695981039346656037));
  std::uint64_t full_timing_checksum = UINT64_C(14695981039346656037);

  constexpr std::size_t operation_count =
      kNetworkStageBenchOperations.size() + 1;
  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    const std::size_t rotation = repeat % operation_count;
    for (std::size_t offset = 0; offset < operation_count; ++offset) {
      const std::size_t operation = (rotation + offset) % operation_count;
      if (operation == kNetworkStageBenchOperations.size()) {
        full_samples.Add(MeasureNnueFullNetworkStageAfterWarmup(
            corpus, full_timing_checksum));
      } else {
        stage_samples[operation].Add(MeasureNnueNetworkStageAfterWarmup(
            corpus, operation, stage_checksums[operation]));
      }
    }
  }

  const NnueBenchSummary full_summary =
      SummarizeNnueBenchSamples(full_samples);
  double summed_stage_medians = 0.0;
  bool all_intermediate_checksums_match = true;
  for (std::size_t stage = 0;
       stage < kNetworkStageBenchOperations.size(); ++stage) {
    PrintNnueNetworkStageSamples(kNetworkStageBenchNames[stage],
                                 stage_samples[stage], full_summary.median);
    summed_stage_medians +=
        SummarizeNnueBenchSamples(stage_samples[stage]).median;
    std::cout << "  timing checksum : 0x" << std::hex
              << stage_checksums[stage] << std::dec << std::endl
              << "  captured checksum: 0x" << std::hex
              << captured_checksums[stage] << std::endl
              << "  recomputed checksum: 0x"
              << recomputed_checksums[stage] << std::dec << std::endl
              << "  intermediate match: "
              << (captured_checksums[stage] == recomputed_checksums[stage]
                      ? "yes"
                      : "NO")
              << std::endl;
    if (mismatches[stage].count != 0) {
      const auto& mismatch = mismatches[stage];
      std::cout << "  first mismatch sample : " << mismatch.first_sample
                << std::endl
                << "  first mismatch element: " << mismatch.first_element
                << std::endl
                << "  captured value        : " << mismatch.first_captured
                << std::endl
                << "  recomputed value      : " << mismatch.first_recomputed
                << std::endl
                << "  max abs diff          : " << mismatch.max_abs_diff
                << std::endl
                << "  mismatch count        : " << mismatch.count
                << std::endl;
    }
    if (float_mismatches[stage].count != 0) {
      const auto& mismatch = float_mismatches[stage];
      std::cout << "  float first mismatch sample : "
                << mismatch.first_sample << std::endl
                << "  float first mismatch element: "
                << mismatch.first_element << std::endl
                << "  float captured value        : "
                << std::setprecision(9) << mismatch.first_captured
                << std::endl
                << "  float recomputed value      : "
                << mismatch.first_recomputed << std::endl
                << "  float captured bits         : 0x" << std::hex
                << mismatch.first_captured_bits << std::endl
                << "  float recomputed bits       : 0x"
                << mismatch.first_recomputed_bits << std::dec << std::endl
                << "  float max abs diff          : "
                << mismatch.max_abs_diff << std::endl
                << "  float bit mismatch count    : "
                << mismatch.count << std::endl;
    }
    if (stage == 3 && float_mismatches[stage].count == 0)
      std::cout << "  float bit mismatch count    : 0" << std::endl;
    if (stage == 14) {
      std::cout << "  L2 real range         : [0, 190)" << std::endl
                << "  L2 padding range      : [190, 192)" << std::endl
                << "  L2 padding match      : "
                << (l2_padding_mismatch.count == 0 ? "yes" : "NO")
                << std::endl;
      if (l2_padding_mismatch.count != 0) {
        std::cout << "  padding first sample  : "
                  << l2_padding_mismatch.first_sample << std::endl
                  << "  padding first element : "
                  << l2_padding_mismatch.first_element << std::endl
                  << "  padding captured      : "
                  << l2_padding_mismatch.first_captured << std::endl
                  << "  padding recomputed    : "
                  << l2_padding_mismatch.first_recomputed << std::endl
                  << "  padding max abs diff  : "
                  << l2_padding_mismatch.max_abs_diff << std::endl
                  << "  padding mismatch count: "
                  << l2_padding_mismatch.count << std::endl;
      }
    }
    all_intermediate_checksums_match &=
        captured_checksums[stage] == recomputed_checksums[stage];
  }
  all_intermediate_checksums_match &= l2_padding_mismatch.count == 0;

  PrintNnueNetworkStageSamples("full Network::Propagate", full_samples,
                               full_summary.median);
  std::cout << "  timing checksum : 0x" << std::hex
            << full_timing_checksum << std::dec << std::endl
            << "summed isolated stage medians : " << std::fixed
            << std::setprecision(1) << summed_stage_medians << " ns/call"
            << std::endl
            << "full Network median           : " << full_summary.median
            << " ns/call" << std::endl
            << "all intermediate stages match : "
            << (all_intermediate_checksums_match ? "yes" : "NO")
            << std::endl
            << "normal output checksum        : 0x" << std::hex
            << normal_output_checksum << std::endl
            << "staged output checksum        : 0x"
            << staged_output_checksum << std::dec << std::endl
            << "final output checksum match   : "
            << (normal_output_checksum == staged_output_checksum ? "yes"
                                                                  : "NO")
            << std::endl
            << "note: isolated stage medians need not sum to the full median;"
            << std::endl
            << "      cache state, materialized buffers, call boundaries, and"
            << std::endl
            << "      producer/consumer locality differ." << std::endl;

  DiagnoseNnueNetworkPhaseAndLca(corpus);
  ValidateNnueNetworkStagedEndToEnd(corpus);
}

#if defined(USE_AVX2)

enum class FmAbsSquaredBenchImplementation {
  CurrentScalar,
  Avx2Exact,
};

template<FmAbsSquaredBenchImplementation Implementation>
NnueBenchTiming MeasureFmAbsSquaredCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::uint8_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    if constexpr (Implementation ==
                  FmAbsSquaredBenchImplementation::Avx2Exact)
      selected_network.BenchmarkAbsSquaredAvx2(
          sample.intermediate.abs_ac_out, output);
    else
      selected_network.BenchmarkAbsSquaredScalar(
          sample.intermediate.abs_ac_out, output);

    const std::size_t checksum_index =
        static_cast<std::size_t>(timing.calls) & 31;
    MixNnueBenchChecksum(checksum, output[checksum_index]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<FmAbsSquaredBenchImplementation Implementation>
NnueBenchTiming MeasureFmAbsSquaredCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureFmAbsSquaredCorpus<Implementation>(corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureFmAbsSquaredCorpus<Implementation>(corpus, checksum);
}

void TestFmAbsSquaredBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: FM Abs squared scalar / AVX2 exact]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  elements/call: 32" << std::endl
            << "  order        : even=scalar,AVX2 odd=AVX2,scalar"
            << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: FM Abs squared benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  compared values: " << corpus.size() * 32 << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  NnueBenchSamples scalar_samples;
  NnueBenchSamples avx2_samples;
  std::uint64_t scalar_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t avx2_timing_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      scalar_samples.Add(MeasureFmAbsSquaredCorpusAfterWarmup<
          FmAbsSquaredBenchImplementation::CurrentScalar>(
              corpus, scalar_timing_checksum));
      avx2_samples.Add(MeasureFmAbsSquaredCorpusAfterWarmup<
          FmAbsSquaredBenchImplementation::Avx2Exact>(
              corpus, avx2_timing_checksum));
    } else {
      avx2_samples.Add(MeasureFmAbsSquaredCorpusAfterWarmup<
          FmAbsSquaredBenchImplementation::Avx2Exact>(
              corpus, avx2_timing_checksum));
      scalar_samples.Add(MeasureFmAbsSquaredCorpusAfterWarmup<
          FmAbsSquaredBenchImplementation::CurrentScalar>(
              corpus, scalar_timing_checksum));
    }
  }

  PrintNnueBenchSamples("A. current scalar source", scalar_samples);
  PrintNnueBenchSamples("B. AVX2 exact", avx2_samples);

  const NnueBenchSummary scalar_summary =
      SummarizeNnueBenchSamples(scalar_samples);
  const NnueBenchSummary avx2_summary =
      SummarizeNnueBenchSamples(avx2_samples);
  const double difference = avx2_summary.median - scalar_summary.median;
  const double improvement = scalar_summary.median == 0.0
      ? 0.0
      : (scalar_summary.median - avx2_summary.median)
            * 100.0 / scalar_summary.median;

  alignas(kCacheLineSize) std::uint8_t scalar_output[32];
  alignas(kCacheLineSize) std::uint8_t avx2_output[32];
  std::uint64_t scalar_checksum = UINT64_C(14695981039346656037);
  std::uint64_t avx2_checksum = UINT64_C(14695981039346656037);
  NnueNetworkStageMismatch mismatch{};
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    selected_network.BenchmarkAbsSquaredScalar(
        sample.intermediate.abs_ac_out, scalar_output);
    selected_network.BenchmarkAbsSquaredAvx2(
        sample.intermediate.abs_ac_out, avx2_output);
    MixNnueBenchRange(scalar_checksum, scalar_output, 32);
    MixNnueBenchRange(avx2_checksum, avx2_output, 32);
    CompareNnueNetworkStageRange(
        scalar_output, avx2_output, 32, sample_index, mismatch);
  }

  std::cout << "  difference (AVX2 - scalar) : " << std::fixed
            << std::setprecision(1) << difference << " ns/call" << std::endl
            << "  improvement                : " << std::setprecision(2)
            << improvement << "%" << std::endl
            << "  scalar checksum : 0x" << std::hex << scalar_checksum
            << std::endl
            << "  AVX2 checksum   : 0x" << avx2_checksum << std::dec
            << std::endl
            << "  checksum match  : "
            << (scalar_checksum == avx2_checksum ? "yes" : "NO")
            << std::endl
            << "  mismatch count  : " << mismatch.count << std::endl;
  if (mismatch.count != 0) {
    std::cout << "  first mismatch sample : " << mismatch.first_sample
              << std::endl
              << "  first mismatch element: " << mismatch.first_element
              << std::endl
              << "  scalar value          : " << mismatch.first_captured
              << std::endl
              << "  AVX2 value            : " << mismatch.first_recomputed
              << std::endl
              << "  max abs diff          : " << mismatch.max_abs_diff
              << std::endl;
  }
  std::cout << "  scalar timing checksum : 0x" << std::hex
            << scalar_timing_checksum << std::endl
            << "  AVX2 timing checksum   : 0x" << avx2_timing_checksum
            << std::dec << std::endl
            << "  timing checksum match  : "
            << (scalar_timing_checksum == avx2_timing_checksum
                    ? "yes"
                    : "NO")
            << std::endl;
}

#endif  // defined(USE_AVX2)

struct alignas(kCacheLineSize) SqrClippedReluBenchCase {
  std::array<std::int32_t, 32> input;
  int selected_bucket = 0;
};

std::vector<SqrClippedReluBenchCase> MakeSqrClippedReluBenchCorpus() {
  const auto network_corpus = MakeNnueNetworkBenchCorpus();
  std::vector<SqrClippedReluBenchCase> corpus;
  corpus.reserve(network_corpus.size());
  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];

  for (const auto& sample : network_corpus) {
#if defined(SFNNwoPSQT)
    network[sample.selected_bucket]->Propagate(
#else
    network->Propagate(
#endif
        sample.transformed.data(), sample.diff_transformed.data(),
        sample.abs_transformed.data(), sample.material_bucket,
        network_buffer);
    const auto& network_work =
        *reinterpret_cast<const Network::Buffer*>(network_buffer);

    SqrClippedReluBenchCase sqr_sample;
    std::copy_n(network_work.fc_0_out, 32, sqr_sample.input.begin());
    sqr_sample.selected_bucket = sample.selected_bucket;
    corpus.emplace_back(std::move(sqr_sample));
  }
  return corpus;
}

enum class SqrClippedReluBenchImplementation {
  Baseline,
  Avx2,
};

template<SqrClippedReluBenchImplementation Implementation>
NnueBenchTiming MeasureSqrClippedReluCorpus(
    const std::vector<SqrClippedReluBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::uint8_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
#if defined(SFNNwoPSQT)
    const auto& activation = network[sample.selected_bucket]->ac_sqr_0;
#else
    const auto& activation = network->ac_sqr_0;
#endif
    if constexpr (Implementation == SqrClippedReluBenchImplementation::Avx2)
      activation.BenchmarkPropagateAvx2(sample.input.data(), output);
    else
      activation.BenchmarkPropagateBaseline(sample.input.data(), output);

    const std::size_t checksum_index =
        static_cast<std::size_t>(timing.calls) & 31;
    MixNnueBenchChecksum(checksum, output[checksum_index]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<SqrClippedReluBenchImplementation Implementation>
NnueBenchTiming MeasureSqrClippedReluCorpusAfterWarmup(
    const std::vector<SqrClippedReluBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureSqrClippedReluCorpus<Implementation>(corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureSqrClippedReluCorpus<Implementation>(corpus, checksum);
}

template<SqrClippedReluBenchImplementation Implementation>
std::uint64_t ChecksumAllSqrClippedReluOutputs(
    const std::vector<SqrClippedReluBenchCase>& corpus) {
  alignas(kCacheLineSize) std::uint8_t output[32];
  std::uint64_t checksum = UINT64_C(14695981039346656037);

  for (const auto& sample : corpus) {
#if defined(SFNNwoPSQT)
    const auto& activation = network[sample.selected_bucket]->ac_sqr_0;
#else
    const auto& activation = network->ac_sqr_0;
#endif
    if constexpr (Implementation == SqrClippedReluBenchImplementation::Avx2)
      activation.BenchmarkPropagateAvx2(sample.input.data(), output);
    else
      activation.BenchmarkPropagateBaseline(sample.input.data(), output);

    for (const std::uint8_t value : output)
      MixNnueBenchChecksum(checksum, value);
  }
  return checksum;
}

void TestSqrClippedReluBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: SqrClippedReLU SSE2 / AVX2 comparison]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : even=SSE2,AVX2 odd=AVX2,SSE2" << std::endl;

  const auto corpus = MakeSqrClippedReluBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: SqrClippedReLU benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  NnueBenchSamples baseline_samples;
  NnueBenchSamples avx2_samples;
  std::uint64_t baseline_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t avx2_timing_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      baseline_samples.Add(MeasureSqrClippedReluCorpusAfterWarmup<
          SqrClippedReluBenchImplementation::Baseline>(
              corpus, baseline_timing_checksum));
      avx2_samples.Add(MeasureSqrClippedReluCorpusAfterWarmup<
          SqrClippedReluBenchImplementation::Avx2>(
              corpus, avx2_timing_checksum));
    } else {
      avx2_samples.Add(MeasureSqrClippedReluCorpusAfterWarmup<
          SqrClippedReluBenchImplementation::Avx2>(
              corpus, avx2_timing_checksum));
      baseline_samples.Add(MeasureSqrClippedReluCorpusAfterWarmup<
          SqrClippedReluBenchImplementation::Baseline>(
              corpus, baseline_timing_checksum));
    }
  }

  PrintNnueBenchSamples("existing SSE2", baseline_samples);
  PrintNnueBenchSamples("AVX2", avx2_samples);

  const NnueBenchSummary baseline_summary =
      SummarizeNnueBenchSamples(baseline_samples);
  const NnueBenchSummary avx2_summary = SummarizeNnueBenchSamples(avx2_samples);
  const double difference = avx2_summary.median - baseline_summary.median;
  const double improvement = baseline_summary.median == 0.0
      ? 0.0
      : (baseline_summary.median - avx2_summary.median)
            * 100.0 / baseline_summary.median;

  const std::uint64_t baseline_checksum =
      ChecksumAllSqrClippedReluOutputs<
          SqrClippedReluBenchImplementation::Baseline>(corpus);
  const std::uint64_t avx2_checksum =
      ChecksumAllSqrClippedReluOutputs<
          SqrClippedReluBenchImplementation::Avx2>(corpus);

  std::cout << "  difference (AVX2 - SSE2) : " << std::fixed
            << std::setprecision(1) << difference << " ns/call" << std::endl
            << "  improvement              : " << std::setprecision(2)
            << improvement << "%" << std::endl
            << "  SSE2 checksum : 0x" << std::hex << baseline_checksum
            << std::endl
            << "  AVX2 checksum : 0x" << avx2_checksum << std::dec << std::endl
            << "  checksum match: "
            << (baseline_checksum == avx2_checksum ? "yes" : "NO")
            << std::endl
            << "  timing checksum match: "
            << (baseline_timing_checksum == avx2_timing_checksum ? "yes"
                                                                 : "NO")
            << std::endl;
}

#if defined(USE_AVX2) && !defined(USE_AVX512)

struct Fc0PreparedNnz {
  std::array<std::uint16_t, Network::kBenchmarkFc0InputBlocks> indices{};
  IndexType count = 0;
};

std::vector<Fc0PreparedNnz> PrepareFc0NnzCorpus(
    const std::vector<NetworkBenchCase>& corpus) {
  std::vector<Fc0PreparedNnz> prepared(corpus.size());
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    prepared[sample_index].count = selected_network.BenchmarkMainFc0FindNnz(
        sample.transformed.data(), prepared[sample_index].indices.data());
  }
  return prepared;
}

double Fc0Percentile(const std::vector<IndexType>& sorted_counts,
                     const double percentile) {
  if (sorted_counts.empty())
    return 0.0;
  const double position = percentile * 0.01
      * static_cast<double>(sorted_counts.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, sorted_counts.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return static_cast<double>(sorted_counts[lower]) * (1.0 - fraction)
      + static_cast<double>(sorted_counts[upper]) * fraction;
}

void PrintFc0ActiveBlockDensity(
    const std::vector<Fc0PreparedNnz>& prepared) {
  std::vector<IndexType> counts;
  counts.reserve(prepared.size());
  std::array<std::uint64_t, 10> histogram{};
  double total = 0.0;
  for (const auto& sample : prepared) {
    counts.push_back(sample.count);
    total += static_cast<double>(sample.count);
    const std::size_t bin = std::min<std::size_t>(
        static_cast<std::size_t>(sample.count) * 10
            / Network::kBenchmarkFc0InputBlocks,
        histogram.size() - 1);
    ++histogram[bin];
  }
  std::sort(counts.begin(), counts.end());

  const double mean = counts.empty()
      ? 0.0
      : total / static_cast<double>(counts.size());
  const double median = Fc0Percentile(counts, 50.0);
  const auto print_value = [](const char* const name, const double value) {
    const double rate = 100.0 * value
        / static_cast<double>(Network::kBenchmarkFc0InputBlocks);
    std::cout << "  " << std::left << std::setw(6) << name << std::right
              << ": " << std::fixed << std::setprecision(1) << value
              << " / " << Network::kBenchmarkFc0InputBlocks << " ("
              << std::setprecision(2) << rate << "%)" << std::endl;
  };

  std::cout << "[fc_0 active 4-byte block density]" << std::endl;
  print_value("mean", mean);
  print_value("median", median);
  print_value("min", counts.empty() ? 0.0 : counts.front());
  print_value("max", counts.empty() ? 0.0 : counts.back());
  print_value("p10", Fc0Percentile(counts, 10.0));
  print_value("p50", Fc0Percentile(counts, 50.0));
  print_value("p90", Fc0Percentile(counts, 90.0));
  print_value("p95", Fc0Percentile(counts, 95.0));

  std::cout << "  histogram" << std::endl;
  for (std::size_t bin = 0; bin < histogram.size(); ++bin) {
    const std::uint64_t count = histogram[bin];
    const double fraction = prepared.empty()
        ? 0.0
        : 100.0 * static_cast<double>(count)
            / static_cast<double>(prepared.size());
    std::cout << "    " << std::setw(3) << bin * 10 << "-"
              << std::setw(3) << (bin + 1) * 10 << "% : "
              << std::setw(5) << count << " (" << std::fixed
              << std::setprecision(2) << fraction << "%)" << std::endl;
  }
}

enum class Fc0BenchImplementation {
  CurrentSparse,
  StreamingSparse,
  Dense,
};

template<Fc0BenchImplementation Implementation>
NnueBenchTiming MeasureFc0Corpus(
    const std::vector<NetworkBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    if constexpr (Implementation == Fc0BenchImplementation::CurrentSparse)
      selected_network.BenchmarkMainFc0(sample.transformed.data(), output);
    else if constexpr (
        Implementation == Fc0BenchImplementation::StreamingSparse)
      selected_network.BenchmarkMainFc0StreamingSparse(
          sample.transformed.data(), output);
    else
      selected_network.BenchmarkMainFc0Dense(
          sample.transformed.data(), output);

    const std::size_t index = static_cast<std::size_t>(timing.calls) & 31;
    MixNnueBenchChecksum(checksum, output[index]);
    MixNnueBenchChecksum(checksum, output[(index + 17) & 31]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<Fc0BenchImplementation Implementation>
NnueBenchTiming MeasureFc0CorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus,
    std::uint64_t& checksum) {
  MeasureFc0Corpus<Implementation>(corpus, checksum);
  return MeasureFc0Corpus<Implementation>(corpus, checksum);
}

NnueBenchTiming MeasureFc0FindNnzCorpus(
    const std::vector<NetworkBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize)
      std::uint16_t nnz[Network::kBenchmarkFc0InputBlocks];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    const IndexType count = selected_network.BenchmarkMainFc0FindNnz(
        sample.transformed.data(), nnz);
    MixNnueBenchChecksum(checksum, count);
    if (count != 0)
      MixNnueBenchChecksum(
          checksum, nnz[static_cast<std::size_t>(timing.calls) % count]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureFc0FindNnzCorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus,
    std::uint64_t& checksum) {
  MeasureFc0FindNnzCorpus(corpus, checksum);
  return MeasureFc0FindNnzCorpus(corpus, checksum);
}

NnueBenchTiming MeasureFc0PreparedAccumulateCorpus(
    const std::vector<NetworkBenchCase>& corpus,
    const std::vector<Fc0PreparedNnz>& prepared,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& nnz = prepared[sample_index];
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    selected_network.BenchmarkMainFc0AccumulatePreparedNnz(
        sample.transformed.data(), nnz.indices.data(), nnz.count, output);
    const std::size_t index = static_cast<std::size_t>(timing.calls) & 31;
    MixNnueBenchChecksum(checksum, output[index]);
    MixNnueBenchChecksum(checksum, output[(index + 17) & 31]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureFc0PreparedAccumulateCorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus,
    const std::vector<Fc0PreparedNnz>& prepared,
    std::uint64_t& checksum) {
  MeasureFc0PreparedAccumulateCorpus(corpus, prepared, checksum);
  return MeasureFc0PreparedAccumulateCorpus(corpus, prepared, checksum);
}

struct Fc0Mismatch {
  std::uint64_t count = 0;
  std::size_t first_sample = 0;
  IndexType first_output = 0;
  std::int32_t current_value = 0;
  std::int32_t candidate_value = 0;
};

void RecordFc0Mismatch(Fc0Mismatch& mismatch,
                       const std::size_t sample_index,
                       const IndexType output_index,
                       const std::int32_t current_value,
                       const std::int32_t candidate_value) {
  if (current_value == candidate_value)
    return;
  if (mismatch.count == 0) {
    mismatch.first_sample = sample_index;
    mismatch.first_output = output_index;
    mismatch.current_value = current_value;
    mismatch.candidate_value = candidate_value;
  }
  ++mismatch.count;
}

void ValidateFc0Implementations(
    const std::vector<NetworkBenchCase>& corpus,
    std::uint64_t& current_checksum,
    std::uint64_t& streaming_checksum,
    std::uint64_t& dense_checksum,
    Fc0Mismatch& streaming_mismatch,
    Fc0Mismatch& dense_mismatch) {
  alignas(kCacheLineSize) std::int32_t current_output[32];
  alignas(kCacheLineSize) std::int32_t streaming_output[32];
  alignas(kCacheLineSize) std::int32_t dense_output[32];
  current_checksum = UINT64_C(14695981039346656037);
  streaming_checksum = UINT64_C(14695981039346656037);
  dense_checksum = UINT64_C(14695981039346656037);

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    selected_network.BenchmarkMainFc0(
        sample.transformed.data(), current_output);
    selected_network.BenchmarkMainFc0StreamingSparse(
        sample.transformed.data(), streaming_output);
    selected_network.BenchmarkMainFc0Dense(
        sample.transformed.data(), dense_output);

    for (IndexType output_index = 0; output_index < 32; ++output_index) {
      MixNnueBenchChecksum(current_checksum, current_output[output_index]);
      MixNnueBenchChecksum(
          streaming_checksum, streaming_output[output_index]);
      MixNnueBenchChecksum(dense_checksum, dense_output[output_index]);
      RecordFc0Mismatch(streaming_mismatch, sample_index, output_index,
                        current_output[output_index],
                        streaming_output[output_index]);
      RecordFc0Mismatch(dense_mismatch, sample_index, output_index,
                        current_output[output_index],
                        dense_output[output_index]);
    }
  }
}

void PrintFc0Comparison(const char* const name,
                        const NnueBenchSamples& samples,
                        const double current_median) {
  PrintNnueBenchSamples(name, samples);
  const double median = SummarizeNnueBenchSamples(samples).median;
  const double relative = current_median == 0.0
      ? 0.0 : median * 100.0 / current_median;
  const double improvement = current_median == 0.0
      ? 0.0 : (current_median - median) * 100.0 / current_median;
  std::cout << "  relative to current: " << std::fixed
            << std::setprecision(2) << relative << "%" << std::endl
            << "  improvement        : " << improvement << "%" << std::endl;
}

void PrintFc0Mismatch(const char* const name,
                      const Fc0Mismatch& mismatch) {
  std::cout << "  " << name << " mismatch count : " << mismatch.count
            << std::endl;
  if (mismatch.count != 0) {
    std::cout << "    first mismatch sample : " << mismatch.first_sample
              << std::endl
              << "    first mismatch output : " << mismatch.first_output
              << std::endl
              << "    current sparse value  : " << mismatch.current_value
              << std::endl
              << "    candidate value       : " << mismatch.candidate_value
              << std::endl;
  }
}

void TestFc0SparseDenseBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Main fc_0 sparse / streaming / dense]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : A/B/C rotated by one per repeat" << std::endl;

  const auto corpus = MakeNnueNetworkBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: Main fc_0 benchmark corpus is empty" << std::endl;
    return;
  }
  const auto prepared = PrepareFc0NnzCorpus(corpus);
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  input blocks : " << Network::kBenchmarkFc0InputBlocks
            << " x 4 bytes" << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;
  PrintFc0ActiveBlockDensity(prepared);

  std::uint64_t current_checksum;
  std::uint64_t streaming_checksum;
  std::uint64_t dense_checksum;
  Fc0Mismatch streaming_mismatch;
  Fc0Mismatch dense_mismatch;
  ValidateFc0Implementations(
      corpus, current_checksum, streaming_checksum, dense_checksum,
      streaming_mismatch, dense_mismatch);

  NnueBenchSamples current_samples;
  NnueBenchSamples streaming_samples;
  NnueBenchSamples dense_samples;
  NnueBenchSamples find_samples;
  NnueBenchSamples prepared_accumulate_samples;
  std::uint64_t current_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t streaming_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t dense_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t find_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t accumulate_timing_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      const std::size_t operation = (repeat + offset) % 3;
      if (operation == 0)
        current_samples.Add(MeasureFc0CorpusAfterWarmup<
            Fc0BenchImplementation::CurrentSparse>(
                corpus, current_timing_checksum));
      else if (operation == 1)
        streaming_samples.Add(MeasureFc0CorpusAfterWarmup<
            Fc0BenchImplementation::StreamingSparse>(
                corpus, streaming_timing_checksum));
      else
        dense_samples.Add(MeasureFc0CorpusAfterWarmup<
            Fc0BenchImplementation::Dense>(
                corpus, dense_timing_checksum));
    }

    if ((repeat & 1) == 0) {
      find_samples.Add(MeasureFc0FindNnzCorpusAfterWarmup(
          corpus, find_timing_checksum));
      prepared_accumulate_samples.Add(
          MeasureFc0PreparedAccumulateCorpusAfterWarmup(
              corpus, prepared, accumulate_timing_checksum));
    } else {
      prepared_accumulate_samples.Add(
          MeasureFc0PreparedAccumulateCorpusAfterWarmup(
              corpus, prepared, accumulate_timing_checksum));
      find_samples.Add(MeasureFc0FindNnzCorpusAfterWarmup(
          corpus, find_timing_checksum));
    }
  }

  const double current_median =
      SummarizeNnueBenchSamples(current_samples).median;
  PrintFc0Comparison("A. current sparse", current_samples, current_median);
  PrintFc0Comparison("B. streaming sparse", streaming_samples,
                     current_median);
  PrintFc0Comparison("C. dense", dense_samples, current_median);
  PrintNnueBenchSamples("A1. find_nnz only", find_samples);
  PrintNnueBenchSamples("A2. accumulate from prepared nnz",
                        prepared_accumulate_samples);
  std::cout << "  note: isolated A1/A2 medians need not sum exactly to A;"
            << std::endl
            << "        cache state and prepared-index memory access differ."
            << std::endl;

  std::cout << "[fc_0 correctness]" << std::endl
            << "  A checksum : 0x" << std::hex << current_checksum
            << std::endl
            << "  B checksum : 0x" << streaming_checksum << std::endl
            << "  C checksum : 0x" << dense_checksum << std::dec << std::endl
            << "  A vs B checksum match : "
            << (current_checksum == streaming_checksum ? "yes" : "NO")
            << std::endl
            << "  A vs C checksum match : "
            << (current_checksum == dense_checksum ? "yes" : "NO")
            << std::endl;
  PrintFc0Mismatch("A vs B", streaming_mismatch);
  PrintFc0Mismatch("A vs C", dense_mismatch);
  std::cout << "  A timing checksum : 0x" << std::hex
            << current_timing_checksum << std::endl
            << "  B timing checksum : 0x" << streaming_timing_checksum
            << std::endl
            << "  C timing checksum : 0x" << dense_timing_checksum
            << std::dec << std::endl
            << "  A/B/C timing checksum match: "
            << (current_timing_checksum == streaming_timing_checksum
                    && current_timing_checksum == dense_timing_checksum
                ? "yes" : "NO")
            << std::endl;
}

NnueBenchTiming MeasureFc0TwoBankAccumulateCorpus(
    const std::vector<NetworkBenchCase>& corpus,
    const std::vector<Fc0PreparedNnz>& prepared,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& nnz = prepared[sample_index];
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    selected_network.BenchmarkMainFc0AccumulatePreparedNnzTwoBank(
        sample.transformed.data(), nnz.indices.data(), nnz.count, output);
    const std::size_t index = static_cast<std::size_t>(timing.calls) & 31;
    MixNnueBenchChecksum(checksum, output[index]);
    MixNnueBenchChecksum(checksum, output[(index + 17) & 31]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureFc0TwoBankAccumulateCorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus,
    const std::vector<Fc0PreparedNnz>& prepared,
    std::uint64_t& checksum) {
  MeasureFc0TwoBankAccumulateCorpus(corpus, prepared, checksum);
  return MeasureFc0TwoBankAccumulateCorpus(corpus, prepared, checksum);
}

void ValidateFc0AccumulatorImplementations(
    const std::vector<NetworkBenchCase>& corpus,
    const std::vector<Fc0PreparedNnz>& prepared,
    std::uint64_t& current_checksum,
    std::uint64_t& two_bank_checksum,
    Fc0Mismatch& mismatch) {
  alignas(kCacheLineSize) std::int32_t current_output[32];
  alignas(kCacheLineSize) std::int32_t two_bank_output[32];
  current_checksum = UINT64_C(14695981039346656037);
  two_bank_checksum = UINT64_C(14695981039346656037);

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& nnz = prepared[sample_index];
    const auto& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    selected_network.BenchmarkMainFc0AccumulatePreparedNnz(
        sample.transformed.data(), nnz.indices.data(), nnz.count,
        current_output);
    selected_network.BenchmarkMainFc0AccumulatePreparedNnzTwoBank(
        sample.transformed.data(), nnz.indices.data(), nnz.count,
        two_bank_output);

    for (IndexType output_index = 0; output_index < 32; ++output_index) {
      MixNnueBenchChecksum(current_checksum, current_output[output_index]);
      MixNnueBenchChecksum(two_bank_checksum, two_bank_output[output_index]);
      RecordFc0Mismatch(mismatch, sample_index, output_index,
                        current_output[output_index],
                        two_bank_output[output_index]);
    }
  }
}

void TestFc0AccumulatorBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Main fc_0 accumulators]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  range        : prepared-nnz accumulation only" << std::endl
            << "  order        : even=A,B odd=B,A" << std::endl;

  const auto corpus = MakeNnueNetworkBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: Main fc_0 accumulator benchmark corpus is empty"
              << std::endl;
    return;
  }
  const auto prepared = PrepareFc0NnzCorpus(corpus);
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  std::uint64_t current_checksum;
  std::uint64_t two_bank_checksum;
  Fc0Mismatch mismatch;
  ValidateFc0AccumulatorImplementations(
      corpus, prepared, current_checksum, two_bank_checksum, mismatch);

  NnueBenchSamples current_samples;
  NnueBenchSamples two_bank_samples;
  std::uint64_t current_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t two_bank_timing_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      current_samples.Add(MeasureFc0PreparedAccumulateCorpusAfterWarmup(
          corpus, prepared, current_timing_checksum));
      two_bank_samples.Add(MeasureFc0TwoBankAccumulateCorpusAfterWarmup(
          corpus, prepared, two_bank_timing_checksum));
    } else {
      two_bank_samples.Add(MeasureFc0TwoBankAccumulateCorpusAfterWarmup(
          corpus, prepared, two_bank_timing_checksum));
      current_samples.Add(MeasureFc0PreparedAccumulateCorpusAfterWarmup(
          corpus, prepared, current_timing_checksum));
    }
  }

  const NnueBenchSummary current_summary =
      SummarizeNnueBenchSamples(current_samples);
  const NnueBenchSummary two_bank_summary =
      SummarizeNnueBenchSamples(two_bank_samples);
  PrintNnueBenchSamples("A. current 4 accumulators", current_samples);
  PrintNnueBenchSamples("B. two banks / 8 accumulators", two_bank_samples);

  const double difference =
      two_bank_summary.median - current_summary.median;
  const double improvement = current_summary.median == 0.0
      ? 0.0
      : (current_summary.median - two_bank_summary.median)
          * 100.0 / current_summary.median;
  std::cout << "  difference (B - A) : " << std::fixed
            << std::setprecision(1) << difference << " ns/call" << std::endl
            << "  improvement        : " << std::setprecision(2)
            << improvement << "%" << std::endl
            << "[fc_0 accumulator correctness]" << std::endl
            << "  A checksum : 0x" << std::hex << current_checksum
            << std::endl
            << "  B checksum : 0x" << two_bank_checksum << std::dec
            << std::endl
            << "  checksum match : "
            << (current_checksum == two_bank_checksum ? "yes" : "NO")
            << std::endl;
  PrintFc0Mismatch("A vs B", mismatch);
  std::cout << "  A timing checksum : 0x" << std::hex
            << current_timing_checksum << std::endl
            << "  B timing checksum : 0x" << two_bank_timing_checksum
            << std::dec << std::endl
            << "  timing checksum match: "
            << (current_timing_checksum == two_bank_timing_checksum
                    ? "yes" : "NO")
            << std::endl;
}

template<bool UseTiledFc1>
NnueBenchTiming MeasureFc1OutputTilingCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[kHidden2Dims];
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    if constexpr (UseTiledFc1)
      selected_network.BenchmarkFc1OutputTiled(
          sample.intermediate.l2_input, output);
    else
      selected_network.BenchmarkFc1(sample.intermediate.l2_input, output);
    const std::size_t index =
        static_cast<std::size_t>(timing.calls) % kHidden2Dims;
    MixNnueBenchChecksum(checksum, output[index]);
    MixNnueBenchChecksum(
        checksum, output[(index + 47) % kHidden2Dims]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<bool UseTiledFc1>
NnueBenchTiming MeasureFc1OutputTilingCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  MeasureFc1OutputTilingCorpus<UseTiledFc1>(corpus, checksum);
  return MeasureFc1OutputTilingCorpus<UseTiledFc1>(corpus, checksum);
}

struct Fc1OutputTilingValidation {
  NnueBenchIntegerDiagnostic fc1;
  NnueBenchIntegerDiagnostic ac1;
  NnueBenchIntegerDiagnostic final_output;
  NnueBenchIntegerDiagnostic staged_output;
  NnueBenchIntegerDiagnostic normal_vs_tiled_staged;
};

Fc1OutputTilingValidation ValidateFc1OutputTiling(
    const std::vector<NetworkStageBenchCase>& corpus) {
  Fc1OutputTilingValidation validation;
  alignas(kCacheLineSize) Network::Buffer current_work{};
  alignas(kCacheLineSize) Network::Buffer tiled_work{};

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);

    selected_network.BenchmarkFc1(
        sample.intermediate.l2_input, current_work.fc_1_out);
    selected_network.BenchmarkFc1OutputTiled(
        sample.intermediate.l2_input, tiled_work.fc_1_out);
    AddNnueBenchIntegerDiagnostic(
        validation.fc1, current_work.fc_1_out, tiled_work.fc_1_out,
        kHidden2Dims, sample_index);

    selected_network.BenchmarkAc1(
        current_work.fc_1_out, current_work.ac_1_out);
    selected_network.BenchmarkAc1(
        tiled_work.fc_1_out, tiled_work.ac_1_out);
    AddNnueBenchIntegerDiagnostic(
        validation.ac1, current_work.ac_1_out, tiled_work.ac_1_out,
        kHidden2Dims, sample_index);

    selected_network.BenchmarkFc2(
        current_work.ac_1_out, current_work.fc_2_out);
    selected_network.BenchmarkFc2(
        tiled_work.ac_1_out, tiled_work.fc_2_out);
    const std::int32_t current_final = selected_network.BenchmarkBlend(
        sample.intermediate.fc_0_out[31], current_work.fc_2_out[0]);
    const std::int32_t tiled_final = selected_network.BenchmarkBlend(
        sample.intermediate.fc_0_out[31], tiled_work.fc_2_out[0]);
    AddNnueBenchIntegerDiagnostic(
        validation.final_output, &current_final, &tiled_final, 1,
        sample_index);

    const std::int32_t current_staged =
        ComputeNnueNetworkStagedOutput<false>(sample, current_work);
    const std::int32_t tiled_staged =
        ComputeNnueNetworkStagedOutput<true>(sample, tiled_work);
    AddNnueBenchIntegerDiagnostic(
        validation.staged_output, &current_staged, &tiled_staged, 1,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        validation.normal_vs_tiled_staged, &sample.final_output,
        &tiled_staged, 1, sample_index);
  }
  return validation;
}

void TestFc1OutputTilingBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: fc_1 current / output-tiled]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  range        : fc_1 only (192 -> 96)" << std::endl
            << "  candidate    : 64 outputs + 32 outputs" << std::endl
            << "  order        : even=current,tiled odd=tiled,current"
            << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: fc_1 output-tiling benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  const auto validation = ValidateFc1OutputTiling(corpus);
  NnueBenchSamples current_samples;
  NnueBenchSamples tiled_samples;
  std::uint64_t current_timing_checksum = UINT64_C(14695981039346656037);
  std::uint64_t tiled_timing_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      current_samples.Add(MeasureFc1OutputTilingCorpusAfterWarmup<false>(
          corpus, current_timing_checksum));
      tiled_samples.Add(MeasureFc1OutputTilingCorpusAfterWarmup<true>(
          corpus, tiled_timing_checksum));
    } else {
      tiled_samples.Add(MeasureFc1OutputTilingCorpusAfterWarmup<true>(
          corpus, tiled_timing_checksum));
      current_samples.Add(MeasureFc1OutputTilingCorpusAfterWarmup<false>(
          corpus, current_timing_checksum));
    }
  }

  const auto current_summary = SummarizeNnueBenchSamples(current_samples);
  const auto tiled_summary = SummarizeNnueBenchSamples(tiled_samples);
  PrintNnueBenchSamples("A. current 12 accumulators", current_samples);
  PrintNnueBenchSamples("B. output-tiled 8 + 4 accumulators",
                        tiled_samples);
  const double difference = tiled_summary.median - current_summary.median;
  const double improvement = current_summary.median == 0.0
      ? 0.0
      : (current_summary.median - tiled_summary.median)
          * 100.0 / current_summary.median;
  std::cout << "  difference (B - A) : " << std::fixed
            << std::setprecision(1) << difference << " ns/call" << std::endl
            << "  improvement        : " << std::setprecision(2)
            << improvement << "%" << std::endl
            << "  A timing checksum  : 0x" << std::hex
            << current_timing_checksum << std::endl
            << "  B timing checksum  : 0x" << tiled_timing_checksum
            << std::dec << std::endl
            << "  timing checksum match: "
            << (current_timing_checksum == tiled_timing_checksum
                    ? "yes" : "NO")
            << std::endl;

  std::cout << "[fc_1 output-tiling correctness]" << std::endl;
  PrintNnueBenchIntegerDiagnostic("fc_1_out[96] A vs B", validation.fc1);
  PrintNnueBenchIntegerDiagnostic("ac_1_out[96] A vs B", validation.ac1);
  PrintNnueBenchIntegerDiagnostic(
      "final Network output A vs B", validation.final_output);
  PrintNnueBenchIntegerDiagnostic(
      "staged end-to-end A vs B", validation.staged_output);
  PrintNnueBenchIntegerDiagnostic(
      "normal vs tiled staged end-to-end",
      validation.normal_vs_tiled_staged);
  std::cout << "  note: isolated LCA float re-evaluation is intentionally not"
            << std::endl
            << "        part of the fc_1 candidate correctness decision."
            << std::endl;
}

#endif  // defined(USE_AVX2) && !defined(USE_AVX512)

#endif  // defined(ENABLE_NNUE_BENCH)

#if defined(ENABLE_NNUE_TRACE)

constexpr std::size_t kTraceFmDimensions = 32;
constexpr std::size_t kTraceFmOutputDimensions = 4 * kTraceFmDimensions;
constexpr std::size_t kTracePairDimensions = 640;
constexpr std::size_t kTraceRouterInputDimensions = 384;
constexpr std::size_t kTraceRouterOutputDimensions = kLayerStacks;
constexpr std::size_t kTraceFmFcOutputDimensions = 64;
constexpr std::size_t kTraceFmHiddenDimensions = 32;
constexpr std::size_t kTraceLcaQueryInputDimensions = 31;
constexpr std::size_t kTraceLcaFmInputDimensions = 64;
constexpr std::size_t kTracePhaseDimensions = 6;
constexpr std::size_t kTraceCrossInputDimensions = 16;
constexpr std::size_t kTraceBucketInputDimensions = 192;
constexpr std::size_t kTraceBucketHiddenDimensions = 96;

static_assert(FeatureTransformer::kOutputDimensions ==
                  2 * kTracePairDimensions,
              "NNUE trace expects the 1280-dimensional main path");
static_assert(Router::kInputDimensions == kTraceRouterInputDimensions,
              "NNUE trace expects the 384-dimensional Router input");
static_assert(Router::kOutputDimensions >= kTraceRouterOutputDimensions,
              "NNUE Router output is smaller than the layer-stack count");

struct TraceFmAccumulator {
  std::array<std::int64_t, kTraceFmDimensions> halfka_sum_v;
  std::array<std::int64_t, kTraceFmDimensions> halfka_sum_v2;
  std::array<std::int64_t, kTraceFmDimensions> ksdg_sum_v;
  std::array<std::int64_t, kTraceFmDimensions> ksdg_sum_v2;
};

struct TraceFmInteraction {
  std::array<std::int64_t, kTraceFmDimensions> ih;
  std::array<std::int64_t, kTraceFmDimensions> ik;
  std::array<std::int64_t, kTraceFmDimensions> sh;
  std::array<std::int64_t, kTraceFmDimensions> sk;
};

struct TraceMainPair {
  std::array<std::int32_t, kTracePairDimensions> a;
  std::array<std::int32_t, kTracePairDimensions> b;
  std::array<std::int32_t, kTracePairDimensions> mul_term;
  std::array<std::int32_t, kTracePairDimensions> diff_sq_term;
  std::array<std::int32_t, kTracePairDimensions> sum_term;
  std::array<std::int32_t, kTracePairDimensions> mixed_numerator;
  std::array<FeatureTransformer::OutputType, kTracePairDimensions> output;
};

struct TraceRouter {
  std::array<std::uint8_t, kTraceRouterInputDimensions> input;
  std::array<std::int32_t, kTraceRouterOutputDimensions> logits;
  int selected_bucket;
};

struct TraceFmPath {
  int selected_bucket;
  alignas(kCacheLineSize)
      std::array<std::uint8_t, kTraceFmOutputDimensions> diff_input;
  alignas(kCacheLineSize)
      std::array<std::uint8_t, kTraceFmOutputDimensions> abs_input;
  alignas(kCacheLineSize)
      std::array<std::int32_t, kTraceFmFcOutputDimensions> diff_fc_preact;
  alignas(kCacheLineSize)
      std::array<std::int32_t, kTraceFmFcOutputDimensions> abs_fc_preact;
  std::array<std::int32_t, kTraceFmHiddenDimensions> diff_gate_preact;
  std::array<std::int32_t, kTraceFmHiddenDimensions> diff_value_preact;
  std::uint32_t diff_rms_sum_sq_f32_bits;
  std::uint32_t diff_inv_rms_f32_bits;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      diff_normalized_f32_bits;
  std::array<std::int32_t, kTraceFmHiddenDimensions>
      diff_normalized_scaled_centered;
  std::array<std::uint8_t, kTraceFmHiddenDimensions> diff_output_pre_lca;
  std::array<std::int32_t, kTraceFmHiddenDimensions> diff_main_gate_q64;
  std::array<std::int32_t, kTraceFmHiddenDimensions>
      diff_main_gate_multiplier_q128;
  std::array<std::int32_t, kTraceFmHiddenDimensions> abs_gate_preact;
  std::array<std::int32_t, kTraceFmHiddenDimensions> abs_value_preact;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      abs_gate_sigmoid_f32_bits;
  std::array<std::int32_t, kTraceFmHiddenDimensions> abs_gated_value;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      abs_scaled_before_round_f32_bits;
  std::array<std::uint8_t, kTraceFmHiddenDimensions> abs_output;
  std::array<std::uint8_t, kTraceFmHiddenDimensions> abs_squared_output;
};

struct TraceLca {
  int selected_bucket;
  std::array<std::int32_t, kTraceFmHiddenDimensions>
      main_fc_preact_before_gate;
  std::array<std::int32_t, kTraceFmHiddenDimensions>
      main_fc_preact_after_gate;
  std::array<std::uint8_t, kTraceLcaQueryInputDimensions> query_input;
  std::array<std::uint8_t, kTraceLcaFmInputDimensions> fm_input;
  std::array<std::int32_t, kTraceFmHiddenDimensions> query_preact;
  std::array<std::int32_t, kTraceFmHiddenDimensions> key_preact;
  std::array<std::int32_t, kTraceFmHiddenDimensions> value_preact;
  std::uint32_t temperature_f32_bits;
  std::uint32_t dot_product_f32_bits;
  std::uint32_t attention_logit_f32_bits;
  std::uint32_t attention_score_f32_bits;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      value_clamped_f32_bits;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      correction_f32_bits;
  std::array<std::uint32_t, kTraceFmHiddenDimensions>
      output_post_lca_f32_bits;
  std::array<std::uint8_t, kTraceFmHiddenDimensions> output_post_lca;
};

struct TraceDeepPath {
  int selected_bucket;
  std::array<std::uint8_t, kTraceRouterInputDimensions> phase_input;
  std::array<std::int32_t, kTracePhaseDimensions> phase_preact;
  std::array<std::uint32_t, kTracePhaseDimensions> phase_logit_f32_bits;
  std::array<std::uint32_t, kTracePhaseDimensions> phase_sigmoid_f32_bits;
  std::array<std::uint32_t, kTracePhaseDimensions> phase_value_f32_bits;
  std::array<std::uint32_t, kTracePhaseDimensions> channel_scale_f32_bits;
  std::array<std::uint8_t, kTraceLcaQueryInputDimensions> main_raw;
  std::array<std::uint8_t, kTraceLcaQueryInputDimensions> main_squared;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_main_squared;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_diff;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_main_raw;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_abs;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_product_diff;
  std::array<std::uint8_t, kTraceCrossInputDimensions> cross_product_abs;
  std::array<std::uint8_t, 2 * kTraceCrossInputDimensions> cross_input;
  std::array<std::int32_t, kTraceFmHiddenDimensions> cross_preact;
  std::array<std::uint8_t, kTraceFmHiddenDimensions> cross_output;
  std::array<std::uint8_t, kTraceBucketInputDimensions> fc1_input;
  std::array<std::int32_t, kTraceBucketHiddenDimensions> fc1_preact;
  std::array<std::uint8_t, kTraceBucketHiddenDimensions> fc1_output;
  std::array<std::int32_t, 1> fc2_preact;
};

struct TraceFinalPath {
  int selected_bucket;
  int material_bucket;
  std::array<std::int32_t, 1> deep_output;
  std::array<std::int32_t, 1> bypass_input;
  std::array<std::int32_t, 1> bypass_preact;
  std::array<std::int32_t, 1> bypass_scaled_numerator;
  std::array<std::int32_t, 1> bypass_output;
  std::array<std::int32_t, 1> alpha_q14;
  std::array<std::int32_t, 1> inv_alpha_q14;
  std::array<std::int64_t, 1> deep_term;
  std::array<std::int64_t, 1> bypass_term;
  std::array<std::int64_t, 1> blend_numerator;
  std::array<std::int32_t, 1> blend_output;
  std::array<std::int32_t, 1> network_output;
  std::array<std::int32_t, 1> fv_scale;
  std::array<std::int32_t, 1> eval_before_clamp;
  std::array<std::int32_t, 1> value_max_eval;
  std::array<std::int32_t, 1> eval_after_clamp;
};

struct TracePerspectiveData {
  std::vector<IndexType> active_indices;
  std::array<std::int16_t, kTransformedFeatureDimensions> main_accumulator;
  TraceFmAccumulator fm_accumulator;
  TraceFmInteraction fm_interaction;
  TraceMainPair main_pair;
};

struct NnueTraceSnapshot {
  std::string sfen;
  Color side_to_move;
  int pair_bucket;
  std::array<TracePerspectiveData, COLOR_NB> perspective;
  std::array<std::int16_t, kTracePairDimensions> pair_weight_mul;
  std::array<std::int16_t, kTracePairDimensions> pair_weight_diff;
  std::array<std::int16_t, kTracePairDimensions> pair_weight_sum;
  TraceFmInteraction raw_diff;
  TraceFmInteraction raw_abs;
  std::array<FeatureTransformer::OutputType, kTraceFmOutputDimensions>
      scaled_diff;
  std::array<FeatureTransformer::OutputType, kTraceFmOutputDimensions>
      scaled_abs;
  TraceRouter router;
  TraceFmPath fm_path;
  TraceLca lca;
  TraceDeepPath deep_path;
  TraceFinalPath final_path;
};

const char* TracePerspectiveName(const Color perspective) {
  return perspective == BLACK ? "BLACK" : "WHITE";
}

int TracePairBucket(const Position& pos) {
  // Keep this trace-only calculation identical to stack_index_for_nnue().
  constexpr int bucket_by_material[24] = {
      0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8,
      8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11};
  return bucket_by_material[std::min(
      (std::abs(pos.state()->materialValue) + 99) / 100, 23)];
}

std::uint32_t TraceFloatBits(const float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t Fnv1a64Indices(const std::vector<IndexType>& indices) {
  constexpr std::uint64_t kOffsetBasis = UINT64_C(14695981039346656037);
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  std::uint64_t hash = kOffsetBasis;
  for (const IndexType index : indices) {
    const std::uint32_t value = static_cast<std::uint32_t>(index);
    for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
      hash ^= static_cast<std::uint8_t>(value >> (byte_index * 8));
      hash *= kPrime;
    }
  }
  return hash;
}

bool MakeNnueTraceSnapshot(const std::string& sfen,
                           NnueTraceSnapshot* const snapshot,
                           std::string* const error_message) {
  if (!feature_transformer) {
    *error_message = "NNUE feature transformer is not loaded";
    return false;
  }
  if (!router) {
    *error_message = "NNUE router is not loaded";
    return false;
  }

  Position trace_position;
  StateInfo trace_state;
  trace_position.set(sfen, &trace_state);

  snapshot->sfen = trace_position.sfen();
  snapshot->side_to_move = trace_position.side_to_move();
  snapshot->pair_bucket = TracePairBucket(trace_position);

  for (std::size_t trigger_index = 0;
       trigger_index < kRefreshTriggers.size(); ++trigger_index) {
    Features::IndexList active_indices[COLOR_NB];
    RawFeatures::AppendActiveIndices(
        trace_position, kRefreshTriggers[trigger_index], active_indices);
    for (const Color perspective : {BLACK, WHITE}) {
      auto& destination = snapshot->perspective[perspective].active_indices;
      destination.insert(destination.end(), active_indices[perspective].begin(),
                         active_indices[perspective].end());
    }
  }
  for (const Color perspective : {BLACK, WHITE}) {
    auto& indices = snapshot->perspective[perspective].active_indices;
    std::sort(indices.begin(), indices.end());
  }

  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions>
          transformed_output;
  feature_transformer->Transform(
      trace_position, transformed_output.data(), snapshot->scaled_diff.data(),
      snapshot->scaled_abs.data(), true, snapshot->pair_bucket);

  // Reproduce SelectBucketWithRouter() using the actual transformed byte
  // arrays and the already loaded Router. This remains trace-only code.
  for (std::size_t index = 0; index < kTraceFmOutputDimensions; ++index) {
    const std::int32_t abs_value = snapshot->scaled_abs[index];
    snapshot->router.input[index] = static_cast<std::uint8_t>(
        std::clamp((abs_value - 64) * 2, 0, 127));
    snapshot->router.input[index + kTraceFmOutputDimensions] =
        snapshot->scaled_diff[index];
    snapshot->router.input[index + 2 * kTraceFmOutputDimensions] =
        transformed_output[index];
  }

  alignas(kCacheLineSize) Router::OutputBuffer router_output;
  router->PropagatePrefix<12>(snapshot->router.input.data(), router_output);
  std::copy_n(router_output, kTraceRouterOutputDimensions,
              snapshot->router.logits.begin());
  snapshot->router.selected_bucket = 0;
  for (std::size_t bucket = 1; bucket < kTraceRouterOutputDimensions;
       ++bucket) {
    if (snapshot->router.logits[bucket] >
        snapshot->router.logits[snapshot->router.selected_bucket])
      snapshot->router.selected_bucket = static_cast<int>(bucket);
  }

  auto& fm_path = snapshot->fm_path;
  fm_path.selected_bucket = snapshot->router.selected_bucket;
  std::copy_n(snapshot->scaled_diff.begin(), kTraceFmOutputDimensions,
              fm_path.diff_input.begin());
  std::copy_n(snapshot->scaled_abs.begin(), kTraceFmOutputDimensions,
              fm_path.abs_input.begin());

  const auto& selected_network = network[fm_path.selected_bucket];
  if (!selected_network) {
    *error_message = "selected NNUE bucket network is not loaded";
    return false;
  }
  selected_network->fc_diff.Propagate(fm_path.diff_input.data(),
                                      fm_path.diff_fc_preact.data());
  selected_network->fc_abs.Propagate(fm_path.abs_input.data(),
                                     fm_path.abs_fc_preact.data());

  float diff_sum_sq = 0.0f;
  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    const float value =
        static_cast<float>(fm_path.diff_fc_preact[index +
                                                  kTraceFmHiddenDimensions]);
    diff_sum_sq += value * value;
  }
  const float diff_inv_rms =
      1.0f / std::sqrt(diff_sum_sq / 32.0f + 1e-8f);
  fm_path.diff_rms_sum_sq_f32_bits = TraceFloatBits(diff_sum_sq);
  fm_path.diff_inv_rms_f32_bits = TraceFloatBits(diff_inv_rms);

  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    const std::int32_t diff_gate = fm_path.diff_fc_preact[index];
    const std::int32_t diff_value =
        fm_path.diff_fc_preact[index + kTraceFmHiddenDimensions];
    const std::int32_t abs_gate = fm_path.abs_fc_preact[index];
    const std::int32_t abs_value =
        fm_path.abs_fc_preact[index + kTraceFmHiddenDimensions];

    fm_path.diff_gate_preact[index] = diff_gate;
    fm_path.diff_value_preact[index] = diff_value;
    fm_path.abs_gate_preact[index] = abs_gate;
    fm_path.abs_value_preact[index] = abs_value;

    const float diff_normalized =
        static_cast<float>(diff_value) * diff_inv_rms;
    const std::int32_t diff_centered =
        static_cast<std::int32_t>(diff_normalized * 25.4f);
    fm_path.diff_normalized_f32_bits[index] =
        TraceFloatBits(diff_normalized);
    fm_path.diff_normalized_scaled_centered[index] = diff_centered;
    fm_path.diff_output_pre_lca[index] = static_cast<std::uint8_t>(
        std::clamp(diff_centered + 64, 0, 127));

    const std::int32_t main_gate_q64 = Network::sigmoid_gate_slow(
        diff_gate - 2438, 64);
    fm_path.diff_main_gate_q64[index] = main_gate_q64;
    fm_path.diff_main_gate_multiplier_q128[index] = 64 + main_gate_q64;

    const float abs_gate_sigmoid =
        1.0f /
        (1.0f + std::exp(-static_cast<float>(abs_gate) / 8128.0f));
    const std::int32_t abs_gated =
        Network::sigmoid_gate_slow(abs_gate, abs_value);
    const float abs_gated_float =
        static_cast<float>(abs_gated) / 8128.0f;
    const float abs_scaled_before_round =
        std::clamp(abs_gated_float * 0.05f + 0.6f, 0.0f, 1.0f)
        * 127.0f;
    const std::int32_t abs_scaled =
        static_cast<std::int32_t>(std::round(abs_scaled_before_round));

    fm_path.abs_gate_sigmoid_f32_bits[index] =
        TraceFloatBits(abs_gate_sigmoid);
    fm_path.abs_gated_value[index] = abs_gated;
    fm_path.abs_scaled_before_round_f32_bits[index] =
        TraceFloatBits(abs_scaled_before_round);
    fm_path.abs_output[index] = static_cast<std::uint8_t>(abs_scaled);
    fm_path.abs_squared_output[index] = static_cast<std::uint8_t>(
        (abs_scaled * abs_scaled) / 127);
  }

  // Reproduce only the Main input dependency and LCA section of
  // Network::Propagate(). The deep bucket network remains outside this trace.
  auto& lca = snapshot->lca;
  lca.selected_bucket = fm_path.selected_bucket;
  alignas(kCacheLineSize) Network::Buffer lca_buffer{};

  selected_network->fc_0.Propagate(transformed_output.data(),
                                   lca_buffer.fc_0_out);
  std::copy_n(lca_buffer.fc_0_out, kTraceFmHiddenDimensions,
              lca.main_fc_preact_before_gate.begin());
  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    const std::int32_t main_gate_q64 = fm_path.diff_main_gate_q64[index];
    lca_buffer.fc_0_out[index] = static_cast<std::int32_t>(
        (lca_buffer.fc_0_out[index] * (64 + main_gate_q64)) / 128);
    if (index < kTraceLcaQueryInputDimensions)
      lca_buffer.fc_0_out[index] =
          std::clamp(lca_buffer.fc_0_out[index], 0, 8128);
  }
  std::copy_n(lca_buffer.fc_0_out, kTraceFmHiddenDimensions,
              lca.main_fc_preact_after_gate.begin());

  selected_network->ac_0.Propagate(lca_buffer.fc_0_out,
                                   lca_buffer.ac_0_out);
  std::copy_n(lca_buffer.ac_0_out, kTraceLcaQueryInputDimensions,
              lca.query_input.begin());

  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    lca_buffer.fm_cat_uint8[index] = fm_path.diff_output_pre_lca[index];
    lca_buffer.fm_cat_uint8[index + kTraceFmHiddenDimensions] =
        fm_path.abs_output[index];
  }
  std::copy_n(lca_buffer.fm_cat_uint8, kTraceLcaFmInputDimensions,
              lca.fm_input.begin());

  selected_network->lca_q.Propagate(lca_buffer.ac_0_out,
                                    lca_buffer.lca_q_out);
  selected_network->lca_k.Propagate(lca_buffer.fm_cat_uint8,
                                    lca_buffer.lca_k_out);
  selected_network->lca_v.Propagate(lca_buffer.fm_cat_uint8,
                                    lca_buffer.lca_v_out);
  std::copy_n(lca_buffer.lca_q_out, kTraceFmHiddenDimensions,
              lca.query_preact.begin());
  std::copy_n(lca_buffer.lca_k_out, kTraceFmHiddenDimensions,
              lca.key_preact.begin());
  std::copy_n(lca_buffer.lca_v_out, kTraceFmHiddenDimensions,
              lca.value_preact.begin());

  float lca_dot_product = 0.0f;
  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    lca_dot_product +=
        (static_cast<float>(lca_buffer.lca_q_out[index]) / 8128.0f)
        * (static_cast<float>(lca_buffer.lca_k_out[index]) / 8128.0f);
  }
  const float lca_attention_logit =
      (lca_dot_product * 0.17677f) / selected_network->lca_temp;
  const float lca_attention_score =
      1.0f / (1.0f + std::exp(-lca_attention_logit));
  lca.temperature_f32_bits = TraceFloatBits(selected_network->lca_temp);
  lca.dot_product_f32_bits = TraceFloatBits(lca_dot_product);
  lca.attention_logit_f32_bits = TraceFloatBits(lca_attention_logit);
  lca.attention_score_f32_bits = TraceFloatBits(lca_attention_score);

  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    const float current_diff =
        static_cast<float>(fm_path.diff_output_pre_lca[index]) / 127.0f;
    const float value =
        static_cast<float>(lca_buffer.lca_v_out[index]) / 8128.0f;
    const float value_clamped =
        std::clamp(value * 0.4f + 0.5f, 0.0f, 1.0f);
    const float output_post_lca =
        current_diff * (1.0f - lca_attention_score)
        + value_clamped * lca_attention_score;
    const float correction = output_post_lca - current_diff;

    lca.value_clamped_f32_bits[index] = TraceFloatBits(value_clamped);
    lca.correction_f32_bits[index] = TraceFloatBits(correction);
    lca.output_post_lca_f32_bits[index] = TraceFloatBits(output_post_lca);
    lca.output_post_lca[index] =
        static_cast<std::uint8_t>(output_post_lca * 127.0f);
    lca_buffer.diff_ac_out[index] = lca.output_post_lca[index];
    lca_buffer.abs_ac_out[index] = fm_path.abs_output[index];
    lca_buffer.abs_sqr_out[index] = fm_path.abs_squared_output[index];
  }

  auto& deep_path = snapshot->deep_path;
  deep_path.selected_bucket = lca.selected_bucket;

  for (std::size_t index = 0; index < kTraceFmOutputDimensions; ++index) {
    const std::int32_t abs_value = snapshot->scaled_abs[index];
    lca_buffer.phase_input[index] = static_cast<std::uint8_t>(
        std::clamp((abs_value - 64) * 2, 0, 127));
    lca_buffer.phase_input[index + kTraceFmOutputDimensions] =
        snapshot->scaled_diff[index];
    lca_buffer.phase_input[index + 2 * kTraceFmOutputDimensions] =
        transformed_output[index];
  }
  lca_buffer.phase_input[127] = static_cast<std::uint8_t>(
      (snapshot->pair_bucket * 127) / 11);
  std::copy_n(lca_buffer.phase_input, kTraceRouterInputDimensions,
              deep_path.phase_input.begin());

  selected_network->phase_proj.PropagatePrefix<6>(lca_buffer.phase_input,
                                                  lca_buffer.phase_out);
  float channel_scales[kTracePhaseDimensions];
  constexpr float scale_multipliers[kTracePhaseDimensions] = {
      1.3f, 1.5f, 1.0f, 0.7f, 0.88f, 1.5f};
  for (std::size_t index = 0; index < kTracePhaseDimensions; ++index) {
    const float phase_logit =
        (static_cast<float>(lca_buffer.phase_out[index]) / 8128.0f)
        * 3.0f + 1.0f;
    const float phase_sigmoid =
        1.0f / (1.0f + std::exp(-phase_logit));
    const float phase_value = 0.1f + 0.9f * phase_sigmoid;
    const float channel_scale =
        (0.5f + 0.5f * phase_value) * scale_multipliers[index];

    deep_path.phase_preact[index] = lca_buffer.phase_out[index];
    deep_path.phase_logit_f32_bits[index] = TraceFloatBits(phase_logit);
    deep_path.phase_sigmoid_f32_bits[index] = TraceFloatBits(phase_sigmoid);
    deep_path.phase_value_f32_bits[index] = TraceFloatBits(phase_value);
    deep_path.channel_scale_f32_bits[index] = TraceFloatBits(channel_scale);
    channel_scales[index] = channel_scale;
  }

  selected_network->ac_sqr_0.Propagate(lca_buffer.fc_0_out,
                                       lca_buffer.ac_sqr_0_out_temp);
  std::copy_n(lca_buffer.ac_0_out, kTraceLcaQueryInputDimensions,
              deep_path.main_raw.begin());
  std::copy_n(lca_buffer.ac_sqr_0_out_temp,
              kTraceLcaQueryInputDimensions,
              deep_path.main_squared.begin());

  for (std::size_t index = 0; index < kTraceCrossInputDimensions; ++index) {
    const std::uint8_t main_squared = lca_buffer.ac_sqr_0_out_temp[index];
    const std::uint8_t diff = lca_buffer.diff_ac_out[index];
    const std::uint8_t main_raw = lca_buffer.ac_0_out[index];
    const std::uint8_t abs = lca_buffer.abs_ac_out[index];
    const std::uint8_t product_diff = static_cast<std::uint8_t>(
        (main_squared * diff) / 127);
    const std::uint8_t product_abs = static_cast<std::uint8_t>(
        (main_raw * abs) / 127);

    deep_path.cross_main_squared[index] = main_squared;
    deep_path.cross_diff[index] = diff;
    deep_path.cross_main_raw[index] = main_raw;
    deep_path.cross_abs[index] = abs;
    deep_path.cross_product_diff[index] = product_diff;
    deep_path.cross_product_abs[index] = product_abs;
    lca_buffer.cross_cat[index] = product_diff;
    lca_buffer.cross_cat[index + kTraceCrossInputDimensions] = product_abs;
  }
  std::copy_n(lca_buffer.cross_cat, 2 * kTraceCrossInputDimensions,
              deep_path.cross_input.begin());
  selected_network->fc_cross.Propagate(lca_buffer.cross_cat,
                                       lca_buffer.cross_fc_out);
  selected_network->ac_cross.Propagate(lca_buffer.cross_fc_out,
                                       lca_buffer.cross_feat);
  std::copy_n(lca_buffer.cross_fc_out, kTraceFmHiddenDimensions,
              deep_path.cross_preact.begin());
  std::copy_n(lca_buffer.cross_feat, kTraceFmHiddenDimensions,
              deep_path.cross_output.begin());

  for (std::size_t index = 0; index < kTraceLcaQueryInputDimensions; ++index) {
    lca_buffer.l2_input[index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.ac_sqr_0_out_temp[index]
                            * channel_scales[0],
                        0, 127));
    lca_buffer.l2_input[index + kTraceLcaQueryInputDimensions] =
        static_cast<std::uint8_t>(
            std::clamp<int>(lca_buffer.ac_0_out[index]
                                * channel_scales[1],
                            0, 127));
  }
  for (std::size_t index = 0; index < kTraceFmHiddenDimensions; ++index) {
    lca_buffer.l2_input[62 + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.diff_ac_out[index] * channel_scales[2],
                        0, 127));
    lca_buffer.l2_input[94 + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.abs_ac_out[index] * channel_scales[3],
                        0, 127));
    lca_buffer.l2_input[126 + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.abs_sqr_out[index] * channel_scales[4],
                        0, 127));
    lca_buffer.l2_input[158 + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.cross_feat[index] * channel_scales[5],
                        0, 127));
  }
  std::memset(lca_buffer.l2_input + 190, 0, 2);
  std::copy_n(lca_buffer.l2_input, kTraceBucketInputDimensions,
              deep_path.fc1_input.begin());

  selected_network->fc_1.Propagate(lca_buffer.l2_input,
                                   lca_buffer.fc_1_out);
  selected_network->ac_1.Propagate(lca_buffer.fc_1_out,
                                   lca_buffer.ac_1_out);
  selected_network->fc_2.Propagate(lca_buffer.ac_1_out,
                                   lca_buffer.fc_2_out);
  std::copy_n(lca_buffer.fc_1_out, kTraceBucketHiddenDimensions,
              deep_path.fc1_preact.begin());
  std::copy_n(lca_buffer.ac_1_out, kTraceBucketHiddenDimensions,
              deep_path.fc1_output.begin());
  deep_path.fc2_preact[0] = lca_buffer.fc_2_out[0];

  auto& final_path = snapshot->final_path;
  final_path.selected_bucket = deep_path.selected_bucket;
  final_path.material_bucket = snapshot->pair_bucket;
  final_path.deep_output[0] = lca_buffer.fc_2_out[0];
  final_path.bypass_input[0] = lca_buffer.fc_0_out[31];
  final_path.bypass_preact[0] = lca_buffer.fc_0_out[31];
  final_path.bypass_scaled_numerator[0] =
      static_cast<std::int32_t>(lca_buffer.fc_0_out[31] * (600 * 16));
  final_path.bypass_output[0] =
      final_path.bypass_scaled_numerator[0] / (127 * 64);
  final_path.alpha_q14[0] = selected_network->bucket_blend_alpha;
  final_path.inv_alpha_q14[0] = 16384 - final_path.alpha_q14[0];
  final_path.deep_term[0] =
      static_cast<std::int64_t>(final_path.deep_output[0])
      * final_path.alpha_q14[0];
  final_path.bypass_term[0] =
      static_cast<std::int64_t>(final_path.bypass_output[0])
      * final_path.inv_alpha_q14[0];
  final_path.blend_numerator[0] =
      final_path.deep_term[0] + final_path.bypass_term[0];
  final_path.blend_output[0] = static_cast<std::int32_t>(
      final_path.blend_numerator[0] / 16384);
  final_path.network_output[0] = final_path.blend_output[0];
  final_path.fv_scale[0] = FV_SCALE;
  final_path.eval_before_clamp[0] =
      final_path.network_output[0] / final_path.fv_scale[0];
  final_path.value_max_eval[0] = VALUE_MAX_EVAL;
  final_path.eval_after_clamp[0] = Math::clamp(
      final_path.eval_before_clamp[0], -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

  feature_transformer->TracePairWeights(
      snapshot->pair_bucket, snapshot->pair_weight_mul.data(),
      snapshot->pair_weight_diff.data(), snapshot->pair_weight_sum.data());

  const auto& accumulator = trace_position.state()->accumulator;
  for (const Color perspective : {BLACK, WHITE}) {
    auto& output = snapshot->perspective[perspective];
    std::copy_n(accumulator.accumulation[perspective][0],
                kTransformedFeatureDimensions,
                output.main_accumulator.begin());

    feature_transformer->TraceMainPair(
        trace_position, perspective, snapshot->pair_bucket,
        output.main_pair.a.data(), output.main_pair.b.data(),
        output.main_pair.mul_term.data(),
        output.main_pair.diff_sq_term.data(),
        output.main_pair.sum_term.data(),
        output.main_pair.mixed_numerator.data());

    const std::size_t transformed_offset =
        perspective == snapshot->side_to_move ? 0 : kTracePairDimensions;
    std::copy_n(transformed_output.begin() + transformed_offset,
                kTracePairDimensions, output.main_pair.output.begin());

    const auto& factors = accumulator.factors[perspective];
    std::copy_n(factors.halfka.sum_v, kTraceFmDimensions,
                output.fm_accumulator.halfka_sum_v.begin());
    std::copy_n(factors.halfka.sum_v2, kTraceFmDimensions,
                output.fm_accumulator.halfka_sum_v2.begin());
    std::copy_n(factors.ksdg.sum_v, kTraceFmDimensions,
                output.fm_accumulator.ksdg_sum_v.begin());
    std::copy_n(factors.ksdg.sum_v2, kTraceFmDimensions,
                output.fm_accumulator.ksdg_sum_v2.begin());

    for (std::size_t index = 0; index < kTraceFmDimensions; ++index) {
      const auto interaction = [](const std::int64_t sum_v,
                                  const std::int64_t sum_v2) {
        return (sum_v * sum_v - sum_v2) / 2;
      };
      output.fm_interaction.ih[index] =
          interaction(factors.halfka.sum_v[index],
                      factors.halfka.sum_v2[index]);
      output.fm_interaction.ik[index] =
          interaction(factors.ksdg.sum_v[index], factors.ksdg.sum_v2[index]);
      output.fm_interaction.sh[index] = factors.halfka.sum_v[index];
      output.fm_interaction.sk[index] = factors.ksdg.sum_v[index];
    }
  }

  const auto& us = snapshot->perspective[snapshot->side_to_move].fm_interaction;
  const auto& them = snapshot->perspective[~snapshot->side_to_move].fm_interaction;
  for (std::size_t index = 0; index < kTraceFmDimensions; ++index) {
    snapshot->raw_diff.ih[index] = us.ih[index] - them.ih[index];
    snapshot->raw_diff.ik[index] = us.ik[index] - them.ik[index];
    snapshot->raw_diff.sh[index] = us.sh[index] - them.sh[index];
    snapshot->raw_diff.sk[index] = us.sk[index] - them.sk[index];

    // Current C++ Abs path is the side-to-move perspective only.
    snapshot->raw_abs.ih[index] = us.ih[index];
    snapshot->raw_abs.ik[index] = us.ik[index];
    snapshot->raw_abs.sh[index] = us.sh[index];
    snapshot->raw_abs.sk[index] = us.sk[index];
  }

  return true;
}

template <typename Container>
void WriteTraceArray(std::ostream& output, const std::string& name,
                     const char* const type_name, const Container& values) {
  output << name << '\t' << type_name << '\t' << values.size();
  for (const auto value : values)
    output << '\t' << static_cast<std::int64_t>(value);
  output << '\n';
}

void WriteTraceString(std::ostream& output, const char* const name,
                      const std::string& value) {
  output << name << "\tstr\t1\t" << value << '\n';
}

void WriteTraceScalar(std::ostream& output, const std::string& name,
                      const char* const type_name, const std::uint64_t value) {
  output << name << '\t' << type_name << "\t1\t" << value << '\n';
}

void WriteTraceHash(std::ostream& output, const std::string& name,
                    const std::uint64_t value) {
  const auto flags = output.flags();
  const auto fill = output.fill();
  output << name << "\tu64_hex\t1\t0x" << std::hex << std::setw(16)
         << std::setfill('0') << value << '\n';
  output.flags(flags);
  output.fill(fill);
}

void WriteFmAccumulator(std::ostream& output, const std::string& prefix,
                        const TraceFmAccumulator& values) {
  WriteTraceArray(output, prefix + ".halfka.sum_v", "i64", values.halfka_sum_v);
  WriteTraceArray(output, prefix + ".halfka.sum_v2", "i64", values.halfka_sum_v2);
  WriteTraceArray(output, prefix + ".ksdg.sum_v", "i64", values.ksdg_sum_v);
  WriteTraceArray(output, prefix + ".ksdg.sum_v2", "i64", values.ksdg_sum_v2);
}

void WriteFmInteraction(std::ostream& output, const std::string& prefix,
                        const TraceFmInteraction& values) {
  WriteTraceArray(output, prefix + ".ih", "i64", values.ih);
  WriteTraceArray(output, prefix + ".ik", "i64", values.ik);
  WriteTraceArray(output, prefix + ".sh", "i64", values.sh);
  WriteTraceArray(output, prefix + ".sk", "i64", values.sk);
}

void WriteMainPair(std::ostream& output, const std::string& prefix,
                   const TraceMainPair& values) {
  WriteTraceArray(output, prefix + ".a", "i32", values.a);
  WriteTraceArray(output, prefix + ".b", "i32", values.b);
  WriteTraceArray(output, prefix + ".mul_term", "i32", values.mul_term);
  WriteTraceArray(output, prefix + ".diff_sq_term", "i32",
                  values.diff_sq_term);
  WriteTraceArray(output, prefix + ".sum_term", "i32", values.sum_term);
  WriteTraceArray(output, prefix + ".mixed_numerator", "i32",
                  values.mixed_numerator);
  WriteTraceArray(output, prefix + ".output", "u8", values.output);
}

bool WriteNnueTrace(const NnueTraceSnapshot& snapshot,
                    const std::string& output_file,
                    std::string* const error_message) {
  std::ofstream output(output_file, std::ios::binary);
  if (!output) {
    *error_message = "failed to open trace output file: " + output_file;
    return false;
  }

  output << "NNUE_TRACE_V1\n";
  WriteTraceString(output, "meta.sfen", snapshot.sfen);
  WriteTraceString(output, "meta.side_to_move",
                   TracePerspectiveName(snapshot.side_to_move));
  WriteTraceString(output, "meta.us_perspective",
                   TracePerspectiveName(snapshot.side_to_move));
  WriteTraceString(output, "meta.them_perspective",
                   TracePerspectiveName(~snapshot.side_to_move));
  WriteTraceString(output, "meta.pytorch.white_indices_perspective", "BLACK");
  WriteTraceString(output, "meta.pytorch.black_indices_perspective", "WHITE");
  WriteTraceString(output, "meta.pytorch.t_w_v_w_perspective", "BLACK");
  WriteTraceString(output, "meta.pytorch.t_b_v_b_perspective", "WHITE");
  WriteTraceString(output, "meta.raw_abs_source", "us_perspective_only");
  WriteTraceString(output, "meta.feature_index_order", "sorted_ascending");
  WriteTraceString(output, "meta.feature_hash_encoding",
                   "fnv1a64_sorted_u32_little_endian");
  WriteTraceString(output, "meta.scaled_layout",
                   "ih[0:32],ik[32:64],sh[64:96],sk[96:128]");
  WriteTraceScalar(output, "meta.refresh_trigger_count", "u64",
                   kRefreshTriggers.size());
  WriteTraceScalar(output, "meta.main_accumulator_trigger_index", "u64", 0);
  WriteTraceScalar(output, "meta.main_accumulator_trigger_event", "u64",
                   static_cast<std::uint64_t>(kRefreshTriggers[0]));
  WriteTraceScalar(output, "pair.bucket_id", "u64", snapshot.pair_bucket);
  WriteTraceString(output, "pair.bucket_source",
                   "stack_index_for_nnue_material_value");
  WriteTraceString(output, "pair.main.output_order", "us_then_them");
  WriteTraceString(output, "router.input_layout",
                   "abs_centered[0:128],diff[128:256],main_us[256:384]");
  WriteTraceArray(output, "pair.weight.mul", "i16",
                  snapshot.pair_weight_mul);
  WriteTraceArray(output, "pair.weight.diff", "i16",
                  snapshot.pair_weight_diff);
  WriteTraceArray(output, "pair.weight.sum", "i16",
                  snapshot.pair_weight_sum);

  for (const Color perspective : {BLACK, WHITE}) {
    const std::string perspective_name = TracePerspectiveName(perspective);
    const auto& values = snapshot.perspective[perspective];
    const std::string feature_prefix = "feature." + perspective_name;
    const auto& indices = values.active_indices;
    std::uint64_t index_sum = 0;
    for (const IndexType index : indices)
      index_sum += static_cast<std::uint64_t>(index);

    WriteTraceScalar(output, feature_prefix + ".count", "u64", indices.size());
    WriteTraceScalar(output, feature_prefix + ".sum", "u64", index_sum);
    WriteTraceScalar(output, feature_prefix + ".min", "u32",
                     indices.empty() ? 0 : indices.front());
    WriteTraceScalar(output, feature_prefix + ".max", "u32",
                     indices.empty() ? 0 : indices.back());
    WriteTraceHash(output, feature_prefix + ".fnv1a64",
                   Fnv1a64Indices(indices));
    WriteTraceArray(output, feature_prefix + ".indices", "u32", indices);

    WriteTraceArray(output, "ft.main." + perspective_name, "i16",
                    values.main_accumulator);
    WriteFmAccumulator(output, "fm.accumulator." + perspective_name,
                       values.fm_accumulator);
    WriteFmInteraction(output, "fm.interaction." + perspective_name,
                       values.fm_interaction);
    WriteMainPair(output, "pair.main." + perspective_name,
                  values.main_pair);
  }

  WriteFmInteraction(output, "fm.raw_diff", snapshot.raw_diff);
  WriteFmInteraction(output, "fm.raw_abs", snapshot.raw_abs);
  WriteTraceArray(output, "fm.scaled_diff", "u8", snapshot.scaled_diff);
  WriteTraceArray(output, "fm.scaled_abs", "u8", snapshot.scaled_abs);
  WriteTraceArray(output, "router.input", "u8", snapshot.router.input);
  WriteTraceArray(output, "router.logits", "i32", snapshot.router.logits);
  WriteTraceScalar(output, "router.selected_bucket", "u64",
                   snapshot.router.selected_bucket);
  WriteTraceString(output, "fm.path.scope", "direct_fc_gate_before_lca");
  WriteTraceString(output, "fm.path.diff_gate_target", "main_path");
  WriteTraceScalar(output, "fm.path.selected_bucket", "u64",
                   snapshot.fm_path.selected_bucket);
  WriteTraceArray(output, "fm.path.diff.input", "u8",
                  snapshot.fm_path.diff_input);
  WriteTraceArray(output, "fm.path.abs.input", "u8",
                  snapshot.fm_path.abs_input);
  WriteTraceArray(output, "fm.path.diff.fc_preact", "i32",
                  snapshot.fm_path.diff_fc_preact);
  WriteTraceArray(output, "fm.path.abs.fc_preact", "i32",
                  snapshot.fm_path.abs_fc_preact);
  WriteTraceArray(output, "fm.path.diff.gate_preact", "i32",
                  snapshot.fm_path.diff_gate_preact);
  WriteTraceArray(output, "fm.path.diff.value_preact", "i32",
                  snapshot.fm_path.diff_value_preact);
  WriteTraceScalar(output, "fm.path.diff.rms_sum_sq_f32_bits", "u32",
                   snapshot.fm_path.diff_rms_sum_sq_f32_bits);
  WriteTraceScalar(output, "fm.path.diff.inv_rms_f32_bits", "u32",
                   snapshot.fm_path.diff_inv_rms_f32_bits);
  WriteTraceArray(output, "fm.path.diff.normalized_f32_bits", "u32",
                  snapshot.fm_path.diff_normalized_f32_bits);
  WriteTraceArray(output, "fm.path.diff.normalized_scaled_centered", "i32",
                  snapshot.fm_path.diff_normalized_scaled_centered);
  WriteTraceArray(output, "fm.path.diff.output_pre_lca", "u8",
                  snapshot.fm_path.diff_output_pre_lca);
  WriteTraceArray(output, "fm.path.diff.main_gate_q64", "i32",
                  snapshot.fm_path.diff_main_gate_q64);
  WriteTraceArray(output, "fm.path.diff.main_gate_multiplier_q128", "i32",
                  snapshot.fm_path.diff_main_gate_multiplier_q128);
  WriteTraceArray(output, "fm.path.abs.gate_preact", "i32",
                  snapshot.fm_path.abs_gate_preact);
  WriteTraceArray(output, "fm.path.abs.value_preact", "i32",
                  snapshot.fm_path.abs_value_preact);
  WriteTraceArray(output, "fm.path.abs.gate_sigmoid_f32_bits", "u32",
                  snapshot.fm_path.abs_gate_sigmoid_f32_bits);
  WriteTraceArray(output, "fm.path.abs.gated_value", "i32",
                  snapshot.fm_path.abs_gated_value);
  WriteTraceArray(output, "fm.path.abs.scaled_before_round_f32_bits",
                  "u32",
                  snapshot.fm_path.abs_scaled_before_round_f32_bits);
  WriteTraceArray(output, "fm.path.abs.output", "u8",
                  snapshot.fm_path.abs_output);
  WriteTraceArray(output, "fm.path.abs.squared_output", "u8",
                  snapshot.fm_path.abs_squared_output);
  WriteTraceString(output, "lca.scope", "selected_bucket_before_cross");
  WriteTraceScalar(output, "lca.selected_bucket", "u64",
                   snapshot.lca.selected_bucket);
  WriteTraceArray(output, "lca.main_fc_preact_before_gate", "i32",
                  snapshot.lca.main_fc_preact_before_gate);
  WriteTraceArray(output, "lca.main_fc_preact_after_gate", "i32",
                  snapshot.lca.main_fc_preact_after_gate);
  WriteTraceArray(output, "lca.query_input", "u8",
                  snapshot.lca.query_input);
  WriteTraceArray(output, "lca.fm_input", "u8", snapshot.lca.fm_input);
  WriteTraceArray(output, "lca.query_preact", "i32",
                  snapshot.lca.query_preact);
  WriteTraceArray(output, "lca.key_preact", "i32",
                  snapshot.lca.key_preact);
  WriteTraceArray(output, "lca.value_preact", "i32",
                  snapshot.lca.value_preact);
  WriteTraceScalar(output, "lca.temperature_f32_bits", "u32",
                   snapshot.lca.temperature_f32_bits);
  WriteTraceScalar(output, "lca.dot_product_f32_bits", "u32",
                   snapshot.lca.dot_product_f32_bits);
  WriteTraceScalar(output, "lca.attention_logit_f32_bits", "u32",
                   snapshot.lca.attention_logit_f32_bits);
  WriteTraceScalar(output, "lca.attention_score_f32_bits", "u32",
                   snapshot.lca.attention_score_f32_bits);
  WriteTraceArray(output, "lca.value_clamped_f32_bits", "u32",
                  snapshot.lca.value_clamped_f32_bits);
  WriteTraceArray(output, "lca.correction_f32_bits", "u32",
                  snapshot.lca.correction_f32_bits);
  WriteTraceArray(output, "lca.output_post_lca_f32_bits", "u32",
                  snapshot.lca.output_post_lca_f32_bits);
  WriteTraceArray(output, "lca.output_post_lca", "u8",
                  snapshot.lca.output_post_lca);
  WriteTraceString(output, "deep.scope", "cross_through_fc2_preblend");
  WriteTraceString(output, "deep.phase.bucket_source", "material_pair_bucket");
  WriteTraceScalar(output, "deep.phase.bucket_id", "u64",
                   snapshot.pair_bucket);
  WriteTraceScalar(output, "deep.scale.activation", "u64", 127);
  WriteTraceScalar(output, "deep.scale.hidden_preact", "u64", 8128);
  WriteTraceScalar(output, "deep.scale.fc2_preact", "u64", 9600);
  WriteTraceScalar(output, "deep.selected_bucket", "u64",
                   snapshot.deep_path.selected_bucket);
  WriteTraceArray(output, "deep.phase.input", "u8",
                  snapshot.deep_path.phase_input);
  WriteTraceArray(output, "deep.phase.preact", "i32",
                  snapshot.deep_path.phase_preact);
  WriteTraceArray(output, "deep.phase.logit_f32_bits", "u32",
                  snapshot.deep_path.phase_logit_f32_bits);
  WriteTraceArray(output, "deep.phase.sigmoid_f32_bits", "u32",
                  snapshot.deep_path.phase_sigmoid_f32_bits);
  WriteTraceArray(output, "deep.phase.value_f32_bits", "u32",
                  snapshot.deep_path.phase_value_f32_bits);
  WriteTraceArray(output, "deep.phase.channel_scale_f32_bits", "u32",
                  snapshot.deep_path.channel_scale_f32_bits);
  WriteTraceArray(output, "deep.main.raw", "u8",
                  snapshot.deep_path.main_raw);
  WriteTraceArray(output, "deep.main.squared", "u8",
                  snapshot.deep_path.main_squared);
  WriteTraceArray(output, "deep.cross.main_squared", "u8",
                  snapshot.deep_path.cross_main_squared);
  WriteTraceArray(output, "deep.cross.diff", "u8",
                  snapshot.deep_path.cross_diff);
  WriteTraceArray(output, "deep.cross.main_raw", "u8",
                  snapshot.deep_path.cross_main_raw);
  WriteTraceArray(output, "deep.cross.abs", "u8",
                  snapshot.deep_path.cross_abs);
  WriteTraceArray(output, "deep.cross.product_diff", "u8",
                  snapshot.deep_path.cross_product_diff);
  WriteTraceArray(output, "deep.cross.product_abs", "u8",
                  snapshot.deep_path.cross_product_abs);
  WriteTraceArray(output, "deep.cross.input", "u8",
                  snapshot.deep_path.cross_input);
  WriteTraceArray(output, "deep.cross.preact", "i32",
                  snapshot.deep_path.cross_preact);
  WriteTraceArray(output, "deep.cross.output", "u8",
                  snapshot.deep_path.cross_output);
  WriteTraceArray(output, "deep.fc1.input", "u8",
                  snapshot.deep_path.fc1_input);
  WriteTraceArray(output, "deep.fc1.preact", "i32",
                  snapshot.deep_path.fc1_preact);
  WriteTraceArray(output, "deep.fc1.output", "u8",
                  snapshot.deep_path.fc1_output);
  WriteTraceArray(output, "deep.fc2.preact", "i32",
                  snapshot.deep_path.fc2_preact);
  WriteTraceArray(output, "deep.fc2.output_preblend", "i32",
                  snapshot.deep_path.fc2_preact);
  WriteTraceString(output, "final.scope", "fc2_through_evaluate");
  WriteTraceString(output, "final.score_perspective", "side_to_move");
  WriteTraceString(output, "final.side_adjustment", "none_already_side_to_move");
  WriteTraceString(output, "final.tempo", "none");
  WriteTraceScalar(output, "final.selected_bucket", "u64",
                   snapshot.final_path.selected_bucket);
  WriteTraceScalar(output, "final.material_bucket", "u64",
                   snapshot.final_path.material_bucket);
  WriteTraceScalar(output, "final.scale.deep_output", "u64", 9600);
  WriteTraceScalar(output, "final.scale.bypass_input", "u64", 8128);
  WriteTraceScalar(output, "final.scale.bypass_output", "u64", 9600);
  WriteTraceScalar(output, "final.scale.blend_alpha", "u64", 16384);
  WriteTraceScalar(output, "final.scale.blend_numerator", "u64",
                   UINT64_C(9600) * UINT64_C(16384));
  WriteTraceScalar(output, "final.scale.network_output", "u64", 9600);
  WriteTraceScalar(output, "final.scale.eval_value", "u64", 1);
  WriteTraceScalar(output, "final.scale.pytorch_nnue2score", "u64", 600);
  WriteTraceArray(output, "final.deep_output", "i32",
                  snapshot.final_path.deep_output);
  WriteTraceArray(output, "final.bypass.input", "i32",
                  snapshot.final_path.bypass_input);
  WriteTraceArray(output, "final.bypass.preact", "i32",
                  snapshot.final_path.bypass_preact);
  WriteTraceArray(output, "final.bypass.scaled_numerator", "i32",
                  snapshot.final_path.bypass_scaled_numerator);
  WriteTraceArray(output, "final.bypass.output", "i32",
                  snapshot.final_path.bypass_output);
  WriteTraceArray(output, "final.blend.parameter_raw", "i32",
                  snapshot.final_path.alpha_q14);
  WriteTraceArray(output, "final.blend.alpha_q14", "i32",
                  snapshot.final_path.alpha_q14);
  WriteTraceArray(output, "final.blend.inv_alpha_q14", "i32",
                  snapshot.final_path.inv_alpha_q14);
  WriteTraceArray(output, "final.blend.deep_term", "i64",
                  snapshot.final_path.deep_term);
  WriteTraceArray(output, "final.blend.bypass_term", "i64",
                  snapshot.final_path.bypass_term);
  WriteTraceArray(output, "final.blend.numerator", "i64",
                  snapshot.final_path.blend_numerator);
  WriteTraceArray(output, "final.blend.output", "i32",
                  snapshot.final_path.blend_output);
  WriteTraceArray(output, "final.network_output", "i32",
                  snapshot.final_path.network_output);
  WriteTraceArray(output, "final.fv_scale", "i32",
                  snapshot.final_path.fv_scale);
  WriteTraceArray(output, "final.eval_before_clamp", "i32",
                  snapshot.final_path.eval_before_clamp);
  WriteTraceArray(output, "final.value_max_eval", "i32",
                  snapshot.final_path.value_max_eval);
  WriteTraceArray(output, "final.eval_after_clamp", "i32",
                  snapshot.final_path.eval_after_clamp);

  if (!output) {
    *error_message = "failed while writing trace output file: " + output_file;
    return false;
  }
  return true;
}

void PrintNnueTraceSummary(const NnueTraceSnapshot& snapshot) {
  std::cout << "NNUE trace" << std::endl
            << "  SFEN              : " << snapshot.sfen << std::endl
            << "  side to move / us : "
            << TracePerspectiveName(snapshot.side_to_move) << std::endl
            << "  them              : "
            << TracePerspectiveName(~snapshot.side_to_move) << std::endl
            << "  pair bucket       : " << snapshot.pair_bucket << std::endl
            << "  router bucket     : " << snapshot.router.selected_bucket
            << std::endl
            << "  FM path bucket    : " << snapshot.fm_path.selected_bucket
            << std::endl
            << "  PyTorch white_indices / t_w / v_w = C++ BLACK" << std::endl
            << "  PyTorch black_indices / t_b / v_b = C++ WHITE" << std::endl;

  for (const Color perspective : {BLACK, WHITE}) {
    const auto& indices = snapshot.perspective[perspective].active_indices;
    std::uint64_t index_sum = 0;
    for (const IndexType index : indices)
      index_sum += static_cast<std::uint64_t>(index);

    const auto flags = std::cout.flags();
    const auto fill = std::cout.fill();
    std::cout << "  " << TracePerspectiveName(perspective)
              << " active features" << std::endl
              << "    count : " << indices.size() << std::endl
              << "    sum   : " << index_sum << std::endl
              << "    min   : " << (indices.empty() ? 0 : indices.front())
              << std::endl
              << "    max   : " << (indices.empty() ? 0 : indices.back())
              << std::endl
              << "    FNV-1a: 0x" << std::hex << std::setw(16)
              << std::setfill('0') << Fnv1a64Indices(indices) << std::endl;
    std::cout.flags(flags);
    std::cout.fill(fill);
  }
}

void TraceNnue(std::istream& stream, const bool write_full_trace) {
  std::string output_file;
  if (write_full_trace)
    stream >> std::quoted(output_file);

  std::string sfen;
  std::getline(stream >> std::ws, sfen);
  if ((write_full_trace && output_file.empty()) || sfen.empty()) {
    std::cout << "error: "
              << (write_full_trace
                      ? "usage: test nnue trace_full <output file> <SFEN>"
                      : "usage: test nnue trace <SFEN>")
              << std::endl;
    return;
  }

  NnueTraceSnapshot snapshot;
  std::string error_message;
  if (!MakeNnueTraceSnapshot(sfen, &snapshot, &error_message)) {
    std::cout << "error: " << error_message << std::endl;
    return;
  }

  PrintNnueTraceSummary(snapshot);
  if (write_full_trace) {
    if (!WriteNnueTrace(snapshot, output_file, &error_message)) {
      std::cout << "error: " << error_message << std::endl;
      return;
    }
    std::cout << "  full trace written: " << output_file << std::endl;
  }
}

#endif  // defined(ENABLE_NNUE_TRACE)

// 評価関数の構造を表す文字列を出力する
void PrintInfo(std::istream& stream) {
  std::cout << "network architecture: " << GetArchitectureString() << std::endl;

  while (true) {
    std::string file_name;
    stream >> file_name;
    if (file_name.empty()) break;

    std::uint32_t hash_value;
    std::string architecture;
    const Tools::Result result = [&]() {
      std::ifstream file_stream(file_name, std::ios::binary);
      if (!file_stream) return Tools::Result(Tools::ResultCode::FileReadError);
	  return ReadHeader(file_stream, &hash_value, &architecture);
    }();

    std::cout << file_name << ": ";
    if (result.is_ok()) {
      if (hash_value == kHashValue) {
        std::cout << "matches with this binary";
        if (architecture != GetArchitectureString()) {
          std::cout << ", but architecture string differs: " << architecture;
        }
        std::cout << std::endl;
      } else {
        std::cout << architecture << std::endl;
      }
    } else {
      std::cout << "failed to read header" << std::endl;
    }
  }
}

}  // namespace

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(IEngine& engine, std::istream& stream) {
  std::string sub_command;
  stream >> sub_command;

  auto& pos = engine.get_position();

  if (sub_command == "test_features") {
    TestFeatures(pos);
  } else if (sub_command == "test_accumulator") {
    TestAccumulator(pos);
  } else if (sub_command == "info") {
    PrintInfo(stream);
  } else if (sub_command == "accuracy") {
    TestMoveAccuracy(engine, stream);
#if defined(ENABLE_NNUE_BENCH)
  } else if (sub_command == "bench_ft") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFeatureTransformerBenchmark(repeat_count);
  } else if (sub_command == "bench_network") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestNetworkBenchmark(repeat_count);
  } else if (sub_command == "bench_network_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestNetworkBenchmarkCompare(repeat_count);
  } else if (sub_command == "bench_network_stages") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestNetworkStagesBenchmark(repeat_count);
#if defined(USE_AVX2)
  } else if (sub_command == "bench_fm_abs_squared_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFmAbsSquaredBenchmarkCompare(repeat_count);
#endif
  } else if (sub_command == "bench_sqr_clipped_relu_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestSqrClippedReluBenchmarkCompare(repeat_count);
#if defined(USE_AVX2) && !defined(USE_AVX512)
  } else if (sub_command == "bench_fc0_sparse_dense") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFc0SparseDenseBenchmark(repeat_count);
  } else if (sub_command == "bench_fc0_accumulators") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFc0AccumulatorBenchmark(repeat_count);
  } else if (sub_command == "bench_fc1_output_tiles") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFc1OutputTilingBenchmark(repeat_count);
#endif
#endif
#if defined(ENABLE_NNUE_TRACE)
  } else if (sub_command == "trace") {
    TraceNnue(stream, false);
  } else if (sub_command == "trace_full") {
    TraceNnue(stream, true);
#endif
  } else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nnue test_features" << std::endl;
    std::cout << " test nnue test_accumulator" << std::endl;
    std::cout << " test nnue accuracy <sfenpack file>" << std::endl;
    std::cout << " test nnue info [path/to/" << kFileName << "...]" << std::endl;
#if defined(ENABLE_NNUE_BENCH)
    std::cout << " test nnue bench_ft [repeats]" << std::endl;
    std::cout << " test nnue bench_network [repeats]" << std::endl;
    std::cout << " test nnue bench_network_compare [repeats]" << std::endl;
    std::cout << " test nnue bench_network_stages [repeats]" << std::endl;
#if defined(USE_AVX2)
    std::cout << " test nnue bench_fm_abs_squared_compare [repeats]"
              << std::endl;
#endif
    std::cout << " test nnue bench_sqr_clipped_relu_compare [repeats]"
              << std::endl;
#if defined(USE_AVX2) && !defined(USE_AVX512)
    std::cout << " test nnue bench_fc0_sparse_dense [repeats]"
              << std::endl;
    std::cout << " test nnue bench_fc0_accumulators [repeats]"
              << std::endl;
    std::cout << " test nnue bench_fc1_output_tiles [repeats]"
              << std::endl;
#endif
#endif
#if defined(ENABLE_NNUE_TRACE)
    std::cout << " test nnue trace <SFEN>" << std::endl;
    std::cout << " test nnue trace_full <output file> <SFEN>" << std::endl;
#endif
  }
}

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)
