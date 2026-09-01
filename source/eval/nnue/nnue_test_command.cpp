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
  } else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nnue test_features" << std::endl;
    std::cout << " test nnue test_accumulator" << std::endl;
    std::cout << " test nnue accuracy <sfenpack file>" << std::endl;
    std::cout << " test nnue info [path/to/" << kFileName << "...]" << std::endl;
  }
}

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)
