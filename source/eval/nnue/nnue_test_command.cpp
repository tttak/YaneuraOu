// NNUE評価関数に関するUSI拡張コマンド

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../engine.h"
#include "../../extra/all.h"
#include "../../evaluate.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"
#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
#define NNUE_SIDE_INPUT_KING_SQUARE(pos, color) (pos).square<KING>(color)
#define NNUE_SIDE_INPUT_NAMESPACE_BEGIN namespace YaneuraOu {
#define NNUE_SIDE_INPUT_NAMESPACE_END }
#include "nnue_side_input.h"
#undef NNUE_SIDE_INPUT_NAMESPACE_END
#undef NNUE_SIDE_INPUT_NAMESPACE_BEGIN
#undef NNUE_SIDE_INPUT_KING_SQUARE
#endif
#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
#include "nnue_pair_relation.h"
#endif
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
#include "qsearch_correction_shadow.h"
#endif

#if defined(ENABLE_NNUE_SIGNAL_LOG)
#include "../../engine/yaneuraou-engine/nnue_signal_logger.h"
#endif
#if defined(ENABLE_NNUE_POLICY_SHADOW)
#include "../../engine/yaneuraou-engine/nnue_policy_shadow.h"
#endif
#if defined(ENABLE_NNUE_DECISION_RISK_LMR_COUNTERS)
#include "../../engine/yaneuraou-engine/nnue_decision_risk_lmr_counters.h"
#endif
#if defined(ENABLE_NNUE_ASPIRATION_DIAGNOSTIC)
#include "../../engine/yaneuraou-engine/nnue_aspiration_logger.h"
#endif
#if defined(ENABLE_NNUE_ADAPTIVE_ASPIRATION_COUNTERS)
#include "../../engine/yaneuraou-engine/adaptive_aspiration_counters.h"
#endif
#if defined(ENABLE_NNUE_TT_REUSE_DIAGNOSTIC)
#include "../../engine/yaneuraou-engine/nnue_tt_reuse_logger.h"
#endif
#if defined(ENABLE_ROOT_MOVE_HISTORY_DIAGNOSTIC)
#include "../../engine/yaneuraou-engine/root_move_history_logger.h"
#endif
#if defined(ENABLE_ROOT_TIME_RISK_BUDGET)
#include "../../engine/yaneuraou-engine/root_time_risk_budget.h"
#endif
#if defined(ENABLE_QSEARCH_CORRECTION_PROBE)
#include "../../engine/yaneuraou-engine/yaneuraou-search.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>
#include <vector>

#if defined(ENABLE_NNUE_TRACE) || defined(ENABLE_NNUE_BENCH)
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <system_error>
#endif

#if defined(ENABLE_NNUE_BENCH)
#include "features/king_safety3_distinguishgolds.h"
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

#if defined(ENABLE_QSEARCH_CORRECTION_PROBE)
#include "qsearch_correction_tool.inc"
#include "qsearch_correction_decomposition_tool.inc"
#endif

#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
static_assert(sizeof(MoveAccuracyRecord) == 40,
              "sfenpack record must be exactly 40 bytes");
static_assert(offsetof(MoveAccuracyRecord, score) == 32,
              "PackedSfenValue score must start at byte 32");
static_assert(sizeof(s16) == 2,
              "PackedSfenValue score must be a signed 16-bit value");

void WriteCsvField(std::ostream& output, std::string_view value);

// Replaces only PackedSfenValue::score with a fresh, no-search NNUE
// evaluation.  Input bytes are copied verbatim and the two score bytes are
// overwritten explicitly, so move/gamePly/result/padding cannot be rewritten
// by structure assignment or padding initialization.
void MakeStaticEvalBin(std::istream& stream) {
  std::string input_name;
  std::string output_name;
  std::string csv_name;
  std::uint64_t start_record = 0;
  std::uint64_t max_records = 0;
  std::uint64_t csv_records = 0;
  std::string metadata_name;
  stream >> std::quoted(input_name) >> std::quoted(output_name)
         >> start_record >> max_records >> std::quoted(csv_name) >> csv_records;
  if (!(stream >> std::quoted(metadata_name)))
    stream.clear();
  if (input_name.empty() || output_name.empty()) {
    std::cout << "error: make_static_eval_bin requires input and output paths"
              << std::endl;
    return;
  }

  std::ifstream input(input_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open input: " << input_name << std::endl;
    return;
  }
  input.seekg(0, std::ios::end);
  const auto end_position = input.tellg();
  if (end_position < 0
      || static_cast<std::uint64_t>(end_position) % sizeof(MoveAccuracyRecord) != 0) {
    std::cout << "error: input size is not a multiple of 40 bytes" << std::endl;
    return;
  }
  const std::uint64_t total_records =
      static_cast<std::uint64_t>(end_position) / sizeof(MoveAccuracyRecord);
  if (start_record > total_records) {
    std::cout << "error: start record exceeds input record count" << std::endl;
    return;
  }
  const std::uint64_t available = total_records - start_record;
  const std::uint64_t requested =
      max_records == 0 ? available : std::min(max_records, available);
  input.seekg(static_cast<std::streamoff>(
                  start_record * sizeof(MoveAccuracyRecord)),
              std::ios::beg);

  // Never replace an existing result.  The temporary file is renamed only
  // after every record has been decoded, validated, evaluated and flushed.
  {
    std::ifstream existing(output_name, std::ios::binary);
    if (existing) {
      std::cout << "error: output already exists: " << output_name << std::endl;
      return;
    }
  }
  const std::string temporary_name = output_name + ".tmp";
  {
    std::ifstream existing(temporary_name, std::ios::binary);
    if (existing) {
      std::cout << "error: temporary output already exists: "
                << temporary_name << std::endl;
      return;
    }
  }
  std::ofstream output(temporary_name, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cout << "error: failed to create temporary output: "
              << temporary_name << std::endl;
    return;
  }

  const std::string csv_temporary_name =
      csv_name.empty() ? std::string() : csv_name + ".tmp";
  std::ofstream csv;
  if (!csv_name.empty()) {
    std::ifstream existing(csv_name);
    std::ifstream temporary_existing(csv_temporary_name);
    if (existing || temporary_existing) {
      std::cout << "error: validation CSV or its temporary file already exists"
                << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
    csv.open(csv_temporary_name, std::ios::out | std::ios::trunc);
    if (!csv) {
      std::cout << "error: failed to create validation CSV" << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
    csv << "record_index,sfen,side_to_move,original_depth9_score,"
           "static_eval_score,depth9_minus_static,game_ply,game_result,move\n";
  }

  struct StaticEvalMetadata {
    std::int32_t material_black;
    std::int32_t material_stm;
    std::uint16_t game_ply;
    std::uint8_t selected_bucket;
    std::uint8_t side_to_move;
  };
  static_assert(sizeof(StaticEvalMetadata) == 12, "unexpected metadata layout");
  const std::string metadata_temporary_name =
      metadata_name.empty() ? std::string() : metadata_name + ".tmp";
  std::ofstream metadata;
  if (!metadata_name.empty()) {
    std::ifstream existing(metadata_name, std::ios::binary);
    std::ifstream temporary_existing(metadata_temporary_name, std::ios::binary);
    if (existing || temporary_existing) {
      std::cout << "error: metadata output or temporary already exists" << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
    metadata.open(metadata_temporary_name, std::ios::binary | std::ios::trunc);
    if (!metadata) {
      std::cout << "error: failed to create metadata output" << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
  }

  std::uint64_t processed = 0;
  std::uint64_t decode_errors = 0;
  std::uint64_t illegal_positions = 0;
  std::uint64_t non_score_byte_mismatches = 0;
  bool failed = false;
  std::string failure;
  const auto started = std::chrono::steady_clock::now();
  auto last_report = started;
  std::array<char, sizeof(MoveAccuracyRecord)> original_bytes{};
  std::array<char, sizeof(MoveAccuracyRecord)> output_bytes{};

  while (processed < requested
         && input.read(original_bytes.data(), original_bytes.size())) {
    MoveAccuracyRecord record;
    std::memcpy(&record, original_bytes.data(), sizeof(record));

    Position position;
    StateInfo state;
    if (position.set_from_packed_sfen(
            record.sfen, &state, false, record.game_ply).is_not_ok()) {
      ++decode_errors;
      failed = true;
      failure = "PackedSfen decode failed at record "
              + std::to_string(start_record + processed);
      break;
    }
    if (!position.pos_is_ok()) {
      ++illegal_positions;
      failed = true;
      failure = "illegal/inconsistent position at record "
              + std::to_string(start_record + processed);
      break;
    }

    // Eval::evaluate() performs no search.  set_from_packed_sfen() leaves the
    // accumulator invalid, so this call takes the normal full-refresh path.
    // Its return convention is side-to-move, matching PackedSfenValue::score.
    const Value evaluated = ::YaneuraOu::Eval::evaluate(position);
    const s16 static_score = static_cast<s16>(evaluated);

    if (metadata.is_open()) {
      const int material_black = position.state()->materialValue;
      const int material_stm = material_black
          * (position.side_to_move() == BLACK ? 1 : -1);
#if defined(ENABLE_NNUE_SIGNAL_LOG)
      const auto& signal_access = LastNnueSignalAccess();
      const int selected_bucket = signal_access.signal.valid
          ? signal_access.signal.selected_bucket : -1;
#else
      const int selected_bucket = -1;
#endif
      const StaticEvalMetadata value{
          material_black, material_stm, record.game_ply,
          static_cast<std::uint8_t>(std::clamp(selected_bucket, 0, 255)),
          static_cast<std::uint8_t>(position.side_to_move())};
      metadata.write(reinterpret_cast<const char*>(&value), sizeof(value));
      if (!metadata) {
        failed = true;
        failure = "failed while writing metadata output";
        break;
      }
    }

    output_bytes = original_bytes;
    std::memcpy(output_bytes.data() + offsetof(MoveAccuracyRecord, score),
                &static_score, sizeof(static_score));
    for (std::size_t i = 0; i < output_bytes.size(); ++i)
      if ((i < offsetof(MoveAccuracyRecord, score)
           || i >= offsetof(MoveAccuracyRecord, score) + sizeof(static_score))
          && output_bytes[i] != original_bytes[i])
        ++non_score_byte_mismatches;
    output.write(output_bytes.data(), output_bytes.size());
    if (!output) {
      failed = true;
      failure = "failed while writing temporary output";
      break;
    }

    if (csv && processed < csv_records) {
      csv << (start_record + processed) << ',';
      WriteCsvField(csv, position.sfen());
      csv << ',' << static_cast<int>(position.side_to_move())
          << ',' << record.score << ',' << static_score
          << ',' << (static_cast<int>(record.score) - static_cast<int>(static_score))
          << ',' << record.game_ply
          << ',' << static_cast<int>(record.game_result)
          << ',' << Move16(record.move).to_usi_string() << '\n';
    }
    ++processed;

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5)) {
      last_report = now;
      const double seconds = std::chrono::duration<double>(now - started).count();
      const double rate = seconds > 0.0 ? processed / seconds : 0.0;
      const double eta = rate > 0.0 ? (requested - processed) / rate : 0.0;
      std::cout << "static_eval_bin progress=" << processed << '/' << requested
                << " records_per_sec=" << std::fixed << std::setprecision(1)
                << rate << " eta_sec=" << std::setprecision(0) << eta
                << std::endl;
    }
  }

  if (!failed && processed != requested) {
    failed = true;
    failure = "input ended before the requested record count";
  }
  output.flush();
  if (!failed && !output) {
    failed = true;
    failure = "failed while flushing temporary output";
  }
  if (csv) {
    csv.flush();
    if (!failed && !csv) {
      failed = true;
      failure = "failed while flushing validation CSV";
    }
  }
  output.close();
  if (csv)
    csv.close();
  if (metadata.is_open()) {
    metadata.flush();
    if (!failed && !metadata) {
      failed = true;
      failure = "failed while flushing metadata output";
    }
    metadata.close();
  }

  if (failed || decode_errors != 0 || illegal_positions != 0
      || non_score_byte_mismatches != 0) {
    std::remove(temporary_name.c_str());
    if (!csv_temporary_name.empty())
      std::remove(csv_temporary_name.c_str());
    if (!metadata_temporary_name.empty())
      std::remove(metadata_temporary_name.c_str());
    std::cout << "static_eval_bin status=error processed=" << processed
              << " decode_errors=" << decode_errors
              << " illegal_positions=" << illegal_positions
              << " non_score_byte_mismatches=" << non_score_byte_mismatches
              << " message=" << failure << std::endl;
    return;
  }

  if (std::rename(temporary_name.c_str(), output_name.c_str()) != 0) {
    std::cout << "error: failed to rename completed output: "
              << temporary_name << std::endl;
    return;
  }
  if (!csv_temporary_name.empty()
      && std::rename(csv_temporary_name.c_str(), csv_name.c_str()) != 0) {
    std::cout << "error: output completed, but validation CSV rename failed: "
              << csv_temporary_name << std::endl;
    return;
  }
  if (!metadata_temporary_name.empty()
      && std::rename(metadata_temporary_name.c_str(), metadata_name.c_str()) != 0) {
    std::cout << "error: output completed, but metadata rename failed: "
              << metadata_temporary_name << std::endl;
    return;
  }

  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "static_eval_bin status=ok input_records=" << total_records
            << " start_record=" << start_record
            << " processed=" << processed
            << " output_bytes=" << processed * sizeof(MoveAccuracyRecord)
            << " decode_errors=0 illegal_positions=0"
            << " non_score_byte_mismatches=0 records_per_sec="
            << std::fixed << std::setprecision(1)
            << (seconds > 0.0 ? processed / seconds : 0.0)
            << " elapsed_sec=" << std::setprecision(3) << seconds
            << std::endl;
}

#endif

#if defined(ENABLE_NNUE_BENCH)
// Diagnostic decoder for Phase-1 teacher-corpus inventory.  It deliberately
// consumes the same 40-byte record layout as the existing accuracy command;
// no NNUE value is evaluated and production paths are untouched.
void InspectPackedSfenSample(std::istream& stream) {
  std::string input_name;
  std::string output_name;
  stream >> std::quoted(input_name) >> std::quoted(output_name);
  int include_sfen = 0;
  stream >> include_sfen;
  if (input_name.empty() || output_name.empty()) {
    std::cout << "error: inspect_packed_sfen_sample requires input and output paths"
              << std::endl;
    return;
  }

  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  if (!input || !output) {
    std::cout << "error: failed to open packed-sfen sample input/output" << std::endl;
    return;
  }

  output << "sample_index,decoded,score,game_ply,move_raw,move_none_or_zero,"
            "move16_structurally_valid,side_to_move,material_black,material_stm,"
            "teacher_move_pseudo_legal,teacher_move_legal,illegal_reason,"
            "legal_move_count,teacher_capture,teacher_quiet,teacher_check,"
            "teacher_promotion,teacher_drop,teacher_piece_type";
  if (include_sfen)
    output << ",teacher_move_usi,sfen,legal_move_features";
  output << '\n';
  std::uint64_t sample_index = 0;
  std::uint64_t decode_errors = 0;
  MoveAccuracyRecord record;
  while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position position;
    StateInfo state;
    const bool decoded = position
        .set_from_packed_sfen(record.sfen, &state, false, record.game_ply)
        .is_ok();
    output << sample_index++ << ',' << (decoded ? 1 : 0);
    if (!decoded) {
      ++decode_errors;
      output << ",,,,,,,,,,,packed_sfen_decode_failed,,,,,,,";
      if (include_sfen)
        output << ",,,";
      output << '\n';
      continue;
    }

    const int material_black = static_cast<int>(Eval::material(position));
    const int side = static_cast<int>(position.side_to_move());
    const Move16 move16(record.move);
    const bool none_or_zero = move16 == Move16::none();
    const bool ordinary_type = !(move16.is_drop() && move16.is_promote());
    const bool valid_to = static_cast<int>(move16.to_sq()) >= 0
                       && static_cast<int>(move16.to_sq()) < SQ_NB;
    const bool valid_from_or_drop = move16.is_drop()
      ? (PAWN <= move16.move_dropped_piece()
         && move16.move_dropped_piece() < KING)
      : (static_cast<int>(move16.from_sq()) >= 0
         && static_cast<int>(move16.from_sq()) < SQ_NB);
    const bool structurally_valid = !none_or_zero && move16.is_ok()
                                 && ordinary_type && valid_to
                                 && valid_from_or_drop;
    bool pseudo_legal = false;
    bool legal = false;
    bool teacher_capture = false;
    bool teacher_check = false;
    bool teacher_promotion = false;
    bool teacher_drop = false;
    int teacher_piece_type = 0;
    Move teacher_move = Move::none();
    std::string illegal_reason;
    if (structurally_valid) {
      teacher_move = position.to_move(move16);
      pseudo_legal = position.pseudo_legal_s<true>(teacher_move);
      legal = pseudo_legal && position.legal(teacher_move);
      if (legal) {
        teacher_capture = position.capture(teacher_move);
        teacher_check = position.gives_check(teacher_move);
        teacher_promotion = teacher_move.is_promote();
        teacher_drop = teacher_move.is_drop();
        teacher_piece_type = static_cast<int>(
            raw_type_of(position.moved_piece_before(teacher_move)));
      }
      if (!pseudo_legal)
        illegal_reason = "pseudo_legal_false";
      else if (!legal)
        illegal_reason = "legal_false_self_check";
    } else if (none_or_zero) {
      illegal_reason = "move_none_or_zero";
    } else if (move16 == Move16::null()) {
      illegal_reason = "move_null";
    } else if (move16 == Move16::resign()) {
      illegal_reason = "move_resign";
    } else if (move16 == Move16::win()) {
      illegal_reason = "move_win";
    } else {
      illegal_reason = "move16_structurally_invalid";
    }
    MoveList<LEGAL_ALL> legal_moves(position);
    output << ',' << record.score << ',' << record.game_ply << ',' << record.move
           << ',' << (none_or_zero ? 1 : 0)
           << ',' << (structurally_valid ? 1 : 0)
           << ',' << side << ',' << material_black << ','
           << (position.side_to_move() == BLACK ? material_black : -material_black)
           << ',' << (pseudo_legal ? 1 : 0) << ',' << (legal ? 1 : 0)
           << ',' << illegal_reason << ',' << legal_moves.size()
           << ',' << (teacher_capture ? 1 : 0)
           << ',' << (legal && !teacher_capture ? 1 : 0)
           << ',' << (teacher_check ? 1 : 0)
           << ',' << (teacher_promotion ? 1 : 0)
           << ',' << (teacher_drop ? 1 : 0)
           << ',' << teacher_piece_type;
    if (include_sfen) {
      output << ',' << (structurally_valid ? move16.to_usi_string() : std::string())
             << ',' << position.sfen() << ',';
      bool first_move = true;
      for (const auto& ext_move : legal_moves) {
        const Move move = ext_move;
        if (!first_move)
          output << '|';
        first_move = false;
        const Move16 compact(move.to_u16());
        const int from_or_drop = move.is_drop()
            ? 81 + static_cast<int>(move.move_dropped_piece())
            : static_cast<int>(move.from_sq());
        const int piece_type = static_cast<int>(
            raw_type_of(position.moved_piece_before(move)));
        const Color us = position.side_to_move();
        const Color them = ~us;
        const Square to = move.to_sq();
        const Square normalized_to = us == BLACK ? to : Inv(to);
        const Square own_king = us == BLACK
            ? position.square<KING>(us) : Inv(position.square<KING>(us));
        const Square enemy_king = us == BLACK
            ? position.square<KING>(them) : Inv(position.square<KING>(them));
        const int captured_type = position.capture(move)
            ? static_cast<int>(type_of(position.piece_on(to))) : 0;
        const int moved_after_type = static_cast<int>(
            type_of(position.moved_piece_after(move)));
        const int delta_file = move.is_drop() ? 17
            : int(file_of(normalized_to))
              - int(file_of(us == BLACK ? move.from_sq() : Inv(move.from_sq()))) + 8;
        const int delta_rank = move.is_drop() ? 17
            : int(rank_of(normalized_to))
              - int(rank_of(us == BLACK ? move.from_sq() : Inv(move.from_sq()))) + 8;
        const int own_king_file = int(file_of(normalized_to)) - int(file_of(own_king)) + 8;
        const int own_king_rank = int(rank_of(normalized_to)) - int(rank_of(own_king)) + 8;
        const int enemy_king_file = int(file_of(normalized_to)) - int(file_of(enemy_king)) + 8;
        const int enemy_king_rank = int(rank_of(normalized_to)) - int(rank_of(enemy_king)) + 8;
        output << compact.to_u16() << ':' << from_or_drop << ':'
               << static_cast<int>(move.to_sq()) << ':' << piece_type << ':'
               << (move.is_promote() ? 1 : 0) << ':'
               << (move.is_drop() ? 1 : 0) << ':'
               << (position.capture(move) ? 1 : 0) << ':'
               << (position.gives_check(move) ? 1 : 0) << ':'
               << captured_type << ':' << moved_after_type << ':'
               << delta_file << ':' << delta_rank << ':'
               << int(file_of(normalized_to)) << ':' << int(rank_of(normalized_to)) << ':'
               << own_king_file << ':' << own_king_rank << ':'
               << enemy_king_file << ':' << enemy_king_rank << ':'
               << dist(normalized_to, enemy_king) << ':'
               << std::min(int(position.board_effect[us].effect(to)), 3) << ':'
               << std::min(int(position.board_effect[them].effect(to)), 3);
      }
    }
    output << '\n';
  }

  std::cout << "inspect_packed_sfen_sample: records=" << sample_index
            << " decode_errors=" << decode_errors
            << " output=" << output_name << std::endl;
}

// Experiment 73 only: compact shogi-specific tactical features. All legality
// and pin semantics are delegated to Position/MoveList. This command exists
// only in ENABLE_NNUE_BENCH diagnostic binaries.
void ExportShogiTacticalFeatures(std::istream& stream) {
  std::string input_name, output_name;
  std::uint64_t max_records = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_name) >> max_records;
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  if (!input || !output) {
    std::cout << "error: export_shogi_tactical_features open failed" << std::endl;
    return;
  }
  output << "record,decoded,stm"
            ",s1_legal_mask_p0,s1_safe_mask_p0,s1_friendly_p0,s1_enemy_p0,s1_enemy_control_p0,s1_friendly_defended_p0"
            ",s1_legal_mask_p1,s1_safe_mask_p1,s1_friendly_p1,s1_enemy_p1,s1_enemy_control_p1,s1_friendly_defended_p1";
  for (int p = 0; p < 2; ++p)
    for (int pt = PAWN; pt < KING; ++pt)
      output << ",s2_legal_check_p" << p << "_pt" << pt;
  for (int p = 0; p < 2; ++p)
    for (int pt = PAWN; pt < KING; ++pt)
      output << ",s2_hand_p" << p << "_pt" << pt;
  output << ",s2_pseudo_pawn_check_p0,s2_pseudo_pawn_check_p1"
            ",s3_count_p0,s3_count_p1,s3_indices_p0,s3_indices_p1\n";

  auto direction_index = [](Square from, Square to, Color c) {
    if (c == WHITE) { from = Inv(from); to = Inv(to); }
    const int df = int(file_of(to)) - int(file_of(from));
    const int dr = int(rank_of(to)) - int(rank_of(from));
    int index = (df + 1) * 3 + dr + 1;
    if (index > 4) --index;
    return index;
  };

  MoveAccuracyRecord record;
  std::uint64_t row = 0, decode_errors = 0;
  while ((max_records == 0 || row < max_records)
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position original;
    StateInfo original_state;
    const bool decoded = original
        .set_from_packed_sfen(record.sfen, &original_state, false, record.game_ply)
        .is_ok();
    output << row << ',' << (decoded ? 1 : 0);
    if (!decoded) {
      ++decode_errors;
      output << ",0";
      for (int i = 0; i < 47; ++i) output << ',';
      output << '\n'; ++row; continue;
    }
    const Color stm = original.side_to_move();
    output << ',' << int(stm);
    std::array<int, 2> legal_mask{}, safe_mask{}, friendly_mask{}, enemy_mask{},
                       enemy_control_mask{}, friendly_defended_mask{}, pseudo_pawn{};
    std::array<std::array<int, 7>, 2> drop_checks{}, hand_counts{};

    const std::string original_sfen = original.sfen();
    for (int perspective = 0; perspective < 2; ++perspective) {
      const Color us = perspective == 0 ? stm : ~stm;
      std::string sfen = original_sfen;
      const auto side_pos = sfen.find(' ');
      if (side_pos != std::string::npos && side_pos + 1 < sfen.size())
        sfen[side_pos + 1] = us == BLACK ? 'b' : 'w';
      Position pos;
      StateInfo state;
      pos.set(sfen, &state);
      for (int pt = PAWN; pt < KING; ++pt)
        hand_counts[perspective][pt - 1] = hand_count(pos.hand_of(us), PieceType(pt));
      const Square king = pos.square<KING>(us);
      for (const auto& ext_move : MoveList<LEGAL_ALL>(pos)) {
        const Move move = ext_move;
        if (type_of(pos.moved_piece_before(move)) == KING) {
          const int d = direction_index(king, move.to_sq(), us);
          if (0 <= d && d < 8) legal_mask[perspective] |= 1 << d;
        }
        if (move.is_drop() && pos.gives_check(move)) {
          const int pt = int(move.move_dropped_piece());
          if (PAWN <= pt && pt < KING) ++drop_checks[perspective][pt - 1];
        }
      }
      safe_mask[perspective] = legal_mask[perspective];

      Square normalized_king = us == BLACK ? king : Inv(king);
      const int kf = int(file_of(normalized_king));
      const int kr = int(rank_of(normalized_king));
      int direction = 0;
      for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr) {
          if (df == 0 && dr == 0) continue;
          const int nf = kf + df, nr = kr + dr;
          if (0 <= nf && nf < 9 && 0 <= nr && nr < 9) {
            Square to = File(nf) | Rank(nr);
            if (us == WHITE) to = Inv(to);
            const Piece piece = pos.piece_on(to);
            if (piece != NO_PIECE) {
              if (color_of(piece) == us) friendly_mask[perspective] |= 1 << direction;
              else enemy_mask[perspective] |= 1 << direction;
            }
            if (pos.board_effect[~us].effect(to)) enemy_control_mask[perspective] |= 1 << direction;
            if (pos.board_effect[us].effect(to)) friendly_defended_mask[perspective] |= 1 << direction;
          }
          ++direction;
        }

      for (const auto& ext_move : MoveList<CHECKS_ALL>(pos)) {
        const Move move = ext_move;
        if (move.is_drop() && move.move_dropped_piece() == PAWN)
          ++pseudo_pawn[perspective];
      }
    }
    for (int p = 0; p < 2; ++p)
      output << ',' << legal_mask[p] << ',' << safe_mask[p]
             << ',' << friendly_mask[p] << ',' << enemy_mask[p]
             << ',' << enemy_control_mask[p] << ',' << friendly_defended_mask[p];
    for (int p = 0; p < 2; ++p)
      for (int pt = 0; pt < 7; ++pt) output << ',' << drop_checks[p][pt];
    for (int p = 0; p < 2; ++p)
      for (int pt = 0; pt < 7; ++pt) output << ',' << hand_counts[p][pt];
    output << ',' << pseudo_pawn[0] << ',' << pseudo_pawn[1];

    std::array<std::vector<int>, 2> pin_indices;
    for (int perspective = 0; perspective < 2; ++perspective) {
      const Color us = perspective == 0 ? stm : ~stm;
      const Square king = original.square<KING>(us);
      Bitboard pinned = original.pinned_pieces(us);
      Bitboard pinners = original.pinners(~us);
      while (pinned) {
        const Square pinned_sq = pinned.pop();
        Bitboard pp = pinners;
        while (pp) {
          const Square pinner_sq = pp.pop();
          if (!(between_bb(king, pinner_sq) & pinned_sq)) continue;
          const int pinned_type = int(type_of(original.piece_on(pinned_sq))) - 1;
          const int pinner_raw = int(type_of(original.piece_on(pinner_sq)));
          int pinner_type = pinner_raw == LANCE ? 0
                          : pinner_raw == BISHOP ? 1
                          : pinner_raw == HORSE ? 2
                          : pinner_raw == ROOK ? 3 : 4;
          Square normalized_king = us == BLACK ? king : Inv(king);
          Square normalized_pinner = us == BLACK ? pinner_sq : Inv(pinner_sq);
          int df = int(file_of(normalized_pinner)) - int(file_of(normalized_king));
          int dr = int(rank_of(normalized_pinner)) - int(rank_of(normalized_king));
          df = (df > 0) - (df < 0); dr = (dr > 0) - (dr < 0);
          int direction = (df + 1) * 3 + dr + 1;
          if (direction > 4) --direction;
          const int distance = dist(king, pinner_sq);
          const int distance_class = distance <= 2 ? 0 : (distance <= 4 ? 1 : 2);
          pin_indices[perspective].push_back(
              (((pinned_type * 5 + pinner_type) * 8 + direction) * 3 + distance_class));
        }
      }
    }
    output << ',' << pin_indices[0].size() << ',' << pin_indices[1].size();
    for (int p = 0; p < 2; ++p) {
      output << ',';
      for (std::size_t i = 0; i < pin_indices[p].size(); ++i) {
        if (i) output << ';';
        output << pin_indices[p][i];
      }
    }
    output << '\n';
    ++row;
    if (row % 100000 == 0) std::cout << "tactical_features records=" << row << std::endl;
  }
  std::cout << "export_shogi_tactical_features records=" << row
            << " decode_errors=" << decode_errors << " output=" << output_name << std::endl;
}

struct CheapEscapeMasks {
  int onboard = 0;
  int exact = 0;
  int c1_effect_only = 0;
  int c2_occupancy_effect = 0;
  int c3_capture_xray = 0;
  int c4_origin_capture_xray = 0;
  int friendly_occupancy = 0;
  int enemy_occupancy = 0;
  int enemy_control = 0;
};

// Returns whether a slider attack on `to` appears after removing the king
// origin and/or the captured piece. No legal move generation is used here.
bool CheapEscapeSliderUnsafe(const Position& pos, Color us, Square king,
                             Square to, bool remove_origin,
                             bool remove_capture) {
  Bitboard occupied = pos.pieces();
  if (remove_origin)
    occupied ^= king;
  if (remove_capture && pos.piece_on(to) != NO_PIECE)
    occupied ^= to;
  const Color them = ~us;
  if (rookEffect(to, occupied) & pos.pieces(them, ROOK, DRAGON))
    return true;
  if (bishopEffect(to, occupied) & pos.pieces(them, BISHOP, HORSE))
    return true;
  // Reverse the lance direction: squares reached from `to` by our direction
  // are the enemy lances whose forward ray can reach `to`.
  if (lanceEffect(us, to, occupied) & pos.pieces(them, LANCE))
    return true;
  return false;
}

CheapEscapeMasks MakeCheapEscapeMasks(const Position& pos) {
  CheapEscapeMasks result;
  const Color us = pos.side_to_move();
  const Square king = pos.square<KING>(us);
  int direction = 0;
  for (int df = -1; df <= 1; ++df)
    for (int dr = -1; dr <= 1; ++dr) {
      if (df == 0 && dr == 0)
        continue;
      Square normalized_king = us == BLACK ? king : Inv(king);
      const int nf = int(file_of(normalized_king)) + df;
      const int nr = int(rank_of(normalized_king)) + dr;
      const int bit = 1 << direction++;
      if (nf < 0 || nf >= 9 || nr < 0 || nr >= 9)
        continue;
      result.onboard |= bit;
      Square to = File(nf) | Rank(nr);
      if (us == WHITE)
        to = Inv(to);
      const Piece piece = pos.piece_on(to);
      const bool friendly = piece != NO_PIECE && color_of(piece) == us;
      const bool enemy = piece != NO_PIECE && color_of(piece) != us;
      const bool controlled = bool(pos.board_effect[~us].effect(to));
      if (friendly) result.friendly_occupancy |= bit;
      if (enemy) result.enemy_occupancy |= bit;
      if (controlled) result.enemy_control |= bit;
      if (!controlled)
        result.c1_effect_only |= bit;
      if (!friendly && !controlled) {
        result.c2_occupancy_effect |= bit;
        bool c3_unsafe = enemy
            && CheapEscapeSliderUnsafe(pos, us, king, to, false, true);
        if (!c3_unsafe)
          result.c3_capture_xray |= bit;
        bool c4_unsafe = CheapEscapeSliderUnsafe(pos, us, king, to, true, enemy);
        if (!c4_unsafe)
          result.c4_origin_capture_xray |= bit;
      }
    }
  for (const auto& ext_move : MoveList<LEGAL_ALL>(pos)) {
    const Move move = ext_move;
    if (type_of(pos.moved_piece_before(move)) != KING)
      continue;
    Square normalized_king = us == BLACK ? king : Inv(king);
    Square normalized_to = us == BLACK ? move.to_sq() : Inv(move.to_sq());
    const int df = int(file_of(normalized_to)) - int(file_of(normalized_king));
    const int dr = int(rank_of(normalized_to)) - int(rank_of(normalized_king));
    int index = (df + 1) * 3 + dr + 1;
    if (index > 4) --index;
    if (0 <= index && index < 8)
      result.exact |= 1 << index;
  }
  return result;
}

// Experiment 74: export exact S1 and effect-board-only approximations. The
// cheap masks never call legal move generation; exact does so only as label.
void ExportCheapSafeEscapeFeatures(std::istream& stream) {
  std::string input_name, output_name;
  std::uint64_t max_records = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_name) >> max_records;
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  if (!input || !output) {
    std::cout << "error: export_cheap_safe_escape_features open failed" << std::endl;
    return;
  }
  output << "record,decoded";
  for (int p = 0; p < 2; ++p)
    output << ",onboard_p" << p << ",exact_p" << p << ",c1_p" << p
           << ",c2_p" << p << ",c3_p" << p << ",c4_p" << p
           << ",friendly_p" << p << ",enemy_p" << p << ",enemy_control_p" << p;
  output << '\n';
  MoveAccuracyRecord record;
  std::uint64_t row = 0, decode_errors = 0;
  while ((max_records == 0 || row < max_records)
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position original;
    StateInfo original_state;
    const bool decoded = original
        .set_from_packed_sfen(record.sfen, &original_state, false, record.game_ply)
        .is_ok();
    output << row << ',' << int(decoded);
    if (!decoded) {
      ++decode_errors;
      for (int i = 0; i < 18; ++i) output << ',';
      output << '\n'; ++row; continue;
    }
    const Color stm = original.side_to_move();
    const std::string original_sfen = original.sfen();
    for (int perspective = 0; perspective < 2; ++perspective) {
      const Color us = perspective == 0 ? stm : ~stm;
      std::string sfen = original_sfen;
      const auto side_pos = sfen.find(' ');
      if (side_pos != std::string::npos && side_pos + 1 < sfen.size())
        sfen[side_pos + 1] = us == BLACK ? 'b' : 'w';
      Position pos;
      StateInfo state;
      pos.set(sfen, &state);
      const auto m = MakeCheapEscapeMasks(pos);
      output << ',' << m.onboard << ',' << m.exact << ',' << m.c1_effect_only
             << ',' << m.c2_occupancy_effect << ',' << m.c3_capture_xray
             << ',' << m.c4_origin_capture_xray << ',' << m.friendly_occupancy
             << ',' << m.enemy_occupancy << ',' << m.enemy_control;
    }
    output << '\n';
    ++row;
    if (row % 100000 == 0)
      std::cout << "cheap_escape_features records=" << row << std::endl;
  }
  std::cout << "export_cheap_safe_escape_features records=" << row
            << " decode_errors=" << decode_errors << " output=" << output_name << std::endl;
}

// Per-position microbenchmark. `MakeCheapEscapeMasks` is deliberately not
// used for cheap timings because it also creates the exact label.
void BenchmarkCheapSafeEscapeFeatures(std::istream& stream) {
  std::string input_name, output_name;
  std::uint64_t max_records = 0;
  int repeats = 16;
  stream >> std::quoted(input_name) >> std::quoted(output_name) >> max_records >> repeats;
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  if (!input || !output || repeats <= 0) {
    std::cout << "error: benchmark_cheap_safe_escape_features open failed" << std::endl;
    return;
  }
  output << "record,decode_effect_ns,exact_ns,c1_ns,c2_ns,c3_ns,c4_ns\n";
  using Clock = std::chrono::steady_clock;
  MoveAccuracyRecord record;
  std::uint64_t row = 0;
  volatile int sink = 0;
  while ((max_records == 0 || row < max_records)
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos;
    StateInfo state;
    if (pos.set_from_packed_sfen(record.sfen, &state, false, record.game_ply).is_not_ok())
      continue;
    const Color us = pos.side_to_move();
    const Square king = pos.square<KING>(us);
    auto measure = [&](auto&& fn) {
      const auto begin = Clock::now();
      for (int r = 0; r < repeats; ++r) sink ^= fn();
      return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - begin).count()) / repeats;
    };
    const double decode_effect_ns = measure([&]() {
      Position fresh;
      StateInfo fresh_state;
      const auto status = fresh.set_from_packed_sfen(
          record.sfen, &fresh_state, false, record.game_ply);
      return status.is_ok() ? int(fresh.side_to_move()) + 1 : 0;
    });
    const double exact_ns = measure([&]() {
      int mask = 0;
      for (const auto& ext_move : MoveList<LEGAL_ALL>(pos)) {
        const Move move = ext_move;
        if (type_of(pos.moved_piece_before(move)) == KING)
          mask ^= int(move.to_sq()) + 1;
      }
      return mask;
    });
    auto cheap = [&](int mode) {
      int mask = 0, direction = 0;
      Square nk = us == BLACK ? king : Inv(king);
      for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr) {
          if (df == 0 && dr == 0) continue;
          const int bit = 1 << direction++;
          const int nf = int(file_of(nk)) + df, nr = int(rank_of(nk)) + dr;
          if (nf < 0 || nf >= 9 || nr < 0 || nr >= 9) continue;
          Square to = File(nf) | Rank(nr); if (us == WHITE) to = Inv(to);
          const Piece piece = pos.piece_on(to);
          const bool friendly = piece != NO_PIECE && color_of(piece) == us;
          const bool enemy = piece != NO_PIECE && color_of(piece) != us;
          if (pos.board_effect[~us].effect(to)) continue;
          if (mode >= 2 && friendly) continue;
          if (mode >= 3 && enemy
              && CheapEscapeSliderUnsafe(pos, us, king, to, false, true)) continue;
          if (mode >= 4
              && CheapEscapeSliderUnsafe(pos, us, king, to, true, enemy)) continue;
          mask |= bit;
        }
      return mask;
    };
    const double c1_ns = measure([&]() { return cheap(1); });
    const double c2_ns = measure([&]() { return cheap(2); });
    const double c3_ns = measure([&]() { return cheap(3); });
    const double c4_ns = measure([&]() { return cheap(4); });
    output << row << ',' << decode_effect_ns << ',' << exact_ns << ',' << c1_ns << ',' << c2_ns << ','
           << c3_ns << ',' << c4_ns << '\n';
    ++row;
  }
  std::cout << "benchmark_cheap_safe_escape_features records=" << row
            << " repeats=" << repeats << " sink=" << sink << std::endl;
}
#endif

#if defined(ENABLE_NNUE_POLICY_SHADOW)
void PolicyProbeSelftest(std::istream& stream) {
  std::string input_name, output_name;
  std::uint64_t limit = 1000;
  stream >> std::quoted(input_name) >> std::quoted(output_name) >> limit;
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  std::ofstream queries(output_name + ".queries.csv", std::ios::out | std::ios::trunc);
  if (!input || !output || !queries) {
    std::cout << "error: policy_probe_selftest failed to open input/output" << std::endl;
    return;
  }
  output << "record_index,move_raw,logit16,rank16,logit32,rank32\n";
  queries << "record_index,width,query\n";
  std::uint64_t index = 0, positions = 0, moves = 0, unavailable = 0;
  MoveAccuracyRecord record;
  while (index < limit && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position position;
    StateInfo state;
    if (position.set_from_packed_sfen(record.sfen, &state, false, record.game_ply).is_not_ok()) {
      ++index;
      continue;
    }
    // Force a fresh network pass so the diagnostic L2 query belongs to this
    // exact position rather than an accumulator/eval-hash cache entry.
    Eval::compute_eval(position);
    const auto& access = LastNnueSignalAccess();
    if (access.source != NnueSignalEvalSource::FreshNetwork || !access.signal.valid) {
      ++unavailable; ++index; continue;
    }
    queries << std::setprecision(9) << index << ",16,\"";
    for (int i = 0; i < 16; ++i) {
      if (i) queries << ';';
      queries << access.signal.policy_query16[i];
    }
    queries << "\"\n" << index << ",32,\"";
    for (int i = 0; i < 32; ++i) {
      if (i) queries << ';';
      queries << access.signal.policy_query32[i];
    }
    queries << "\"\n";
    struct Item { Move move; float p16, p32; int r16 = 0, r32 = 0; };
    std::vector<Item> items;
    for (const auto& ext : MoveList<LEGAL_ALL>(position)) {
      const Move move = ext;
      const auto feature = Search::NnuePolicyShadow::Features(position, move);
      items.push_back({move,
        PolicyProbe::Score<16>(access.signal.policy_query16, feature),
        PolicyProbe::Score<32>(access.signal.policy_query32, feature)});
    }
    const auto assign_rank = [&](auto value, auto rank) {
      std::vector<std::size_t> order(items.size());
      for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](const auto a, const auto b) {
        return items[a].*value > items[b].*value;
      });
      for (std::size_t i = 0; i < order.size(); ++i) items[order[i]].*rank = int(i + 1);
    };
    assign_rank(&Item::p16, &Item::r16);
    assign_rank(&Item::p32, &Item::r32);
    output << std::setprecision(9);
    for (const auto& item : items)
      output << index << ',' << item.move.to_u16() << ',' << item.p16 << ','
             << item.r16 << ',' << item.p32 << ',' << item.r32 << '\n';
    ++positions; moves += items.size(); ++index;
  }
  std::cout << "policy_probe_selftest positions=" << positions << " moves=" << moves
            << " unavailable=" << unavailable << " output=" << output_name << std::endl;
}
#endif

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

void WriteCsvField(std::ostream& output, std::string_view value) {
  output.put('"');
  for (const char character : value) {
    if (character == '"')
      output.put('"');
    output.put(character);
  }
  output.put('"');
}

void TestMoveAccuracy(IEngine& engine, std::istream& stream,
                      bool write_details) {
  std::string file_name;
  stream >> file_name;
  if (file_name.empty()) {
    std::cout << "error: sfenpack file path is required" << std::endl;
    return;
  }

  std::string detail_file_name;
  if (write_details) {
    stream >> detail_file_name;
    if (detail_file_name.empty()) {
      std::cout << "error: output CSV path is required" << std::endl;
      return;
    }
  }

  std::ifstream input(file_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open sfenpack file: " << file_name << std::endl;
    return;
  }

  std::ofstream detail_output;
  if (write_details) {
    detail_output.open(detail_file_name, std::ios::out | std::ios::trunc);
    if (!detail_output) {
      std::cout << "error: failed to open output CSV file: "
                << detail_file_name << std::endl;
      return;
    }
    detail_output
        << "record_index,tested_index,sfen,teacher_move,predicted_move,"
           "correct,teacher_score,game_ply,game_result\n";
  }

  std::uint64_t total_records = 0;
  std::uint64_t tested_positions = 0;
  std::uint64_t correct_moves = 0;
  std::uint64_t skipped_terminal_positions = 0;
  std::uint64_t skipped_declaration_wins = 0;
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
          ++skipped_terminal_positions;
          continue;
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
        // A declaration win has no comparable ordinary Move16 in the teacher
        // record.  It is not a network move-accuracy error, so omit it from
        // both numerator and denominator instead of aborting the whole corpus.
        // Move16::from_string() maps some non-move USI tokens to none(), so
        // test the callback text as well as the typed sentinel.
        if (best_move_text == "win" || best_move == Move16::win()) {
          ++skipped_declaration_wins;
          continue;
        }
        if (best_move == Move16::none() || best_move == Move16::resign()) {
          error_message = "depth=1 search returned no comparable best move: "
                        + best_move_text;
          break;
        }

        ++tested_positions;
        const u16 teacher_move = packed_record.move;
        const bool correct = best_move.to_u16() == teacher_move;
        if (correct)
          ++correct_moves;

        if (write_details) {
          detail_output << total_records << ',' << tested_positions << ',';
          WriteCsvField(detail_output, decoded_position.sfen());
          detail_output << ',';
          WriteCsvField(detail_output, Move16(teacher_move).to_usi_string());
          detail_output << ',';
          WriteCsvField(detail_output, best_move_text);
          detail_output << ',' << (correct ? 1 : 0) << ','
                        << packed_record.score << ',' << packed_record.game_ply
                        << ',' << static_cast<int>(packed_record.game_result)
                        << '\n';
          if (!detail_output) {
            error_message = "failed while writing output CSV file: "
                          + detail_file_name;
            break;
          }
        }
      }
    }
  }

  if (write_details) {
    detail_output.flush();
    if (error_message.empty() && !detail_output)
      error_message = "failed while writing output CSV file: "
                    + detail_file_name;
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
  std::cout << "skipped terminal positions = "
            << skipped_terminal_positions << std::endl;
  std::cout << "skipped declaration wins = " << skipped_declaration_wins
            << std::endl;
  std::cout << "accuracy=" << accuracy_text.str() << "%" << std::endl;
  if (write_details)
    std::cout << "detail CSV      = " << detail_file_name << std::endl;
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
                                 const char* left_name,
                                 const std::int64_t left_value,
                                 const char* right_name,
                                 const std::int64_t right_value,
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
    std::cout << "  comparison        : " << left_name << " vs " << right_name
              << std::endl
              << "  " << left_name << " value : " << left_value << std::endl
              << "  " << right_name << " value : " << right_value << std::endl
              << "  difference        : "
              << (left_value - right_value) << std::endl;
  };

  auto compare_accumulators = [&](const Accumulator& left,
                                  const Accumulator& right,
                                  const char* left_name,
                                  const char* right_name,
                                  const std::uint64_t game, const int ply,
                                  const Move move) {
    for (const Color perspective : {BLACK, WHITE}) {
      for (std::size_t trigger_index = 0;
           trigger_index < kRefreshTriggers.size(); ++trigger_index) {
        for (IndexType index = 0; index < kTransformedFeatureDimensions; ++index) {
          const std::int64_t left_value =
              left.accumulation[perspective][trigger_index][index];
          const std::int64_t right_value =
              right.accumulation[perspective][trigger_index][index];
          if (left_value != right_value) {
            print_value_failure(game, ply, move, perspective,
                                "main accumulation", index, left_name,
                                left_value, right_name, right_value,
                                trigger_index, true);
            return false;
          }
        }
      }

      const auto& left_factors = left.factors[perspective];
      const auto& right_factors = right.factors[perspective];
      struct FactorComparison {
        const char* target;
        const std::int64_t* left_values;
        const std::int64_t* right_values;
      };
      const FactorComparison factor_comparisons[] = {
          {"halfka.sum_v", left_factors.halfka.sum_v,
           right_factors.halfka.sum_v},
          {"halfka.sum_v2", left_factors.halfka.sum_v2,
           right_factors.halfka.sum_v2},
          {"ksdg.sum_v", left_factors.ksdg.sum_v,
           right_factors.ksdg.sum_v},
          {"ksdg.sum_v2", left_factors.ksdg.sum_v2,
           right_factors.ksdg.sum_v2},
      };

      for (const auto& comparison : factor_comparisons) {
        for (std::size_t index = 0; index < 32; ++index) {
          const std::int64_t left_value = comparison.left_values[index];
          const std::int64_t right_value = comparison.right_values[index];
          if (left_value != right_value) {
            print_value_failure(game, ply, move, perspective,
                                comparison.target, index, left_name,
                                left_value, right_name, right_value, 0, false);
            return false;
          }
        }
      }
    }
    return true;
  };

  std::cout << "start testing accumulator with random games";

#if defined(USE_FINNY_TABLES)
  feature_transformer->TestResetFinnyCache();
#endif

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

#if defined(USE_FINNY_TABLES)
      feature_transformer->TestRefreshAccumulatorWithFinny(pos);
      if (!current->accumulator.computed_accumulation) {
        print_state_failure(game, ply, "Finny refresh did not compute accumulator", &move);
        std::cout << "failed." << std::endl;
        return;
      }
      const Accumulator finny = current->accumulator;
#endif

      // compute_eval(pos) follows the configured refresh policy and can use
      // Finny. The test oracle must explicitly invoke the true scratch path.
      feature_transformer->TestRefreshAccumulatorFromScratch(pos);
      if (!current->accumulator.computed_accumulation) {
        print_state_failure(game, ply, "scratch refresh did not compute accumulator", &move);
        std::cout << "failed." << std::endl;
        return;
      }
      const Accumulator scratch = current->accumulator;

      if (!compare_accumulators(incremental, scratch, "incremental", "scratch",
                                game, ply, move)) {
        std::cout << "failed." << std::endl;
        return;
      }
#if defined(USE_FINNY_TABLES)
      if (!compare_accumulators(finny, scratch, "Finny", "scratch",
                                game, ply, move)) {
        std::cout << "failed." << std::endl;
        return;
      }
#endif
    }

    pos.set_hirate(&si);
    if ((game % 100) == 0)
      std::cout << "." << std::flush;
  }

  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves" << std::endl;
}

// Deterministic incremental-evaluation checksum for comparing separately
// linked production binaries. This is a test command only; no search or NNUE
// hot-path branch is added.
void TestIncrementalEvalChecksum() {
  constexpr std::uint64_t kSeed = 20171128;
  constexpr int kGames = 64;
  constexpr int kMaxPly = 128;
  constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
  constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
  Position pos;
  StateInfo root;
  std::vector<StateInfo> states(kMaxPly);
  PRNG prng(kSeed);
  std::uint64_t checksum = kFnvOffset;
  std::uint64_t positions = 0;

  for (int game = 0; game < kGames; ++game) {
    pos.set_hirate(&root);
    ::YaneuraOu::Eval::evaluate_with_no_return(pos);
    for (int ply = 0; ply < kMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);
      const Value value = ::YaneuraOu::Eval::evaluate(pos);
      checksum ^= static_cast<std::uint32_t>(static_cast<int>(value));
      checksum *= kFnvPrime;
      checksum ^= move.to_u16();
      checksum *= kFnvPrime;
      ++positions;
    }
  }
  std::cout << "incremental eval checksum positions = " << positions
            << std::endl
            << "incremental eval checksum = 0x" << std::hex << checksum
            << std::dec << std::endl;
}

#if defined(ENABLE_NNUE_BENCH)

constexpr std::uint64_t kNnueBenchSeed = 20171128;
constexpr std::uint64_t kNnueBenchWarmupGames = 8;
constexpr std::uint64_t kNnueBenchMeasuredGames = 64;
constexpr int kNnueBenchMaxPly = 128;

void ExportNnueCalibrationCorpus(std::istream& stream) {
  std::string output_file;
  std::size_t position_count = 64;
  stream >> std::quoted(output_file) >> position_count;
  if (output_file.empty() || position_count == 0) {
    std::cout << "usage: test nnue export_calibration_corpus \"file\" [count]"
              << std::endl;
    return;
  }

  std::ofstream output(output_file);
  if (!output) {
    std::cout << "failed to open calibration corpus: " << output_file << std::endl;
    return;
  }

  PRNG prng(kNnueBenchSeed);
  std::set<std::string> unique_positions;
  std::size_t attempts = 0;
  while (unique_positions.size() < position_count && attempts < position_count * 16) {
    Position position;
    StateInfo root;
    std::vector<StateInfo> states(kNnueBenchMaxPly);
    position.set_hirate(&root);
    const int target_ply = 8 + 16 * static_cast<int>(attempts % 8);
    for (int ply = 0; ply < target_ply; ++ply) {
      MoveList<LEGAL_ALL> moves(position);
      if (moves.size() == 0)
        break;
      position.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);
    }
    if (MoveList<LEGAL_ALL>(position).size() != 0)
      unique_positions.insert(position.sfen());
    ++attempts;
  }

  if (unique_positions.size() != position_count) {
    std::cout << "failed to generate requested calibration positions: "
              << unique_positions.size() << " / " << position_count << std::endl;
    return;
  }
  for (const auto& sfen : unique_positions)
    output << sfen << '\n';
  std::cout << "NNUE calibration corpus written: " << output_file
            << " (" << unique_positions.size() << " positions, seed="
            << kNnueBenchSeed << ')' << std::endl;
}

using NnueBenchClock = std::chrono::steady_clock;

struct NnueBenchTiming {
  std::uint64_t calls = 0;
  double nanoseconds = 0.0;
};


#if defined(ENABLE_NNUE_DECISION_TRACE)
std::uint64_t TraceSplitMix64(std::uint64_t& state) {
  std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void ExtractDecisionTraceRoots(std::istream& stream) {
  std::string input_name, output_name, metadata_name;
  std::uint64_t count = 0, seed = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_name)
         >> count >> seed >> std::quoted(metadata_name);
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::out | std::ios::trunc);
  std::ofstream metadata;
  if (!metadata_name.empty())
    metadata.open(metadata_name, std::ios::out | std::ios::trunc);
  if (!input || !output || (!metadata_name.empty() && !metadata) || count == 0) {
    std::cout << "error: trace_extract_roots input/output/count" << std::endl;
    return;
  }
  input.seekg(0, std::ios::end);
  const auto bytes = input.tellg();
  if (bytes <= 0 || std::uint64_t(bytes) % sizeof(MoveAccuracyRecord)) {
    std::cout << "error: invalid 40-byte PackedSfenValue stream" << std::endl;
    return;
  }
  const std::uint64_t records = std::uint64_t(bytes) / sizeof(MoveAccuracyRecord);
  if (count > records) {
    std::cout << "error: requested root count exceeds record count" << std::endl;
    return;
  }
  metadata << "root_order,source_record,root_hash,game_ply,score\n";
  std::unordered_set<std::uint64_t> selected;
  std::unordered_set<std::uint64_t> packed_hashes;
  selected.reserve(std::size_t(count * 2));
  packed_hashes.reserve(std::size_t(count * 2));
  std::uint64_t state = seed;
  std::uint64_t attempts = 0, decode_errors = 0, duplicates = 0, terminal = 0;
  while (packed_hashes.size() < count && attempts < count * 100) {
    ++attempts;
    const std::uint64_t index = TraceSplitMix64(state) % records;
    if (!selected.insert(index).second)
      continue;
    MoveAccuracyRecord record{};
    input.clear();
    input.seekg(std::streamoff(index * sizeof(record)), std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(&record), sizeof(record)))
      break;
    Position position;
    StateInfo position_state;
    if (position.set_from_packed_sfen(
          record.sfen, &position_state, false, record.game_ply).is_not_ok()) {
      ++decode_errors;
      continue;
    }
    if (MoveList<LEGAL_ALL>(position).size() == 0) {
      ++terminal;
      continue;
    }
    const auto hash = Search::NnueDecisionTrace::Fnv1a(&record.sfen, sizeof(record.sfen));
    if (!packed_hashes.insert(hash).second) {
      ++duplicates;
      continue;
    }
    const auto order = packed_hashes.size() - 1;
    output << position.sfen() << '\n';
    metadata << order << ',' << index << ',' << hash << ','
             << record.game_ply << ',' << record.score << '\n';
  }
  output.flush(); metadata.flush();
  std::cout << "trace_extract_roots requested=" << count
            << " written=" << packed_hashes.size()
            << " attempts=" << attempts
            << " decode_errors=" << decode_errors
            << " duplicate_positions=" << duplicates
            << " terminal=" << terminal << std::endl;
}
#endif

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
  FreshEvaluate,
  CachedEvaluate,
  TimerFloor,
};

void MixNnueBenchChecksum(std::uint64_t& checksum, const std::int64_t value) {
  checksum ^= static_cast<std::uint64_t>(value);
  checksum *= UINT64_C(1099511628211);
}

template<typename T>
inline void KeepNnueBenchObject(const T& object) {
#if defined(__GNUC__) || defined(__clang__)
  // A compiler barrier, not a runtime operation. It makes every byte of a
  // benchmark output observable without adding a hash loop to the timed body.
  asm volatile("" : : "m"(object) : "memory");
#else
  volatile const unsigned char* bytes =
      reinterpret_cast<volatile const unsigned char*>(&object);
  (void)bytes[0];
#endif
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

      if (operation == FtBenchOperation::CachedEvaluate)
        (void)::YaneuraOu::Eval::evaluate(pos);

      const auto begin = NnueBenchClock::now();
      if (operation == FtBenchOperation::IncrementalUpdate) {
        if (!feature_transformer->UpdateAccumulatorIfPossible(pos)) {
          std::cout << "error: benchmark incremental update was unavailable"
                    << std::endl;
          return {};
        }
      } else if (operation == FtBenchOperation::ForcedRefresh) {
        feature_transformer->BenchmarkRefreshAccumulator(pos);
      } else if (operation == FtBenchOperation::TransformOnly) {
        feature_transformer->Transform(
            pos, transformed.data(), diff_transformed.data(),
            abs_transformed.data(), false, NnueBenchMaterialBucket(pos));
      } else if (operation == FtBenchOperation::FreshEvaluate
                 || operation == FtBenchOperation::CachedEvaluate) {
        const Value value = ::YaneuraOu::Eval::evaluate(pos);
        KeepNnueBenchObject(value);
      } else {
        KeepNnueBenchObject(pos.state());
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
  NnueBenchSamples fresh_evaluate;
  NnueBenchSamples cached_evaluate;
  NnueBenchSamples timer_floor;

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

    RunFtBenchmarkPass(FtBenchOperation::FreshEvaluate,
                       kNnueBenchWarmupGames, nullptr, checksum);
    fresh_evaluate.Add(RunFtBenchmarkPass(
        FtBenchOperation::FreshEvaluate, kNnueBenchMeasuredGames, nullptr,
        checksum));

    RunFtBenchmarkPass(FtBenchOperation::CachedEvaluate,
                       kNnueBenchWarmupGames, nullptr, checksum);
    cached_evaluate.Add(RunFtBenchmarkPass(
        FtBenchOperation::CachedEvaluate, kNnueBenchMeasuredGames, nullptr,
        checksum));

    RunFtBenchmarkPass(FtBenchOperation::TimerFloor,
                       kNnueBenchWarmupGames, nullptr, checksum);
    timer_floor.Add(RunFtBenchmarkPass(
        FtBenchOperation::TimerFloor, kNnueBenchMeasuredGames, nullptr,
        checksum));
  }

  PrintNnueBenchSamples("incremental accumulator update", incremental);
  PrintNnueBenchSamples("forced full refresh", refresh);
  PrintNnueBenchSamples("Transform (precomputed accumulator)", transform);
  PrintNnueBenchSamples("fresh Eval::evaluate pipeline", fresh_evaluate);
  PrintNnueBenchSamples("cached-score Eval::evaluate", cached_evaluate);
  PrintNnueBenchSamples("per-position timer floor", timer_floor);

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

enum class MultiDeltaBin : std::size_t {
  OneRemoveOneAdd,
  Delta2Other,
  Delta3,
  Delta4,
  Delta5Plus,
  Count,
};

constexpr std::size_t kMultiDeltaBinCount =
    static_cast<std::size_t>(MultiDeltaBin::Count);
constexpr std::array<const char*, kMultiDeltaBinCount> kMultiDeltaBinNames = {
    "1-remove/1-add (existing fused)",
    "d=2 (other)",
    "d=3",
    "d=4",
    "d=5+",
};

int ClassifyMultiDeltaBin(const Features::IndexList& removed,
                          const Features::IndexList& added) {
  if (removed.size() == 1 && added.size() == 1)
    return static_cast<int>(MultiDeltaBin::OneRemoveOneAdd);
  const std::size_t delta_count = removed.size() + added.size();
  if (delta_count == 2)
    return static_cast<int>(MultiDeltaBin::Delta2Other);
  if (delta_count == 3)
    return static_cast<int>(MultiDeltaBin::Delta3);
  if (delta_count == 4)
    return static_cast<int>(MultiDeltaBin::Delta4);
  if (delta_count >= 5)
    return static_cast<int>(MultiDeltaBin::Delta5Plus);
  return -1;
}

struct MultiDeltaMainPass {
  std::array<NnueBenchTiming, kMultiDeltaBinCount> bins{};
  NnueBenchTiming multi_delta_copy;
  std::array<std::uint64_t, 6> delta_histogram{};
  std::uint64_t reset_samples = 0;
  std::uint64_t checksum = UINT64_C(14695981039346656037);
};

MultiDeltaMainPass RunMultiDeltaMainPass(const bool candidate,
                                         const std::uint64_t num_games,
                                         const bool collect_statistics) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  MultiDeltaMainPass result;

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);

      for (IndexType trigger = 0; trigger < kRefreshTriggers.size(); ++trigger) {
        Features::IndexList removed[2], added[2];
        bool reset[2];
        RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[trigger],
                                          removed, added, reset);
        for (const Color perspective : {BLACK, WHITE}) {
          if (reset[perspective]) {
            if (collect_statistics)
              ++result.reset_samples;
            continue;
          }

          const std::size_t d =
              removed[perspective].size() + added[perspective].size();
          if (collect_statistics)
            ++result.delta_histogram[std::min<std::size_t>(d, 5)];
          const int bin =
              ClassifyMultiDeltaBin(removed[perspective], added[perspective]);
          if (bin < 0)
            continue;

          auto* const destination =
              pos.state()->accumulator.accumulation[perspective][trigger];
          const auto* const previous = pos.state()->previous->accumulator
              .accumulation[perspective][trigger];

          if (d >= 2
              && !(removed[perspective].size() == 1
                   && added[perspective].size() == 1)) {
            const auto copy_begin = NnueBenchClock::now();
            std::memcpy(destination, previous,
                        FeatureTransformer::BenchmarkMainDimensions()
                            * sizeof(FeatureTransformer::BiasType));
            KeepNnueBenchObject(
                pos.state()->accumulator.accumulation[perspective][trigger]);
            const auto copy_end = NnueBenchClock::now();
            result.multi_delta_copy.nanoseconds +=
                std::chrono::duration<double, std::nano>(copy_end - copy_begin)
                    .count();
            ++result.multi_delta_copy.calls;
          }

          const auto begin = NnueBenchClock::now();
          feature_transformer->BenchmarkApplyMainDelta(
              previous, destination, removed[perspective], added[perspective],
              candidate);
          const auto end = NnueBenchClock::now();
          auto& timing = result.bins[static_cast<std::size_t>(bin)];
          timing.nanoseconds +=
              std::chrono::duration<double, std::nano>(end - begin).count();
          ++timing.calls;
          MixNnueBenchChecksum(result.checksum, destination[0]);
          MixNnueBenchChecksum(
              result.checksum,
              destination[FeatureTransformer::BenchmarkMainDimensions() - 1]);
        }
      }

      // Prepare an exact predecessor for the next move outside the timed body.
      feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
    }
  }
  return result;
}

enum class MultiDeltaWholeOperation { FullUpdate, FreshEvaluate };

NnueBenchTiming RunMultiDeltaWholePass(
    const bool candidate, const MultiDeltaWholeOperation operation,
    const std::uint64_t num_games, std::uint64_t& checksum) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  NnueBenchTiming timing;

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);

      const auto begin = NnueBenchClock::now();
      if (candidate)
        feature_transformer->BenchmarkUpdateAccumulatorMultiDelta(pos);
      else
        feature_transformer->BenchmarkUpdateAccumulatorLegacy(pos);
      if (operation == MultiDeltaWholeOperation::FreshEvaluate) {
        const Value value = ::YaneuraOu::Eval::evaluate(pos);
        KeepNnueBenchObject(value);
      }
      const auto end = NnueBenchClock::now();
      timing.nanoseconds +=
          std::chrono::duration<double, std::nano>(end - begin).count();
      ++timing.calls;
      ChecksumAccumulator(pos, checksum);
    }
  }
  return timing;
}

struct MultiDeltaCorrectness {
  std::uint64_t positions = 0;
  std::uint64_t mismatch_count = 0;
  std::uint64_t legacy_checksum = UINT64_C(14695981039346656037);
  std::uint64_t candidate_checksum = UINT64_C(14695981039346656037);
  int first_game = -1;
  int first_ply = -1;
};

MultiDeltaCorrectness CheckMultiDeltaCorrectness() {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  MultiDeltaCorrectness result;

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);

      feature_transformer->BenchmarkUpdateAccumulatorLegacy(pos);
      const Accumulator legacy = pos.state()->accumulator;
      ChecksumAccumulator(pos, result.legacy_checksum);

      feature_transformer->BenchmarkUpdateAccumulatorMultiDelta(pos);
      ChecksumAccumulator(pos, result.candidate_checksum);
      const auto& candidate = pos.state()->accumulator;
      const bool equal =
          std::memcmp(legacy.accumulation, candidate.accumulation,
                      sizeof(legacy.accumulation)) == 0
          && std::memcmp(legacy.factors, candidate.factors,
                         sizeof(legacy.factors)) == 0;
      if (!equal) {
        ++result.mismatch_count;
        if (result.first_game < 0) {
          result.first_game = static_cast<int>(game);
          result.first_ply = ply;
        }
      }
      ++result.positions;
    }
  }
  return result;
}

void PrintMultiDeltaBinSamples(
    const char* const heading,
    const std::array<NnueBenchSamples, kMultiDeltaBinCount>& samples) {
  std::cout << heading << std::endl;
  for (std::size_t bin = 0; bin < kMultiDeltaBinCount; ++bin)
    PrintNnueBenchSamples(kMultiDeltaBinNames[bin], samples[bin]);
}

void TestMultiDeltaAccumulatorBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: multi-delta Main accumulator]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  d definition : removed.size + added.size per "
               "trigger/perspective" << std::endl
            << "  order        : even=A,B odd=B,A" << std::endl;

  std::array<NnueBenchSamples, kMultiDeltaBinCount> legacy_main;
  std::array<NnueBenchSamples, kMultiDeltaBinCount> candidate_main;
  NnueBenchSamples copy_samples;
  NnueBenchSamples legacy_full;
  NnueBenchSamples candidate_full;
  NnueBenchSamples legacy_fresh;
  NnueBenchSamples candidate_fresh;
  MultiDeltaMainPass corpus_statistics;
  std::uint64_t legacy_checksum = UINT64_C(14695981039346656037);
  std::uint64_t candidate_checksum = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    const auto run_main = [&](const bool candidate) {
      RunMultiDeltaMainPass(candidate, kNnueBenchWarmupGames, false);
      const MultiDeltaMainPass pass = RunMultiDeltaMainPass(
          candidate, kNnueBenchMeasuredGames, repeat == 0 && !candidate);
      auto& destination = candidate ? candidate_main : legacy_main;
      for (std::size_t bin = 0; bin < kMultiDeltaBinCount; ++bin)
        destination[bin].Add(pass.bins[bin]);
      if (!candidate)
        copy_samples.Add(pass.multi_delta_copy);
      if (repeat == 0 && !candidate)
        corpus_statistics = pass;
    };
    const auto run_whole = [&](const bool candidate) {
      std::uint64_t& checksum = candidate
          ? candidate_checksum : legacy_checksum;
      RunMultiDeltaWholePass(candidate, MultiDeltaWholeOperation::FullUpdate,
                             kNnueBenchWarmupGames, checksum);
      (candidate ? candidate_full : legacy_full).Add(
          RunMultiDeltaWholePass(
              candidate, MultiDeltaWholeOperation::FullUpdate,
              kNnueBenchMeasuredGames, checksum));
      RunMultiDeltaWholePass(candidate,
                             MultiDeltaWholeOperation::FreshEvaluate,
                             kNnueBenchWarmupGames, checksum);
      (candidate ? candidate_fresh : legacy_fresh).Add(
          RunMultiDeltaWholePass(
              candidate, MultiDeltaWholeOperation::FreshEvaluate,
              kNnueBenchMeasuredGames, checksum));
    };

    if ((repeat & 1) == 0) {
      run_main(false);
      run_main(true);
      run_whole(false);
      run_whole(true);
    } else {
      run_main(true);
      run_main(false);
      run_whole(true);
      run_whole(false);
    }
  }

  PrintMultiDeltaBinSamples("A. legacy Main update", legacy_main);
  PrintMultiDeltaBinSamples("B. chunk-local Main update", candidate_main);
  PrintNnueBenchSamples("memcpy 2560-byte equivalent (multi-delta)",
                        copy_samples);
  PrintNnueBenchSamples("A. full update_accumulator", legacy_full);
  PrintNnueBenchSamples("B. full update_accumulator", candidate_full);
  PrintNnueBenchSamples("A. fresh Eval::evaluate", legacy_fresh);
  PrintNnueBenchSamples("B. fresh Eval::evaluate", candidate_fresh);

  const MultiDeltaCorrectness correctness = CheckMultiDeltaCorrectness();
  const std::uint64_t non_reset =
      corpus_statistics.delta_histogram[0]
      + corpus_statistics.delta_histogram[1]
      + corpus_statistics.delta_histogram[2]
      + corpus_statistics.delta_histogram[3]
      + corpus_statistics.delta_histogram[4]
      + corpus_statistics.delta_histogram[5];
  std::cout << "[delta frequency: one trigger/perspective sample]" << std::endl;
  for (std::size_t d = 0; d < corpus_statistics.delta_histogram.size(); ++d) {
    const auto count = corpus_statistics.delta_histogram[d];
    const double percent = non_reset == 0
        ? 0.0
        : 100.0 * static_cast<double>(count)
              / static_cast<double>(non_reset);
    std::cout << "  d=" << (d == 5 ? "5+" : std::to_string(d))
              << " : " << count << " (" << std::fixed
              << std::setprecision(3) << percent << "%)" << std::endl;
  }
  std::cout << "  reset : " << corpus_statistics.reset_samples << std::endl
            << "[correctness]" << std::endl
            << "  positions          : " << correctness.positions << std::endl
            << "  accumulator mismatch count : "
            << correctness.mismatch_count << std::endl
            << "  first mismatch     : game=" << correctness.first_game
            << " ply=" << correctness.first_ply << std::endl
            << "  legacy checksum    : 0x" << std::hex
            << correctness.legacy_checksum << std::endl
            << "  candidate checksum : 0x"
            << correctness.candidate_checksum << std::endl
            << "  checksum match     : "
            << (correctness.legacy_checksum
                        == correctness.candidate_checksum
                    ? "yes" : "no")
            << std::dec << std::endl
            << "[timing checksums]" << std::endl
            << "  legacy    : 0x" << std::hex << legacy_checksum << std::endl
            << "  candidate : 0x" << candidate_checksum << std::endl
            << "  match     : "
            << (legacy_checksum == candidate_checksum ? "yes" : "no")
            << std::dec << std::endl;
}

using Ksdg3Feature =
    Features::KingSafety3_DistinguishGolds<Features::Side::kFriend>;
using Ksdg3Variant = Features::Ksdg3BenchmarkVariant;

constexpr std::array<Ksdg3Variant, 4> kKsdg3Variants = {
    Ksdg3Variant::kBaseline,
    Ksdg3Variant::kEffectHoist,
    Ksdg3Variant::kNeighborTables,
    Ksdg3Variant::kCombined,
};

constexpr std::array<const char*, 4> kKsdg3VariantNames = {
    "A. baseline",
    "B. effect-board/opponent hoist",
    "C. neighbor + direction tables",
    "D. combined hoist + tables",
};

enum class Ksdg3BenchOperation {
  Active,
  Changed,
  RawChanged,
  FullUpdate,
};

struct Ksdg3CorpusStatistics {
  std::array<std::uint64_t, 3> dirty_num{};
  std::uint64_t positions = 0;
  std::uint64_t perspective_samples = 0;
  std::uint64_t reset_samples = 0;
  std::uint64_t non_reset_samples = 0;
  std::uint64_t valid_neighbors = 0;
  std::uint64_t removed = 0;
  std::uint64_t added = 0;
  std::uint64_t effect_changed_squares = 0;
};

struct Ksdg3PassResult {
  NnueBenchTiming timing;
  std::uint64_t perspective_calls = 0;
  Features::Ksdg3BenchmarkStageTiming stages;
};

bool EqualIndexList(const Features::IndexList& left,
                    const Features::IndexList& right) {
  if (left.size() != right.size())
    return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (left[i] != right[i])
      return false;
  return true;
}

void ChecksumIndexList(const Features::IndexList& list,
                       std::uint64_t& checksum) {
  MixNnueBenchChecksum(checksum, static_cast<std::int64_t>(list.size()));
  for (const auto index : list)
    MixNnueBenchChecksum(checksum, index);
}

void CollectKsdg3CorpusStatistics(const Position& pos,
                                  Ksdg3CorpusStatistics& statistics) {
  ++statistics.positions;
  const auto& dirty_piece = pos.state()->dirtyPiece;
  if (dirty_piece.dirty_num >= 0 && dirty_piece.dirty_num <= 2)
    ++statistics.dirty_num[dirty_piece.dirty_num];

  for (const Color perspective : {BLACK, WHITE}) {
    ++statistics.perspective_samples;
    const bool reset =
        dirty_piece.pieceNo[0] == PIECE_NUMBER_KING + perspective;
    if (reset)
      ++statistics.reset_samples;
    else
      ++statistics.non_reset_samples;

    Features::IndexList active;
    Ksdg3Feature::AppendActiveIndices(pos, perspective, &active);
    statistics.valid_neighbors += active.size();
    if (reset)
      continue;

    Features::IndexList removed;
    Features::IndexList added;
    Ksdg3Feature::AppendChangedIndices(
        pos, perspective, &removed, &added);
    statistics.removed += removed.size();
    statistics.added += added.size();

    const Color opponent = ~perspective;
    const Square king = pos.square<KING>(perspective);
    Bitboard dirty_squares(ZERO);
    for (int i = 0; i < dirty_piece.dirty_num; ++i) {
      for (const BonaPiece piece : {
               static_cast<BonaPiece>(
                   dirty_piece.changed_piece[i].old_piece.from[BLACK]),
               static_cast<BonaPiece>(
                   dirty_piece.changed_piece[i].new_piece.from[BLACK])}) {
        Square square;
        Piece board_piece;
        Ksdg3Feature::GetSquarePieceFromBonaPiece(
            piece, square, board_piece);
        if (square != SQ_NB && dist(king, square) <= 2)
          dirty_squares |= square;
      }
    }
    const SquareWithWall king_with_wall = to_sqww(king);
    for (Effect24::Direct direction : Effect24::Direct()) {
      const SquareWithWall square_with_wall =
          king_with_wall + Effect24::DirectToDeltaWW(direction);
      if (!is_ok(square_with_wall))
        continue;
      const Square square = sqww_to_sq(square_with_wall);
      if (dirty_squares & square)
        continue;
      const int previous_us = std::min(
          int(pos.board_effect_prev[perspective].effect(square)), 3);
      const int previous_them = std::min(
          int(pos.board_effect_prev[opponent].effect(square)), 3);
      const int current_us = std::min(
          int(pos.board_effect[perspective].effect(square)), 3);
      const int current_them = std::min(
          int(pos.board_effect[opponent].effect(square)), 3);
      statistics.effect_changed_squares +=
          previous_us != current_us || previous_them != current_them;
    }
  }
}

Ksdg3PassResult RunKsdg3BenchmarkPass(
    const Ksdg3BenchOperation operation, const Ksdg3Variant variant,
    const std::uint64_t num_games, Ksdg3CorpusStatistics* const statistics,
    std::uint64_t& checksum, const bool collect_stage_timing = false) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  Ksdg3PassResult result;
  Features::SetKsdg3BenchmarkVariant(variant);
  Features::SetKsdg3BenchmarkStageTiming(nullptr);

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      if (collect_stage_timing)
        Features::SetKsdg3BenchmarkStageTiming(&result.stages);
      const auto begin = NnueBenchClock::now();
      if (operation == Ksdg3BenchOperation::Active) {
        Features::IndexList active[COLOR_NB];
        for (const Color perspective : {BLACK, WHITE}) {
          Ksdg3Feature::AppendActiveIndices(pos, perspective,
                                            &active[perspective]);
          ++result.perspective_calls;
        }
        KeepNnueBenchObject(active);
      } else if (operation == Ksdg3BenchOperation::Changed) {
        Features::IndexList removed[COLOR_NB];
        Features::IndexList added[COLOR_NB];
        const auto& dirty_piece = pos.state()->dirtyPiece;
        for (const Color perspective : {BLACK, WHITE}) {
          if (dirty_piece.pieceNo[0] == PIECE_NUMBER_KING + perspective)
            continue;
          Ksdg3Feature::AppendChangedIndices(
              pos, perspective, &removed[perspective], &added[perspective]);
          ++result.perspective_calls;
        }
        KeepNnueBenchObject(removed);
        KeepNnueBenchObject(added);
      } else if (operation == Ksdg3BenchOperation::RawChanged) {
        Features::IndexList removed[COLOR_NB];
        Features::IndexList added[COLOR_NB];
        bool reset[COLOR_NB];
        RawFeatures::AppendChangedIndices(
            pos, kRefreshTriggers[0], removed, added, reset);
        result.perspective_calls += COLOR_NB;
        KeepNnueBenchObject(removed);
        KeepNnueBenchObject(added);
        KeepNnueBenchObject(reset);
      } else {
        if (!feature_transformer->UpdateAccumulatorIfPossible(pos)) {
          std::cout << "error: KSDG3 benchmark incremental update unavailable"
                    << std::endl;
          Features::SetKsdg3BenchmarkStageTiming(nullptr);
          Features::SetKsdg3BenchmarkVariant(Ksdg3Variant::kBaseline);
          return {};
        }
        result.perspective_calls += COLOR_NB;
      }
      const auto end = NnueBenchClock::now();
      // Checksum/statistics regenerate index lists outside the measured
      // interval. Do not charge those diagnostic calls to stages B/C.
      Features::SetKsdg3BenchmarkStageTiming(nullptr);
      result.timing.nanoseconds +=
          std::chrono::duration<double, std::nano>(end - begin).count();
      ++result.timing.calls;

      if (statistics != nullptr)
        CollectKsdg3CorpusStatistics(pos, *statistics);

      if (operation == Ksdg3BenchOperation::Active) {
        Features::IndexList active[COLOR_NB];
        for (const Color perspective : {BLACK, WHITE}) {
          Ksdg3Feature::AppendActiveIndices(pos, perspective,
                                            &active[perspective]);
          ChecksumIndexList(active[perspective], checksum);
        }
      } else if (operation == Ksdg3BenchOperation::Changed) {
        const auto& dirty_piece = pos.state()->dirtyPiece;
        for (const Color perspective : {BLACK, WHITE}) {
          if (dirty_piece.pieceNo[0] == PIECE_NUMBER_KING + perspective)
            continue;
          Features::IndexList removed;
          Features::IndexList added;
          Ksdg3Feature::AppendChangedIndices(
              pos, perspective, &removed, &added);
          ChecksumIndexList(removed, checksum);
          ChecksumIndexList(added, checksum);
        }
      } else if (operation == Ksdg3BenchOperation::RawChanged) {
        Features::IndexList removed[COLOR_NB];
        Features::IndexList added[COLOR_NB];
        bool reset[COLOR_NB];
        RawFeatures::AppendChangedIndices(
            pos, kRefreshTriggers[0], removed, added, reset);
        for (const Color perspective : {BLACK, WHITE}) {
          ChecksumIndexList(removed[perspective], checksum);
          ChecksumIndexList(added[perspective], checksum);
          MixNnueBenchChecksum(checksum, reset[perspective]);
        }
      } else {
        ChecksumAccumulator(pos, checksum);
      }
    }
  }

  Features::SetKsdg3BenchmarkStageTiming(nullptr);
  Features::SetKsdg3BenchmarkVariant(Ksdg3Variant::kBaseline);
  return result;
}

std::uint64_t ValidateKsdg3Candidates() {
  std::uint64_t mismatches = Features::ValidateKsdg3BenchmarkTables();
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);

      Features::SetKsdg3BenchmarkVariant(Ksdg3Variant::kBaseline);
      Features::IndexList baseline_active[COLOR_NB];
      Features::IndexList baseline_removed[COLOR_NB];
      Features::IndexList baseline_added[COLOR_NB];
      bool baseline_reset[COLOR_NB];
      for (const Color perspective : {BLACK, WHITE})
        Ksdg3Feature::AppendActiveIndices(
            pos, perspective, &baseline_active[perspective]);
      RawFeatures::AppendChangedIndices(
          pos, kRefreshTriggers[0], baseline_removed, baseline_added,
          baseline_reset);

      for (std::size_t variant_index = 1;
           variant_index < kKsdg3Variants.size(); ++variant_index) {
        Features::SetKsdg3BenchmarkVariant(kKsdg3Variants[variant_index]);
        Features::IndexList candidate_active[COLOR_NB];
        Features::IndexList candidate_removed[COLOR_NB];
        Features::IndexList candidate_added[COLOR_NB];
        bool candidate_reset[COLOR_NB];
        for (const Color perspective : {BLACK, WHITE})
          Ksdg3Feature::AppendActiveIndices(
              pos, perspective, &candidate_active[perspective]);
        RawFeatures::AppendChangedIndices(
            pos, kRefreshTriggers[0], candidate_removed, candidate_added,
            candidate_reset);
        for (const Color perspective : {BLACK, WHITE}) {
          mismatches += !EqualIndexList(
              baseline_active[perspective], candidate_active[perspective]);
          mismatches += !EqualIndexList(
              baseline_removed[perspective], candidate_removed[perspective]);
          mismatches += !EqualIndexList(
              baseline_added[perspective], candidate_added[perspective]);
          mismatches += baseline_reset[perspective]
                      != candidate_reset[perspective];
        }
      }

      Features::SetKsdg3BenchmarkVariant(Ksdg3Variant::kBaseline);
      if (!feature_transformer->UpdateAccumulatorIfPossible(pos))
        ++mismatches;
    }
  }
  Features::SetKsdg3BenchmarkVariant(Ksdg3Variant::kBaseline);
  return mismatches;
}

void PrintKsdg3Samples(const char* const label,
                       const NnueBenchSamples& samples,
                       const std::uint64_t perspective_calls) {
  PrintNnueBenchSamples(label, samples);
  if (!samples.ns_per_call.empty() && perspective_calls != 0) {
    const auto summary = SummarizeNnueBenchSamples(samples);
    const double perspectives_per_position =
        static_cast<double>(perspective_calls)
        / static_cast<double>(samples.calls_per_repeat);
    std::cout << "  median ns/perspective: " << std::fixed
              << std::setprecision(1)
              << summary.median / perspectives_per_position << std::endl;
  }
}

void TestKsdg3FeaturesBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: KingSafety3_DistinguishGolds features]"
            << std::endl
            << "  seed           : " << kNnueBenchSeed << std::endl
            << "  warm-up games  : " << kNnueBenchWarmupGames << std::endl
            << "  measured games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game   : " << kNnueBenchMaxPly << std::endl
            << "  repeats        : " << repeat_count << std::endl
            << "  variant order  : rotated per repeat" << std::endl;

  constexpr std::size_t kOperationCount = 4;
  std::array<std::array<NnueBenchSamples, kOperationCount>, 4> samples;
  std::array<NnueBenchSamples, 4> dirty_stage_samples;
  std::array<NnueBenchSamples, 4> neighbor_stage_samples;
  std::array<std::uint64_t, kOperationCount> perspective_calls{};
  std::array<std::uint64_t, 4> checksums{};
  checksums.fill(UINT64_C(14695981039346656037));
  Ksdg3CorpusStatistics statistics;

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t order = 0; order < kKsdg3Variants.size(); ++order) {
      const std::size_t variant_index = (order + repeat) % 4;
      const auto variant = kKsdg3Variants[variant_index];
      for (std::size_t operation_index = 0;
           operation_index < kOperationCount; ++operation_index) {
        const auto operation = static_cast<Ksdg3BenchOperation>(operation_index);
        RunKsdg3BenchmarkPass(operation, variant, kNnueBenchWarmupGames,
                              nullptr, checksums[variant_index]);
        auto pass = RunKsdg3BenchmarkPass(
            operation, variant, kNnueBenchMeasuredGames,
            repeat == 0 && variant_index == 0 && operation_index == 1
                ? &statistics : nullptr,
            checksums[variant_index]);
        samples[variant_index][operation_index].Add(pass.timing);
        if (repeat == 0)
          perspective_calls[operation_index] = pass.perspective_calls;
        if (operation == Ksdg3BenchOperation::Changed) {
          // B/C need clocks inside AppendChangedIndices. Measure those in a
          // separate pass so their clock calls do not contaminate D.
          std::uint64_t stage_checksum = UINT64_C(14695981039346656037);
          auto stage_pass = RunKsdg3BenchmarkPass(
              operation, variant, kNnueBenchMeasuredGames, nullptr,
              stage_checksum, true);
          if (stage_pass.stages.calls == 0)
            continue;
          dirty_stage_samples[variant_index].Add({
              stage_pass.stages.calls,
              stage_pass.stages.dirty_nanoseconds});
          neighbor_stage_samples[variant_index].Add({
              stage_pass.stages.calls,
              stage_pass.stages.neighbor_nanoseconds});
        }
      }
    }
  }

  const std::uint64_t mismatch_count = ValidateKsdg3Candidates();
  const std::array<const char*, kOperationCount> operation_names = {
      "A. KSDG3 AppendActiveIndices",
      "D. KSDG3 complete AppendChangedIndices",
      "E. RawFeatures complete AppendChangedIndices",
      "G. full update_accumulator",
  };
  for (std::size_t variant = 0; variant < 4; ++variant) {
    std::cout << '[' << kKsdg3VariantNames[variant] << ']' << std::endl;
    for (std::size_t operation = 0; operation < kOperationCount; ++operation)
      PrintKsdg3Samples(operation_names[operation],
                       samples[variant][operation],
                       perspective_calls[operation]);
    PrintNnueBenchSamples("B. dirty old/new processing (instrumented)",
                          dirty_stage_samples[variant]);
    PrintNnueBenchSamples("C. 24-neighbor comparison (instrumented)",
                          neighbor_stage_samples[variant]);

    // The same fixed corpus/order is used for E and G. Their difference is a
    // useful no-extra-hook estimate of accumulator application cost; report it
    // explicitly as derived rather than pretending it is an independently
    // timed interval.
    const auto raw = SummarizeNnueBenchSamples(samples[variant][2]);
    const auto full = SummarizeNnueBenchSamples(samples[variant][3]);
    std::cout << "F. accumulator application estimate (G median - E median)"
              << std::endl
              << "  median ns/position: " << std::fixed << std::setprecision(1)
              << (full.median - raw.median) << std::endl
              << "  checksum          : 0x" << std::hex << checksums[variant]
              << std::dec << std::endl;
  }

  const auto percentage = [](const std::uint64_t numerator,
                             const std::uint64_t denominator) {
    return denominator == 0 ? 0.0
        : 100.0 * static_cast<double>(numerator) / denominator;
  };
  std::cout << "[fixed corpus statistics]" << std::endl
            << "  positions              : " << statistics.positions << std::endl
            << "  perspective samples     : "
            << statistics.perspective_samples << std::endl
            << "  dirty_num 0 / 1 / 2     : " << statistics.dirty_num[0]
            << " / " << statistics.dirty_num[1]
            << " / " << statistics.dirty_num[2] << std::endl
            << "  reset / non-reset       : " << statistics.reset_samples
            << " / " << statistics.non_reset_samples << std::endl
            << "  valid neighbors total   : " << statistics.valid_neighbors
            << std::endl
            << "  mean valid/perspective  : " << std::fixed
            << std::setprecision(3)
            << static_cast<double>(statistics.valid_neighbors)
                 / statistics.perspective_samples << std::endl
            << "  removed / added         : " << statistics.removed
            << " / " << statistics.added << std::endl
            << "  effect-changed squares  : "
            << statistics.effect_changed_squares << std::endl
            << "  reset rate              : "
            << percentage(statistics.reset_samples,
                          statistics.perspective_samples)
            << "%" << std::endl
            << "[correctness]" << std::endl
            << "  table/corpus mismatches : " << mismatch_count << std::endl
            << "  active/removed/added order match: "
            << (mismatch_count == 0 ? "yes" : "no") << std::endl;
}

constexpr std::array<Ksdg3Variant, 3> kLongEffectMaskVariants = {
    Ksdg3Variant::kBaseline,
    Ksdg3Variant::kMaskOnly,
    Ksdg3Variant::kTouchedMask,
};

constexpr std::array<const char*, 3> kLongEffectMaskVariantNames = {
    "A. production baseline",
    "B. touched mask generation only",
    "C. touched mask + KSDG3 candidate",
};

void ConfigureLongEffectMaskVariant(const Ksdg3Variant variant) {
  const bool use_mask = variant == Ksdg3Variant::kMaskOnly
                     || variant == Ksdg3Variant::kTouchedMask;
  LongEffect::SetEffectTouchedMaskEnabled(use_mask);
  Features::SetKsdg3BenchmarkVariant(variant);
}

enum class LongEffectMaskOperation {
  DoMove,
  KsdgChanged,
  RawChanged,
  FullUpdate,
};

struct LongEffectMaskPassResult {
  NnueBenchTiming timing;
  std::uint64_t perspective_calls = 0;
  Features::Ksdg3BenchmarkStageTiming stages;
};

struct LongEffectMaskCorpusStats {
  std::uint64_t positions = 0;
  std::array<std::uint64_t, 4> dirty_num{};
  std::uint64_t reset_samples = 0;
  std::uint64_t non_reset_samples = 0;
  std::uint64_t removed_features = 0;
  std::uint64_t added_features = 0;
  std::size_t max_removed = 0;
  std::size_t max_added = 0;
};

LongEffectMaskCorpusStats CollectLongEffectMaskCorpusStats() {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  LongEffectMaskCorpusStats statistics;
  ConfigureLongEffectMaskVariant(Ksdg3Variant::kTouchedMask);

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);
      ++statistics.positions;
      const int dirty_num = pos.state()->dirtyPiece.dirty_num;
      ++statistics.dirty_num[std::min(dirty_num, 3)];

      Features::IndexList removed[COLOR_NB];
      Features::IndexList added[COLOR_NB];
      bool reset[COLOR_NB];
      RawFeatures::AppendChangedIndices(
          pos, kRefreshTriggers[0], removed, added, reset);
      for (const Color perspective : {BLACK, WHITE}) {
        if (reset[perspective])
          ++statistics.reset_samples;
        else
          ++statistics.non_reset_samples;
        statistics.removed_features += removed[perspective].size();
        statistics.added_features += added[perspective].size();
        statistics.max_removed =
            std::max(statistics.max_removed, removed[perspective].size());
        statistics.max_added =
            std::max(statistics.max_added, added[perspective].size());
      }
    }
  }
  ConfigureLongEffectMaskVariant(Ksdg3Variant::kBaseline);
  return statistics;
}

LongEffectMaskPassResult RunLongEffectMaskPass(
    const LongEffectMaskOperation operation, const Ksdg3Variant variant,
    const std::uint64_t num_games, std::uint64_t& checksum,
    const bool collect_stage_timing = false,
    LongEffect::EffectTouchedBenchmarkStats* const effect_stats = nullptr) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  LongEffectMaskPassResult result;
  ConfigureLongEffectMaskVariant(variant);
  Features::SetKsdg3BenchmarkStageTiming(nullptr);
  LongEffect::SetEffectTouchedBenchmarkStats(effect_stats);

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      const auto do_move_begin = NnueBenchClock::now();
      pos.do_move(move, states[ply]);
      const auto do_move_end = NnueBenchClock::now();

      if (collect_stage_timing)
        Features::SetKsdg3BenchmarkStageTiming(&result.stages);
      const auto begin = NnueBenchClock::now();
      if (operation == LongEffectMaskOperation::DoMove) {
        result.timing.nanoseconds += std::chrono::duration<double, std::nano>(
            do_move_end - do_move_begin).count();
      } else if (operation == LongEffectMaskOperation::KsdgChanged) {
        Features::IndexList removed[COLOR_NB];
        Features::IndexList added[COLOR_NB];
        const auto& dirty_piece = pos.state()->dirtyPiece;
        for (const Color perspective : {BLACK, WHITE}) {
          if (dirty_piece.pieceNo[0] == PIECE_NUMBER_KING + perspective)
            continue;
          Ksdg3Feature::AppendChangedIndices(
              pos, perspective, &removed[perspective], &added[perspective]);
          ++result.perspective_calls;
        }
        KeepNnueBenchObject(removed);
        KeepNnueBenchObject(added);
      } else if (operation == LongEffectMaskOperation::RawChanged) {
        Features::IndexList removed[COLOR_NB];
        Features::IndexList added[COLOR_NB];
        bool reset[COLOR_NB];
        RawFeatures::AppendChangedIndices(
            pos, kRefreshTriggers[0], removed, added, reset);
        result.perspective_calls += COLOR_NB;
        KeepNnueBenchObject(removed);
        KeepNnueBenchObject(added);
        KeepNnueBenchObject(reset);
      } else {
        if (!feature_transformer->UpdateAccumulatorIfPossible(pos)) {
          std::cout << "error: changed-mask benchmark incremental update unavailable"
                    << std::endl;
          LongEffect::SetEffectTouchedBenchmarkStats(nullptr);
          ConfigureLongEffectMaskVariant(Ksdg3Variant::kBaseline);
          return {};
        }
        result.perspective_calls += COLOR_NB;
        KeepNnueBenchObject(pos.state()->accumulator);
      }
      const auto end = NnueBenchClock::now();
      Features::SetKsdg3BenchmarkStageTiming(nullptr);
      if (operation != LongEffectMaskOperation::DoMove)
        result.timing.nanoseconds +=
            std::chrono::duration<double, std::nano>(end - begin).count();
      ++result.timing.calls;

      // Generate an order-sensitive checksum outside the measured interval.
      Features::IndexList removed[COLOR_NB];
      Features::IndexList added[COLOR_NB];
      bool reset[COLOR_NB];
      RawFeatures::AppendChangedIndices(
          pos, kRefreshTriggers[0], removed, added, reset);
      for (const Color perspective : {BLACK, WHITE}) {
        ChecksumIndexList(removed[perspective], checksum);
        ChecksumIndexList(added[perspective], checksum);
        MixNnueBenchChecksum(checksum, reset[perspective]);
      }
    }
  }

  LongEffect::SetEffectTouchedBenchmarkStats(nullptr);
  Features::SetKsdg3BenchmarkStageTiming(nullptr);
  ConfigureLongEffectMaskVariant(Ksdg3Variant::kBaseline);
  return result;
}

std::uint64_t ValidateLongEffectMaskCandidate() {
  std::uint64_t mismatches = Features::ValidateKsdg3BenchmarkTables();
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  ConfigureLongEffectMaskVariant(Ksdg3Variant::kTouchedMask);

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      pos.do_move(moves.begin()[prng.rand(moves.size())], states[ply]);

      ConfigureLongEffectMaskVariant(Ksdg3Variant::kBaseline);
      Features::IndexList baseline_active[COLOR_NB];
      Features::IndexList baseline_removed[COLOR_NB];
      Features::IndexList baseline_added[COLOR_NB];
      bool baseline_reset[COLOR_NB];
      for (const Color perspective : {BLACK, WHITE})
        Ksdg3Feature::AppendActiveIndices(
            pos, perspective, &baseline_active[perspective]);
      RawFeatures::AppendChangedIndices(
          pos, kRefreshTriggers[0], baseline_removed, baseline_added,
          baseline_reset);

      ConfigureLongEffectMaskVariant(Ksdg3Variant::kTouchedMask);
      Features::IndexList candidate_active[COLOR_NB];
      Features::IndexList candidate_removed[COLOR_NB];
      Features::IndexList candidate_added[COLOR_NB];
      bool candidate_reset[COLOR_NB];
      for (const Color perspective : {BLACK, WHITE})
        Ksdg3Feature::AppendActiveIndices(
            pos, perspective, &candidate_active[perspective]);
      RawFeatures::AppendChangedIndices(
          pos, kRefreshTriggers[0], candidate_removed, candidate_added,
          candidate_reset);
      for (const Color perspective : {BLACK, WHITE}) {
        mismatches += !EqualIndexList(
            baseline_active[perspective], candidate_active[perspective]);
        mismatches += !EqualIndexList(
            baseline_removed[perspective], candidate_removed[perspective]);
        mismatches += !EqualIndexList(
            baseline_added[perspective], candidate_added[perspective]);
        mismatches += baseline_reset[perspective] != candidate_reset[perspective];
      }
    }
  }
  ConfigureLongEffectMaskVariant(Ksdg3Variant::kBaseline);
  return mismatches;
}

void TestLongEffectChangedMaskBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: LongEffect changed-square mask]" << std::endl
            << "  seed           : " << kNnueBenchSeed << std::endl
            << "  warm-up games  : " << kNnueBenchWarmupGames << std::endl
            << "  measured games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game   : " << kNnueBenchMaxPly << std::endl
            << "  repeats        : " << repeat_count << std::endl
            << "  order          : A/B/C rotated per repeat" << std::endl;

  constexpr std::size_t kVariantCount = 3;
  constexpr std::size_t kOperationCount = 4;
  std::array<std::array<NnueBenchSamples, kOperationCount>, kVariantCount>
      samples;
  std::array<NnueBenchSamples, kVariantCount> dirty_samples;
  std::array<NnueBenchSamples, kVariantCount> neighbor_samples;
  std::array<Features::Ksdg3BenchmarkStageTiming, kVariantCount> stage_totals{};
  std::array<std::uint64_t, kVariantCount> checksums{};
  checksums.fill(UINT64_C(14695981039346656037));

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t order = 0; order < kVariantCount; ++order) {
      const std::size_t variant_index = (order + repeat) % kVariantCount;
      const auto variant = kLongEffectMaskVariants[variant_index];
      for (std::size_t operation_index = 0;
           operation_index < kOperationCount; ++operation_index) {
        const auto operation =
            static_cast<LongEffectMaskOperation>(operation_index);
        RunLongEffectMaskPass(operation, variant, kNnueBenchWarmupGames,
                              checksums[variant_index]);
        const auto pass = RunLongEffectMaskPass(
            operation, variant, kNnueBenchMeasuredGames,
            checksums[variant_index]);
        samples[variant_index][operation_index].Add(pass.timing);
        if (operation == LongEffectMaskOperation::KsdgChanged) {
          std::uint64_t stage_checksum = UINT64_C(14695981039346656037);
          const auto stage_pass = RunLongEffectMaskPass(
              operation, variant, kNnueBenchMeasuredGames,
              stage_checksum, true);
          if (stage_pass.stages.calls != 0) {
            dirty_samples[variant_index].Add({
                stage_pass.stages.calls,
                stage_pass.stages.dirty_nanoseconds});
            neighbor_samples[variant_index].Add({
                stage_pass.stages.calls,
                stage_pass.stages.neighbor_nanoseconds});
            if (repeat == 0) {
              stage_totals[variant_index].valid_neighbors =
                  stage_pass.stages.valid_neighbors;
              stage_totals[variant_index].touched_near_king =
                  stage_pass.stages.touched_near_king;
              stage_totals[variant_index].capped_changed_squares =
                  stage_pass.stages.capped_changed_squares;
            }
          }
        }
      }
    }
  }

  LongEffect::EffectTouchedBenchmarkStats effect_stats;
  std::uint64_t stats_checksum = UINT64_C(14695981039346656037);
  RunLongEffectMaskPass(LongEffectMaskOperation::DoMove,
                        Ksdg3Variant::kTouchedMask,
                        kNnueBenchMeasuredGames, stats_checksum,
                        false, &effect_stats);
  const auto corpus_stats = CollectLongEffectMaskCorpusStats();
  const std::uint64_t mismatch_count = ValidateLongEffectMaskCandidate();

  const std::array<const char*, kOperationCount> operation_names = {
      "do_move including LongEffect",
      "KSDG3 AppendChangedIndices",
      "RawFeatures AppendChangedIndices",
      "full update_accumulator",
  };
  for (std::size_t variant = 0; variant < kVariantCount; ++variant) {
    std::cout << '[' << kLongEffectMaskVariantNames[variant] << ']' << std::endl;
    for (std::size_t operation = 0; operation < kOperationCount; ++operation)
      PrintNnueBenchSamples(operation_names[operation],
                            samples[variant][operation]);
    PrintNnueBenchSamples("dirty old/new processing (instrumented)",
                          dirty_samples[variant]);
    PrintNnueBenchSamples("neighbor/effect comparison (instrumented)",
                          neighbor_samples[variant]);
    std::cout << "  valid neighbors       : "
              << stage_totals[variant].valid_neighbors << std::endl
              << "  touched near king     : "
              << stage_totals[variant].touched_near_king << std::endl
              << "  capped changed squares: "
              << stage_totals[variant].capped_changed_squares << std::endl
              << "  checksum              : 0x" << std::hex
              << checksums[variant] << std::dec << std::endl;
  }

  const auto baseline_do = SummarizeNnueBenchSamples(samples[0][0]);
  const auto mask_do = SummarizeNnueBenchSamples(samples[1][0]);
  const auto baseline_full = SummarizeNnueBenchSamples(samples[0][3]);
  const auto candidate_full = SummarizeNnueBenchSamples(samples[2][3]);
  std::cout << "[derived A/B/C deltas]" << std::endl
            << "  mask generation cost B-A (median ns/move): "
            << std::fixed << std::setprecision(1)
            << mask_do.median - baseline_do.median << std::endl
            << "  full update C-A (median ns/call)          : "
            << candidate_full.median - baseline_full.median << std::endl
            << "[LongEffect mask statistics]" << std::endl
            << "  update calls               : " << effect_stats.update_calls
            << std::endl
            << "  square write events        : "
            << effect_stats.square_write_events << std::endl
            << "  color writes               : " << effect_stats.color_writes
            << std::endl
            << "  unique touched squares     : "
            << effect_stats.unique_touched_squares << std::endl
            << "  duplicate square writes    : "
            << effect_stats.duplicate_square_writes << std::endl
            << "  final raw changed squares  : "
            << effect_stats.final_raw_changed_squares << std::endl
            << "  final capped changed squares: "
            << effect_stats.final_capped_changed_squares << std::endl
            << "[fixed-corpus feature statistics]" << std::endl
            << "  positions                  : " << corpus_stats.positions
            << std::endl
            << "  dirty_num 0 / 1 / 2 / 3+   : "
            << corpus_stats.dirty_num[0] << " / "
            << corpus_stats.dirty_num[1] << " / "
            << corpus_stats.dirty_num[2] << " / "
            << corpus_stats.dirty_num[3] << std::endl
            << "  reset / non-reset samples  : "
            << corpus_stats.reset_samples << " / "
            << corpus_stats.non_reset_samples << std::endl
            << "  removed / added features   : "
            << corpus_stats.removed_features << " / "
            << corpus_stats.added_features << std::endl
            << "  max removed / added        : "
            << corpus_stats.max_removed << " / "
            << corpus_stats.max_added << std::endl
            << "[correctness]" << std::endl
            << "  table/corpus mismatches    : " << mismatch_count << std::endl
            << "  active/removed/added order match: "
            << (mismatch_count == 0 ? "yes" : "no") << std::endl;
}

void SelectLongEffectMaskVariant(std::istream& stream) {
  int variant = -1;
  stream >> variant;
  if (variant < 0 || variant >= 3) {
    std::cout << "usage: test nnue long_effect_mask_variant <0..2>" << std::endl
              << "  0 production baseline" << std::endl
              << "  1 touched mask generation only" << std::endl
              << "  2 touched mask + KSDG3 candidate" << std::endl;
    return;
  }
  ConfigureLongEffectMaskVariant(kLongEffectMaskVariants[variant]);
  std::cout << "LongEffect changed-mask diagnostic variant: "
            << kLongEffectMaskVariantNames[variant] << std::endl;
}

void SelectKsdg3BenchmarkVariant(std::istream& stream) {
  int variant = -1;
  stream >> variant;
  if (variant < 0 || variant >= static_cast<int>(kKsdg3Variants.size())) {
    std::cout << "usage: test nnue ksdg3_variant <0..3>" << std::endl
              << "  0 baseline" << std::endl
              << "  1 effect hoist" << std::endl
              << "  2 neighbor + direction tables" << std::endl
              << "  3 combined" << std::endl;
    return;
  }
  Features::SetKsdg3BenchmarkVariant(kKsdg3Variants[variant]);
  std::cout << "KSDG3 diagnostic variant: "
            << kKsdg3VariantNames[variant] << std::endl;
}

#if defined(USE_FINNY_TABLES)
enum class FinnyBenchOperation {
  Cold,
  Warm,
  Scratch,
};

std::uint64_t CountAccumulatorMismatches(const Accumulator& left,
                                         const Accumulator& right) {
  std::uint64_t mismatches = 0;
  for (const Color perspective : {BLACK, WHITE}) {
    for (std::size_t trigger = 0; trigger < kRefreshTriggers.size(); ++trigger)
      for (IndexType index = 0; index < kTransformedFeatureDimensions; ++index)
        mismatches += left.accumulation[perspective][trigger][index]
                    != right.accumulation[perspective][trigger][index];

    const auto& left_factors = left.factors[perspective];
    const auto& right_factors = right.factors[perspective];
    for (IndexType index = 0; index < 32; ++index) {
      mismatches += left_factors.halfka.sum_v[index]
                  != right_factors.halfka.sum_v[index];
      mismatches += left_factors.halfka.sum_v2[index]
                  != right_factors.halfka.sum_v2[index];
      mismatches += left_factors.ksdg.sum_v[index]
                  != right_factors.ksdg.sum_v[index];
      mismatches += left_factors.ksdg.sum_v2[index]
                  != right_factors.ksdg.sum_v2[index];
    }
  }
  return mismatches;
}

NnueBenchTiming RunFinnyBenchmarkPass(
    const FinnyBenchOperation operation, const std::uint64_t num_games,
    FeatureTransformer::BenchmarkFinnyStatistics* const statistics,
    std::uint64_t& checksum) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  NnueBenchTiming timing;

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      // Cache invalidation is deliberately outside the timed interval. Cold
      // measures one uninitialized-entry refresh, not clearing all 162 entries.
      if (operation == FinnyBenchOperation::Cold)
        feature_transformer->BenchmarkResetFinnyCache();

      const auto begin = NnueBenchClock::now();
      if (operation == FinnyBenchOperation::Scratch)
        feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
      else
        feature_transformer->BenchmarkRefreshAccumulatorWithFinny(pos, statistics);
      const auto end = NnueBenchClock::now();

      timing.nanoseconds +=
          std::chrono::duration<double, std::nano>(end - begin).count();
      ++timing.calls;
      ChecksumAccumulator(pos, checksum);
    }
  }
  return timing;
}

std::uint64_t ValidateFinnyAgainstScratch(const bool cold_each_position,
                                          std::uint64_t& finny_checksum,
                                          std::uint64_t& scratch_checksum) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  std::uint64_t mismatches = 0;
  feature_transformer->BenchmarkResetFinnyCache();

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      if (cold_each_position)
        feature_transformer->BenchmarkResetFinnyCache();
      feature_transformer->BenchmarkRefreshAccumulatorWithFinny(pos, nullptr);
      const Accumulator finny = pos.state()->accumulator;
      ChecksumAccumulator(pos, finny_checksum);

      feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
      const Accumulator scratch = pos.state()->accumulator;
      ChecksumAccumulator(pos, scratch_checksum);
      mismatches += CountAccumulatorMismatches(finny, scratch);
    }
  }
  return mismatches;
}

void PrintFinnyStatistics(
    const char* const name,
    const FeatureTransformer::BenchmarkFinnyStatistics& statistics) {
  std::cout << name << std::endl
            << "  cache hits       : " << statistics.hits << std::endl
            << "  cache misses     : " << statistics.misses << std::endl
            << "  removed features : " << statistics.removed_features << std::endl
            << "  added features   : " << statistics.added_features << std::endl;
}

void TestFinnyBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Finny / scratch refresh comparison]" << std::endl
            << "  seed              : " << kNnueBenchSeed << std::endl
            << "  warm-up games     : " << kNnueBenchWarmupGames << std::endl
            << "  measured games    : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game      : " << kNnueBenchMaxPly << std::endl
            << "  repeats           : " << repeat_count << std::endl
            << "  order             : cold/warm/scratch rotated per repeat" << std::endl
            << "  Finny entries     : "
            << FeatureTransformer::BenchmarkFinnyEntryCount() << std::endl
            << "  FinnyEntry bytes  : "
            << FeatureTransformer::BenchmarkFinnyEntrySize() << std::endl
            << "  FinnyCache bytes/thread: "
            << FeatureTransformer::BenchmarkFinnyCacheSize() << std::endl;

  NnueBenchSamples samples[3];
  std::uint64_t timing_checksums[3] = {
      UINT64_C(14695981039346656037), UINT64_C(14695981039346656037),
      UINT64_C(14695981039346656037)};
  FeatureTransformer::BenchmarkFinnyStatistics cold_statistics;
  FeatureTransformer::BenchmarkFinnyStatistics warm_statistics;
  const FinnyBenchOperation operations[3] = {
      FinnyBenchOperation::Cold, FinnyBenchOperation::Warm,
      FinnyBenchOperation::Scratch};

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      const std::size_t operation_index = (repeat + offset) % 3;
      const auto operation = operations[operation_index];
      std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
      feature_transformer->BenchmarkResetFinnyCache();
      RunFinnyBenchmarkPass(operation, kNnueBenchWarmupGames, nullptr,
                            warmup_checksum);

      // A warm measurement starts after an untimed production-shaped pass.
      if (operation == FinnyBenchOperation::Warm) {
        feature_transformer->BenchmarkResetFinnyCache();
        RunFinnyBenchmarkPass(operation, kNnueBenchMeasuredGames, nullptr,
                              warmup_checksum);
      }

      auto* statistics = repeat == 0
          ? (operation == FinnyBenchOperation::Cold
                 ? &cold_statistics
                 : operation == FinnyBenchOperation::Warm
                       ? &warm_statistics
                       : nullptr)
          : nullptr;
      samples[operation_index].Add(RunFinnyBenchmarkPass(
          operation, kNnueBenchMeasuredGames, statistics,
          timing_checksums[operation_index]));
    }
  }

  PrintNnueBenchSamples("cold Finny", samples[0]);
  PrintNnueBenchSamples("warm Finny", samples[1]);
  PrintNnueBenchSamples("true scratch refresh", samples[2]);
  PrintFinnyStatistics("[cold Finny statistics]", cold_statistics);
  PrintFinnyStatistics("[warm Finny statistics]", warm_statistics);

  std::uint64_t cold_checksum = UINT64_C(14695981039346656037);
  std::uint64_t cold_scratch_checksum = UINT64_C(14695981039346656037);
  const std::uint64_t cold_mismatches = ValidateFinnyAgainstScratch(
      true, cold_checksum, cold_scratch_checksum);
  std::uint64_t warm_checksum = UINT64_C(14695981039346656037);
  std::uint64_t warm_scratch_checksum = UINT64_C(14695981039346656037);
  const std::uint64_t warm_mismatches = ValidateFinnyAgainstScratch(
      false, warm_checksum, warm_scratch_checksum);

  std::cout << "[correctness]" << std::endl
            << "  cold Finny checksum   : 0x" << std::hex << cold_checksum
            << std::endl
            << "  cold scratch checksum : 0x" << cold_scratch_checksum
            << std::endl
            << "  cold mismatch count   : " << std::dec << cold_mismatches
            << std::endl
            << "  warm Finny checksum   : 0x" << std::hex << warm_checksum
            << std::endl
            << "  warm scratch checksum : 0x" << warm_scratch_checksum
            << std::endl
            << "  warm mismatch count   : " << std::dec << warm_mismatches
            << std::endl
            << "[timing checksums]" << std::endl
            << "  cold   : 0x" << std::hex << timing_checksums[0] << std::endl
            << "  warm   : 0x" << timing_checksums[1] << std::endl
            << "  scratch: 0x" << timing_checksums[2] << std::dec << std::endl
            << "  timing checksum match: "
            << (timing_checksums[0] == timing_checksums[1]
                    && timing_checksums[0] == timing_checksums[2]
                ? "yes" : "NO") << std::endl;
}

enum class FinnyFmBenchOperation {
  Phase1,
  Phase3,
};

void ResetFinnyFmBenchmarkCache(const FinnyFmBenchOperation operation) {
  if (operation == FinnyFmBenchOperation::Phase1)
    feature_transformer->BenchmarkResetPhase1FinnyCache();
  else
    feature_transformer->BenchmarkResetPhase3FinnyCache();
}

void RefreshWithFinnyFmBenchmarkMode(
    const Position& pos, const FinnyFmBenchOperation operation,
    FeatureTransformer::BenchmarkFinnyStatistics* const statistics) {
  if (operation == FinnyFmBenchOperation::Phase1)
    feature_transformer->BenchmarkRefreshAccumulatorWithPhase1Finny(pos,
                                                                     statistics);
  else
    feature_transformer->BenchmarkRefreshAccumulatorWithPhase3Finny(pos,
                                                                     statistics);
}

NnueBenchTiming RunFinnyFmBenchmarkPass(
    const FinnyFmBenchOperation operation, const std::uint64_t num_games,
    FeatureTransformer::BenchmarkFinnyStatistics* const statistics,
    std::uint64_t& checksum, const bool timed) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  NnueBenchTiming timing;

  for (std::uint64_t game = 0; game < num_games; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      if (timed) {
        const auto begin = NnueBenchClock::now();
        RefreshWithFinnyFmBenchmarkMode(pos, operation, statistics);
        const auto end = NnueBenchClock::now();
        timing.nanoseconds +=
            std::chrono::duration<double, std::nano>(end - begin).count();
      } else {
        RefreshWithFinnyFmBenchmarkMode(pos, operation, statistics);
      }
      ++timing.calls;
      ChecksumAccumulator(pos, checksum);
    }
  }
  return timing;
}

std::uint64_t ValidateFinnyFmModeAgainstScratch(
    const FinnyFmBenchOperation operation, std::uint64_t& finny_checksum,
    std::uint64_t& scratch_checksum) {
  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  std::uint64_t mismatches = 0;
  ResetFinnyFmBenchmarkCache(operation);

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;
      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      RefreshWithFinnyFmBenchmarkMode(pos, operation, nullptr);
      const Accumulator finny = pos.state()->accumulator;
      ChecksumAccumulator(pos, finny_checksum);

      feature_transformer->BenchmarkRefreshAccumulatorFromScratch(pos);
      const Accumulator scratch = pos.state()->accumulator;
      ChecksumAccumulator(pos, scratch_checksum);
      mismatches += CountAccumulatorMismatches(finny, scratch);
    }
  }
  return mismatches;
}

void PrintFinnyFmStatistics(
    const char* const name,
    const FeatureTransformer::BenchmarkFinnyStatistics& statistics) {
  const double mean_removed = statistics.hits == 0
      ? 0.0 : static_cast<double>(statistics.removed_features) / statistics.hits;
  const double mean_added = statistics.hits == 0
      ? 0.0 : static_cast<double>(statistics.added_features) / statistics.hits;
  std::cout << name << std::endl
            << "  cache hits       : " << statistics.hits << std::endl
            << "  cache misses     : " << statistics.misses << std::endl
            << "  mean removed/hit : " << std::fixed << std::setprecision(3)
            << mean_removed << std::endl
            << "  max removed      : " << statistics.max_removed_features << std::endl
            << "  mean added/hit   : " << mean_added << std::endl
            << "  max added        : " << statistics.max_added_features << std::endl;
}

void TestFinnyFmBenchmarkCompare(const std::uint64_t repeat_count) {
  constexpr std::size_t kOperationCount = 2;
  const FinnyFmBenchOperation operations[kOperationCount] = {
      FinnyFmBenchOperation::Phase1, FinnyFmBenchOperation::Phase3};
  const std::size_t entry_count =
      FeatureTransformer::BenchmarkFinnyEntryCount();
  const std::size_t phase3_entry_bytes =
      FeatureTransformer::BenchmarkFinnyEntrySize();
  const std::size_t phase3_cache_bytes =
      FeatureTransformer::BenchmarkFinnyCacheSize();
  const std::size_t fm_bytes =
      FeatureTransformer::BenchmarkFinnyFactorBytesPerEntry();
  const std::size_t phase1_entry_bytes =
      FeatureTransformer::BenchmarkPhase1FinnyEntrySize();
  const std::size_t phase1_cache_bytes =
      FeatureTransformer::BenchmarkPhase1FinnyCacheSize();

  std::cout << "[NNUE benchmark: Finny FM Phase 1 / Phase 3]" << std::endl
            << "  seed              : " << kNnueBenchSeed << std::endl
            << "  warm-up games     : " << kNnueBenchWarmupGames << std::endl
            << "  measured games    : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game      : " << kNnueBenchMaxPly << std::endl
            << "  repeats           : " << repeat_count << std::endl
            << "  order             : even=Phase1,Phase3 odd=Phase3,Phase1"
            << std::endl
            << "  Finny entries     : " << entry_count << std::endl
            << "  Phase 1 entry bytes: "
            << phase1_entry_bytes << std::endl
            << "  Phase 3 entry bytes: " << phase3_entry_bytes << std::endl
            << "  entry increase     : " << fm_bytes << std::endl
            << "  Phase 1 cache bytes/thread: "
            << phase1_cache_bytes << std::endl
            << "  Phase 3 cache bytes/thread: " << phase3_cache_bytes << std::endl
            << "  cache increase/thread: " << phase3_cache_bytes - phase1_cache_bytes
            << std::endl;

  NnueBenchSamples samples[kOperationCount];
  std::uint64_t timing_checksums[kOperationCount] = {
      UINT64_C(14695981039346656037), UINT64_C(14695981039346656037)};
  FeatureTransformer::BenchmarkFinnyStatistics statistics[kOperationCount];

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t offset = 0; offset < kOperationCount; ++offset) {
      const std::size_t operation_index = (repeat + offset) % kOperationCount;
      const auto operation = operations[operation_index];
      std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);

      ResetFinnyFmBenchmarkCache(operation);
      RunFinnyFmBenchmarkPass(operation, kNnueBenchWarmupGames, nullptr,
                              warmup_checksum, false);

      // Populate the entire measured corpus once so the timed pass measures
      // warm entries for both implementations using identical positions.
      ResetFinnyFmBenchmarkCache(operation);
      RunFinnyFmBenchmarkPass(operation, kNnueBenchMeasuredGames, nullptr,
                              warmup_checksum, false);

      auto* const sample_statistics = repeat == 0
          ? &statistics[operation_index] : nullptr;
      samples[operation_index].Add(RunFinnyFmBenchmarkPass(
          operation, kNnueBenchMeasuredGames, sample_statistics,
          timing_checksums[operation_index], true));
    }
  }

  PrintNnueBenchSamples("A. Phase 1: Main Finny + FM scratch", samples[0]);
  PrintNnueBenchSamples("B. Phase 3: Main + FM Finny", samples[1]);
  const auto phase1_summary = SummarizeNnueBenchSamples(samples[0]);
  const auto phase3_summary = SummarizeNnueBenchSamples(samples[1]);
  const double difference = phase3_summary.median - phase1_summary.median;
  const double improvement = phase1_summary.median == 0.0
      ? 0.0 : -difference / phase1_summary.median * 100.0;
  std::cout << "  improvement ns/call : " << std::fixed << std::setprecision(1)
            << -difference << std::endl
            << "  improvement         : " << std::setprecision(2)
            << improvement << "%" << std::endl;
  PrintFinnyFmStatistics("[Phase 1 warm statistics]", statistics[0]);
  PrintFinnyFmStatistics("[Phase 3 warm statistics]", statistics[1]);

  std::uint64_t phase1_checksum = UINT64_C(14695981039346656037);
  std::uint64_t phase1_scratch_checksum = UINT64_C(14695981039346656037);
  const std::uint64_t phase1_mismatches = ValidateFinnyFmModeAgainstScratch(
      FinnyFmBenchOperation::Phase1, phase1_checksum,
      phase1_scratch_checksum);
  std::uint64_t phase3_checksum = UINT64_C(14695981039346656037);
  std::uint64_t phase3_scratch_checksum = UINT64_C(14695981039346656037);
  const std::uint64_t phase3_mismatches = ValidateFinnyFmModeAgainstScratch(
      FinnyFmBenchOperation::Phase3, phase3_checksum,
      phase3_scratch_checksum);

  std::cout << "[correctness against true scratch]" << std::endl
            << "  Phase 1 checksum : 0x" << std::hex << phase1_checksum
            << std::endl
            << "  scratch checksum : 0x" << phase1_scratch_checksum
            << std::dec << std::endl
            << "  Phase 1 mismatch count: " << phase1_mismatches << std::endl
            << "  Phase 3 checksum : 0x" << std::hex << phase3_checksum
            << std::endl
            << "  scratch checksum : 0x" << phase3_scratch_checksum
            << std::dec << std::endl
            << "  Phase 3 mismatch count: " << phase3_mismatches << std::endl
            << "[timing checksums]" << std::endl
            << "  Phase 1: 0x" << std::hex << timing_checksums[0] << std::endl
            << "  Phase 3: 0x" << timing_checksums[1] << std::dec << std::endl
            << "  timing checksum match: "
            << (timing_checksums[0] == timing_checksums[1] ? "yes" : "NO")
            << std::endl;
}
#endif

enum class FtTransformStageOperation {
  MainPairAndPacking,
  FmInteractions,
  FmScalingAndPacking,
  FullTransform,
};

constexpr std::array<FtTransformStageOperation, 4>
    kFtTransformStageOperations = {
        FtTransformStageOperation::MainPairAndPacking,
        FtTransformStageOperation::FmInteractions,
        FtTransformStageOperation::FmScalingAndPacking,
        FtTransformStageOperation::FullTransform,
    };

constexpr std::array<const char*, kFtTransformStageOperations.size()>
    kFtTransformStageNames = {
        "A. Main FT + PairWeight 3-way blend + Main packing",
        "B. FM interaction generation",
        "C. FM diff/abs scaling + clamp + output",
        "D. full Transform (precomputed accumulator)",
    };

struct FtTransformBenchCase {
  std::unique_ptr<StateInfo> state;
  std::unique_ptr<Position> position;
  int material_bucket = 0;
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> expected_main{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> expected_diff{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> expected_abs{};
  FeatureTransformer::BenchmarkFmInteractions interactions{};
};

std::vector<FtTransformBenchCase> MakeFtTransformStageBenchCorpus() {
  Position generator_position;
  StateInfo generator_root;
  std::vector<StateInfo> generator_states(kNnueBenchMaxPly);
  PRNG prng(kNnueBenchSeed);
  std::vector<FtTransformBenchCase> corpus;
  corpus.reserve(kNnueBenchMeasuredGames * kNnueBenchMaxPly);

  for (std::uint64_t game = 0; game < kNnueBenchMeasuredGames; ++game) {
    generator_position.set_hirate(&generator_root);
    for (int ply = 0; ply < kNnueBenchMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(generator_position);
      if (moves.size() == 0)
        break;

      const Move move = moves.begin()[prng.rand(moves.size())];
      generator_position.do_move(move, generator_states[ply]);

      FtTransformBenchCase sample;
      sample.position = std::make_unique<Position>();
      sample.state = std::make_unique<StateInfo>();
      sample.position->set(generator_position.sfen(), sample.state.get());
      sample.material_bucket = NnueBenchMaterialBucket(*sample.position);
      feature_transformer->Transform(
          *sample.position, sample.expected_main.data(),
          sample.expected_diff.data(), sample.expected_abs.data(), false,
          sample.material_bucket);
      feature_transformer->BenchmarkTransformFmInteractionsOnly(
          *sample.position, sample.interactions);
      corpus.emplace_back(std::move(sample));
    }
  }
  return corpus;
}

template<FtTransformStageOperation Operation>
NnueBenchTiming MeasureFtTransformStageCorpus(
    const std::vector<FtTransformBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> main_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> diff_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> abs_output{};
  FeatureTransformer::BenchmarkFmInteractions interactions{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    if constexpr (Operation == FtTransformStageOperation::MainPairAndPacking) {
      feature_transformer->BenchmarkTransformMain(
          *sample.position, main_output.data(), sample.material_bucket);
      KeepNnueBenchObject(main_output);
      MixNnueBenchChecksum(
          checksum, main_output[static_cast<std::size_t>(timing.calls)
                                % main_output.size()]);
    } else if constexpr (Operation ==
                         FtTransformStageOperation::FmInteractions) {
      feature_transformer->BenchmarkTransformFmInteractionsOnly(
          *sample.position, interactions);
      KeepNnueBenchObject(interactions);
      const std::size_t flat_index =
          static_cast<std::size_t>(timing.calls) % (4 * 32);
      MixNnueBenchChecksum(
          checksum, interactions.values[flat_index / 32][flat_index % 32]);
    } else if constexpr (Operation ==
                         FtTransformStageOperation::FmScalingAndPacking) {
      feature_transformer->BenchmarkTransformFmOutputsOnly(
          *sample.position, sample.interactions, diff_output.data(),
          abs_output.data());
      KeepNnueBenchObject(diff_output);
      KeepNnueBenchObject(abs_output);
      const std::size_t index =
          static_cast<std::size_t>(timing.calls) % diff_output.size();
      MixNnueBenchChecksum(checksum, diff_output[index]);
      MixNnueBenchChecksum(checksum, abs_output[index]);
    } else {
      feature_transformer->Transform(
          *sample.position, main_output.data(), diff_output.data(),
          abs_output.data(), false, sample.material_bucket);
      KeepNnueBenchObject(main_output);
      KeepNnueBenchObject(diff_output);
      KeepNnueBenchObject(abs_output);
      const std::size_t main_index =
          static_cast<std::size_t>(timing.calls) % main_output.size();
      const std::size_t fm_index =
          static_cast<std::size_t>(timing.calls) % diff_output.size();
      MixNnueBenchChecksum(checksum, main_output[main_index]);
      MixNnueBenchChecksum(checksum, diff_output[fm_index]);
      MixNnueBenchChecksum(checksum, abs_output[fm_index]);
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

NnueBenchTiming MeasureFtTransformStageByIndex(
    const std::vector<FtTransformBenchCase>& corpus,
    const std::size_t stage_index, std::uint64_t& checksum) {
  switch (kFtTransformStageOperations[stage_index]) {
    case FtTransformStageOperation::MainPairAndPacking:
      return MeasureFtTransformStageCorpus<
          FtTransformStageOperation::MainPairAndPacking>(corpus, checksum);
    case FtTransformStageOperation::FmInteractions:
      return MeasureFtTransformStageCorpus<
          FtTransformStageOperation::FmInteractions>(corpus, checksum);
    case FtTransformStageOperation::FmScalingAndPacking:
      return MeasureFtTransformStageCorpus<
          FtTransformStageOperation::FmScalingAndPacking>(corpus, checksum);
    case FtTransformStageOperation::FullTransform:
      return MeasureFtTransformStageCorpus<
          FtTransformStageOperation::FullTransform>(corpus, checksum);
  }
  return {};
}

NnueBenchTiming MeasureFtTransformStageAfterWarmup(
    const std::vector<FtTransformBenchCase>& corpus,
    const std::size_t stage_index, std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureFtTransformStageByIndex(corpus, stage_index, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureFtTransformStageByIndex(corpus, stage_index, checksum);
}

void TestFeatureTransformerStagesBenchmark(
    const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: FeatureTransformer Transform stages]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  stage order  : rotated by one stage per repeat" << std::endl
            << "  corpus build : real positions and precomputed accumulators..."
            << std::flush;

  const auto corpus = MakeFtTransformStageBenchCorpus();
  std::cout << "done (" << corpus.size() << ")" << std::endl;
  if (corpus.empty()) {
    std::cout << "error: NNUE FT Transform stage benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  std::array<NnueBenchSamples, kFtTransformStageOperations.size()> samples;
  std::array<std::uint64_t, kFtTransformStageOperations.size()>
      timing_checksums;
  timing_checksums.fill(UINT64_C(14695981039346656037));

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    const std::size_t rotation = repeat % kFtTransformStageOperations.size();
    for (std::size_t offset = 0;
         offset < kFtTransformStageOperations.size(); ++offset) {
      const std::size_t stage =
          (rotation + offset) % kFtTransformStageOperations.size();
      samples[stage].Add(MeasureFtTransformStageAfterWarmup(
          corpus, stage, timing_checksums[stage]));
    }
  }

  std::uint64_t captured_checksum = UINT64_C(14695981039346656037);
  std::uint64_t reconstructed_checksum = UINT64_C(14695981039346656037);
  std::uint64_t mismatch_count = 0;
  std::size_t first_mismatch_sample = 0;
  const char* first_mismatch_target = nullptr;
  std::size_t first_mismatch_element = 0;
  int first_captured = 0;
  int first_recomputed = 0;

  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> main_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> diff_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> abs_output{};

  auto compare_range = [&](const auto& captured, const auto& recomputed,
                           const char* target, std::size_t sample_index) {
    for (std::size_t element = 0; element < captured.size(); ++element) {
      MixNnueBenchChecksum(captured_checksum, captured[element]);
      MixNnueBenchChecksum(reconstructed_checksum, recomputed[element]);
      if (captured[element] != recomputed[element]) {
        if (mismatch_count == 0) {
          first_mismatch_sample = sample_index;
          first_mismatch_target = target;
          first_mismatch_element = element;
          first_captured = captured[element];
          first_recomputed = recomputed[element];
        }
        ++mismatch_count;
      }
    }
  };

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    feature_transformer->BenchmarkTransformReconstructed(
        *sample.position, main_output.data(), diff_output.data(),
        abs_output.data(), sample.material_bucket);
    compare_range(sample.expected_main, main_output, "main", sample_index);
    compare_range(sample.expected_diff, diff_output, "diff", sample_index);
    compare_range(sample.expected_abs, abs_output, "abs", sample_index);
  }

  for (std::size_t stage = 0;
       stage < kFtTransformStageOperations.size(); ++stage) {
    PrintNnueBenchSamples(kFtTransformStageNames[stage], samples[stage]);
    std::cout << "  timing checksum : 0x" << std::hex
              << timing_checksums[stage] << std::dec << std::endl;
  }

  const auto main_summary = SummarizeNnueBenchSamples(samples[0]);
  const auto interaction_summary = SummarizeNnueBenchSamples(samples[1]);
  const auto scaling_summary = SummarizeNnueBenchSamples(samples[2]);
  const auto full_summary = SummarizeNnueBenchSamples(samples[3]);
  const double summed_stage_medians = main_summary.median
      + interaction_summary.median + scaling_summary.median;
  std::cout << "summed isolated stage medians : " << std::fixed
            << std::setprecision(1) << summed_stage_medians << " ns/call"
            << std::endl
            << "full Transform median         : " << full_summary.median
            << " ns/call" << std::endl
            << "captured checksum             : 0x" << std::hex
            << captured_checksum << std::endl
            << "reconstructed checksum        : 0x"
            << reconstructed_checksum << std::dec << std::endl
            << "full reconstruction match     : "
            << (mismatch_count == 0 ? "yes" : "NO") << std::endl
            << "mismatch count                : " << mismatch_count
            << std::endl;
  if (mismatch_count != 0) {
    std::cout << "first mismatch sample         : "
              << first_mismatch_sample << std::endl
              << "first mismatch target         : "
              << first_mismatch_target << std::endl
              << "first mismatch element        : "
              << first_mismatch_element << std::endl
              << "captured value                : " << first_captured
              << std::endl
              << "recomputed value              : " << first_recomputed
              << std::endl;
  }
  std::cout << "note: isolated stage medians need not sum to the full median;"
            << std::endl
            << "      stage outputs are materialized and cache locality differs."
            << std::endl;
}

#if defined(USE_AVX2)

enum class FtFmScalingImplementation {
  CurrentScalar,
  HybridAvx2,
};

template<FtFmScalingImplementation Implementation, bool FullTransform>
NnueBenchTiming MeasureFtFmScalingCorpus(
    const std::vector<FtTransformBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> main_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> diff_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> abs_output{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    if constexpr (FullTransform) {
      if constexpr (Implementation ==
                    FtFmScalingImplementation::CurrentScalar) {
        feature_transformer->BenchmarkTransformReconstructedScalarFm(
            *sample.position, main_output.data(), diff_output.data(),
            abs_output.data(), sample.material_bucket);
      } else {
        feature_transformer->BenchmarkTransformReconstructed(
            *sample.position, main_output.data(), diff_output.data(),
            abs_output.data(), sample.material_bucket);
      }
      KeepNnueBenchObject(main_output);
    } else {
      if constexpr (Implementation ==
                    FtFmScalingImplementation::CurrentScalar) {
        feature_transformer->BenchmarkTransformFmOutputsScalarOnly(
            *sample.position, sample.interactions, diff_output.data(),
            abs_output.data());
      } else {
        feature_transformer->BenchmarkTransformFmOutputsHybridOnly(
            *sample.position, sample.interactions, diff_output.data(),
            abs_output.data());
      }
    }
    KeepNnueBenchObject(diff_output);
    KeepNnueBenchObject(abs_output);
    const std::size_t main_index =
        static_cast<std::size_t>(timing.calls) % main_output.size();
    const std::size_t fm_index =
        static_cast<std::size_t>(timing.calls) % diff_output.size();
    if constexpr (FullTransform)
      MixNnueBenchChecksum(checksum, main_output[main_index]);
    MixNnueBenchChecksum(checksum, diff_output[fm_index]);
    MixNnueBenchChecksum(checksum, abs_output[fm_index]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<FtFmScalingImplementation Implementation, bool FullTransform>
NnueBenchTiming MeasureFtFmScalingCorpusAfterWarmup(
    const std::vector<FtTransformBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureFtFmScalingCorpus<Implementation, FullTransform>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureFtFmScalingCorpus<Implementation, FullTransform>(
      corpus, checksum);
}

void PrintFtFmScalingComparison(const char* const title,
                                const NnueBenchSamples& scalar_samples,
                                const NnueBenchSamples& hybrid_samples) {
  const auto scalar = SummarizeNnueBenchSamples(scalar_samples);
  const auto hybrid = SummarizeNnueBenchSamples(hybrid_samples);
  const double saved = scalar.median - hybrid.median;
  const double improvement = scalar.median == 0.0
      ? 0.0
      : saved * 100.0 / scalar.median;
  std::cout << title << std::endl;
  PrintNnueBenchSamples("  A. current scalar", scalar_samples);
  PrintNnueBenchSamples("  B. hybrid AVX2", hybrid_samples);
  std::cout << "  improvement ns/call : " << std::fixed
            << std::setprecision(1) << saved << std::endl
            << "  improvement         : " << std::setprecision(2)
            << improvement << "%" << std::endl;
}

void TestFeatureTransformerFmScalingCompare(
    const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: FM scaling scalar / hybrid AVX2]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  candidate    : scalar ih/ik + AVX2 diff/abs sh/sk"
            << std::endl
            << "  order        : even=scalar,hybrid odd=hybrid,scalar"
            << std::endl
            << "  corpus build : real positions and precomputed accumulators..."
            << std::flush;

  const auto corpus = MakeFtTransformStageBenchCorpus();
  std::cout << "done (" << corpus.size() << ")" << std::endl;
  if (corpus.empty()) {
    std::cout << "error: NNUE FM scaling benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  NnueBenchSamples scalar_scaling;
  NnueBenchSamples hybrid_scaling;
  NnueBenchSamples scalar_full;
  NnueBenchSamples hybrid_full;
  std::uint64_t scalar_scaling_timing = UINT64_C(14695981039346656037);
  std::uint64_t hybrid_scaling_timing = UINT64_C(14695981039346656037);
  std::uint64_t scalar_full_timing = UINT64_C(14695981039346656037);
  std::uint64_t hybrid_full_timing = UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      scalar_scaling.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::CurrentScalar, false>(
              corpus, scalar_scaling_timing));
      hybrid_scaling.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::HybridAvx2, false>(
              corpus, hybrid_scaling_timing));
      scalar_full.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::CurrentScalar, true>(
              corpus, scalar_full_timing));
      hybrid_full.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::HybridAvx2, true>(
              corpus, hybrid_full_timing));
    } else {
      hybrid_scaling.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::HybridAvx2, false>(
              corpus, hybrid_scaling_timing));
      scalar_scaling.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::CurrentScalar, false>(
              corpus, scalar_scaling_timing));
      hybrid_full.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::HybridAvx2, true>(
              corpus, hybrid_full_timing));
      scalar_full.Add(MeasureFtFmScalingCorpusAfterWarmup<
          FtFmScalingImplementation::CurrentScalar, true>(
              corpus, scalar_full_timing));
    }
  }

  std::uint64_t scalar_checksum = UINT64_C(14695981039346656037);
  std::uint64_t hybrid_checksum = UINT64_C(14695981039346656037);
  std::uint64_t mismatch_count = 0;
  std::size_t first_sample = 0;
  const char* first_target = nullptr;
  std::size_t first_element = 0;
  int first_scalar = 0;
  int first_hybrid = 0;
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> scalar_diff{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> scalar_abs{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> hybrid_diff{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> hybrid_abs{};

  auto compare = [&](const auto& scalar_values, const auto& hybrid_values,
                     const char* target, const std::size_t sample_index) {
    for (std::size_t element = 0; element < scalar_values.size(); ++element) {
      MixNnueBenchChecksum(scalar_checksum, scalar_values[element]);
      MixNnueBenchChecksum(hybrid_checksum, hybrid_values[element]);
      if (scalar_values[element] != hybrid_values[element]) {
        if (mismatch_count == 0) {
          first_sample = sample_index;
          first_target = target;
          first_element = element;
          first_scalar = scalar_values[element];
          first_hybrid = hybrid_values[element];
        }
        ++mismatch_count;
      }
    }
  };

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    feature_transformer->BenchmarkTransformFmOutputsScalarOnly(
        *sample.position, sample.interactions, scalar_diff.data(),
        scalar_abs.data());
    feature_transformer->BenchmarkTransformFmOutputsHybridOnly(
        *sample.position, sample.interactions, hybrid_diff.data(),
        hybrid_abs.data());
    compare(scalar_diff, hybrid_diff, "diff_output", sample_index);
    compare(scalar_abs, hybrid_abs, "abs_output", sample_index);
  }

  PrintFtFmScalingComparison("C. FM scaling + clamp + output",
                             scalar_scaling, hybrid_scaling);
  PrintFtFmScalingComparison("full reconstructed Transform",
                             scalar_full, hybrid_full);
  std::cout << "[correctness]" << std::endl
            << "  scalar checksum : 0x" << std::hex << scalar_checksum
            << std::endl
            << "  hybrid checksum : 0x" << hybrid_checksum << std::dec
            << std::endl
            << "  checksum match  : "
            << (scalar_checksum == hybrid_checksum ? "yes" : "NO")
            << std::endl
            << "  mismatch count : " << mismatch_count << std::endl;
  if (mismatch_count != 0) {
    std::cout << "  first mismatch sample : " << first_sample << std::endl
              << "  first mismatch target : " << first_target << std::endl
              << "  first mismatch element: " << first_element << std::endl
              << "  scalar value          : " << first_scalar << std::endl
              << "  hybrid value          : " << first_hybrid << std::endl;
  }
  std::cout << "[timing checksums]" << std::endl
            << "  C scalar : 0x" << std::hex << scalar_scaling_timing
            << std::endl
            << "  C hybrid : 0x" << hybrid_scaling_timing << std::endl
            << "  full scalar: 0x" << scalar_full_timing << std::endl
            << "  full hybrid: 0x" << hybrid_full_timing << std::dec
            << std::endl
            << "  C timing checksum match   : "
            << (scalar_scaling_timing == hybrid_scaling_timing ? "yes"
                                                                : "NO")
            << std::endl
            << "  full timing checksum match: "
            << (scalar_full_timing == hybrid_full_timing ? "yes" : "NO")
            << std::endl;
}

#endif  // defined(USE_AVX2)

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
  PhaseInputAssembly,
  PhaseProjection,
  PhaseSigmoid,
  PhaseChannelScales,
  FcDiff,
  FcAbs,
  DiffRmsNorm,
  DiffQuantize,
  AbsSigmoidGate,
  AbsGateQuantize,
  AbsSquared,
  MainFc0,
  MainGateSigmoid,
  MainGateApply,
  MainGateClamp,
  MainSqrClippedRelu,
  MainClippedRelu,
  LcaFmInput,
  LcaQuery,
  LcaKey,
  LcaValue,
  LcaDotAndLogit,
  LcaAttentionScore,
  LcaValueClampAndCorrection,
  LcaFinalAddAndQuantize,
  Cross,
  L2Assembly,
  Fc1,
  Ac1,
  Fc2,
  Blend,
};

constexpr std::array<NetworkStageBenchOperation, 31>
    kNetworkStageBenchOperations = {
        NetworkStageBenchOperation::PhaseInputAssembly,
        NetworkStageBenchOperation::PhaseProjection,
        NetworkStageBenchOperation::PhaseSigmoid,
        NetworkStageBenchOperation::PhaseChannelScales,
        NetworkStageBenchOperation::FcDiff,
        NetworkStageBenchOperation::FcAbs,
        NetworkStageBenchOperation::DiffRmsNorm,
        NetworkStageBenchOperation::DiffQuantize,
        NetworkStageBenchOperation::AbsSigmoidGate,
        NetworkStageBenchOperation::AbsGateQuantize,
        NetworkStageBenchOperation::AbsSquared,
        NetworkStageBenchOperation::MainFc0,
        NetworkStageBenchOperation::MainGateSigmoid,
        NetworkStageBenchOperation::MainGateApply,
        NetworkStageBenchOperation::MainGateClamp,
        NetworkStageBenchOperation::MainSqrClippedRelu,
        NetworkStageBenchOperation::MainClippedRelu,
        NetworkStageBenchOperation::LcaFmInput,
        NetworkStageBenchOperation::LcaQuery,
        NetworkStageBenchOperation::LcaKey,
        NetworkStageBenchOperation::LcaValue,
        NetworkStageBenchOperation::LcaDotAndLogit,
        NetworkStageBenchOperation::LcaAttentionScore,
        NetworkStageBenchOperation::LcaValueClampAndCorrection,
        NetworkStageBenchOperation::LcaFinalAddAndQuantize,
        NetworkStageBenchOperation::Cross,
        NetworkStageBenchOperation::L2Assembly,
        NetworkStageBenchOperation::Fc1,
        NetworkStageBenchOperation::Ac1,
        NetworkStageBenchOperation::Fc2,
        NetworkStageBenchOperation::Blend};

constexpr std::array<const char*, kNetworkStageBenchOperations.size()>
    kNetworkStageBenchNames = {
        "Phase input assembly",
        "Phase phase_proj affine",
        "Phase sigmoid / phase value",
        "Phase final channel scale coefficients",
        "FM fc_diff",
        "FM fc_abs",
        "FM Diff RMSNorm",
        "FM Diff quantize + clamp",
        "FM Abs sigmoid + value gate",
        "FM Abs gated-value quantize + clamp",
        "FM Abs squared",
        "Main fc_0",
        "Main gate sigmoid (Q64)",
        "Main gate apply (/128)",
        "Main gate clamp (int32, no narrow)",
        "Main SqrClippedReLU",
        "Main ClippedReLU",
        "LCA FM input assembly",
        "LCA Q affine",
        "LCA K affine",
        "LCA V affine",
        "LCA dot product + attention logit",
        "LCA sigmoid / attention score",
        "LCA value clamp + correction term",
        "LCA final add + quantize / narrow",
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

enum class PairWeightReuseStage {
  MainPairAndPacking,
  FullTransform,
  FreshEvaluate,
};

constexpr std::array<const char*, 4> kPairWeightReuseVariantNames = {
    "A. legacy perspective-outer",
    "B1. shared weight load only",
    "B2. shared load + int32 expand",
    "B3. full chunk-outer expanded reuse",
};

constexpr std::array<const char*, 3> kPairWeightReuseStageNames = {
    "PairWeight blend + Main packing (complete Main FT stage)",
    "Transform total (precomputed accumulator)",
    "fresh NNUE evaluate pipeline (precomputed accumulator)",
};

template<int Variant, PairWeightReuseStage Stage>
NnueBenchTiming MeasurePairWeightReuseCorpus(
    const std::vector<FtTransformBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> main_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> diff_output{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType, 128> abs_output{};
  alignas(kCacheLineSize) std::int32_t router_output[32];
  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];
  NetworkBenchCase network_input{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    if constexpr (Stage == PairWeightReuseStage::MainPairAndPacking) {
      feature_transformer->BenchmarkTransformMainPairReuse<Variant>(
          *sample.position, main_output.data(), sample.material_bucket);
      KeepNnueBenchObject(main_output);
      MixNnueBenchChecksum(
          checksum, main_output[static_cast<std::size_t>(timing.calls)
                              % main_output.size()]);
    } else {
      feature_transformer->BenchmarkTransformReconstructedPairReuse<Variant>(
          *sample.position, main_output.data(), diff_output.data(),
          abs_output.data(), sample.material_bucket);
      if constexpr (Stage == PairWeightReuseStage::FullTransform) {
        KeepNnueBenchObject(main_output);
        KeepNnueBenchObject(diff_output);
        KeepNnueBenchObject(abs_output);
        const std::size_t main_index =
            static_cast<std::size_t>(timing.calls) % main_output.size();
        const std::size_t fm_index =
            static_cast<std::size_t>(timing.calls) % diff_output.size();
        MixNnueBenchChecksum(checksum, main_output[main_index]);
        MixNnueBenchChecksum(checksum, diff_output[fm_index]);
        MixNnueBenchChecksum(checksum, abs_output[fm_index]);
      } else {
        network_input.transformed = main_output;
        network_input.diff_transformed = diff_output;
        network_input.abs_transformed = abs_output;
        network_input.material_bucket = sample.material_bucket;
        FillNnueBenchRouterInput(network_input);
        router->PropagatePrefix<12>(
            network_input.router_input.data(), router_output);
        const int selected_bucket = SelectNnueBenchBucket(router_output);
        const std::int32_t final_output =
            NnueBenchSelectedNetwork(selected_bucket)
                .Propagate(
                    network_input.transformed.data(),
                    network_input.diff_transformed.data(),
                    network_input.abs_transformed.data(),
                    network_input.material_bucket, network_buffer)[0];
        MixNnueBenchChecksum(checksum, final_output);
      }
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<int Variant>
NnueBenchTiming MeasurePairWeightReuseByStage(
    const std::vector<FtTransformBenchCase>& corpus,
    const PairWeightReuseStage stage, std::uint64_t& checksum) {
  switch (stage) {
    case PairWeightReuseStage::MainPairAndPacking:
      return MeasurePairWeightReuseCorpus<
          Variant, PairWeightReuseStage::MainPairAndPacking>(corpus,
                                                              checksum);
    case PairWeightReuseStage::FullTransform:
      return MeasurePairWeightReuseCorpus<
          Variant, PairWeightReuseStage::FullTransform>(corpus, checksum);
    case PairWeightReuseStage::FreshEvaluate:
      return MeasurePairWeightReuseCorpus<
          Variant, PairWeightReuseStage::FreshEvaluate>(corpus, checksum);
  }
  return {};
}

NnueBenchTiming MeasurePairWeightReuseVariant(
    const std::vector<FtTransformBenchCase>& corpus, const int variant,
    const PairWeightReuseStage stage, std::uint64_t& checksum) {
  switch (variant) {
    case 0: return MeasurePairWeightReuseByStage<0>(corpus, stage, checksum);
    case 1: return MeasurePairWeightReuseByStage<1>(corpus, stage, checksum);
    case 2: return MeasurePairWeightReuseByStage<2>(corpus, stage, checksum);
    case 3: return MeasurePairWeightReuseByStage<3>(corpus, stage, checksum);
    default: return {};
  }
}

void TestPairWeightPerspectiveReuseBenchmark(
    const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: PairWeight perspective reuse]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  variants     : legacy / load / expand / full chunk"
            << std::endl
            << "  production path changed: no" << std::endl
            << "  corpus build : real positions and precomputed accumulators..."
            << std::flush;
  const auto corpus = MakeFtTransformStageBenchCorpus();
  std::cout << "done (" << corpus.size() << ")" << std::endl;
  if (corpus.empty()) {
    std::cout << "error: PairWeight benchmark corpus is empty" << std::endl;
    return;
  }
  std::cout << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl
            << "[source-level intended counts per 32 outputs]" << std::endl
            << "  note: compiler CSE may share additional extensions"
            << std::endl
            << "  A  : weight loads 24, int32 extends 24" << std::endl
            << "  B1 : weight loads 12, int32 extends 24" << std::endl
            << "  B2 : weight loads 12, int32 extends 12" << std::endl
            << "  B3 : weight loads 12, int32 extends 12; 12 expanded weights live"
            << std::endl;

  constexpr std::array<PairWeightReuseStage, 3> stages = {
      PairWeightReuseStage::MainPairAndPacking,
      PairWeightReuseStage::FullTransform,
      PairWeightReuseStage::FreshEvaluate,
  };
  std::array<std::array<NnueBenchSamples, 4>, 3> samples;
  std::array<std::array<std::uint64_t, 4>, 3> checksums;
  for (auto& row : checksums)
    row.fill(UINT64_C(14695981039346656037));

  for (std::size_t stage_index = 0; stage_index < stages.size();
       ++stage_index) {
    for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
      for (int offset = 0; offset < 4; ++offset) {
        const int variant = static_cast<int>((repeat + offset) % 4);
        std::uint64_t warmup = UINT64_C(14695981039346656037);
        MeasurePairWeightReuseVariant(
            corpus, variant, stages[stage_index], warmup);
        samples[stage_index][variant].Add(MeasurePairWeightReuseVariant(
            corpus, variant, stages[stage_index],
            checksums[stage_index][variant]));
      }
    }
  }

  for (std::size_t stage_index = 0; stage_index < stages.size();
       ++stage_index) {
    std::cout << kPairWeightReuseStageNames[stage_index] << std::endl;
    const auto baseline =
        SummarizeNnueBenchSamples(samples[stage_index][0]);
    for (int variant = 0; variant < 4; ++variant) {
      PrintNnueBenchSamples(
          kPairWeightReuseVariantNames[variant],
          samples[stage_index][variant]);
      if (variant != 0) {
        const auto candidate =
            SummarizeNnueBenchSamples(samples[stage_index][variant]);
        const double improvement = baseline.median == 0.0
            ? 0.0
            : (baseline.median - candidate.median) * 100.0
                / baseline.median;
        std::cout << "  improvement vs A : " << std::fixed
                  << std::setprecision(2) << improvement << "%"
                  << std::endl;
      }
    }
  }

  std::array<std::uint64_t, 4> output_checksums;
  std::array<std::uint64_t, 4> final_checksums;
  output_checksums.fill(UINT64_C(14695981039346656037));
  final_checksums.fill(UINT64_C(14695981039346656037));
  std::array<std::uint64_t, 4> black_mismatches{};
  std::array<std::uint64_t, 4> white_mismatches{};
  std::array<std::uint64_t, 4> packed_mismatches{};
  std::array<std::uint64_t, 4> final_mismatches{};
  std::array<std::int32_t, 4> first_final_reference{};
  std::array<std::int32_t, 4> first_final_candidate{};
  std::array<std::size_t, 4> first_final_sample{};
  alignas(kCacheLineSize)
      std::array<FeatureTransformer::OutputType,
                 FeatureTransformer::kOutputDimensions> candidate_main{};
  alignas(kCacheLineSize) std::int32_t router_output[32];
  alignas(kCacheLineSize) char network_buffer[Network::kBufferSize];
  NetworkBenchCase network_input{};

  auto final_from_main = [&](const FtTransformBenchCase& sample,
                             const auto& main) {
    network_input.transformed = main;
    network_input.diff_transformed = sample.expected_diff;
    network_input.abs_transformed = sample.expected_abs;
    network_input.material_bucket = sample.material_bucket;
    FillNnueBenchRouterInput(network_input);
    router->PropagatePrefix<12>(
        network_input.router_input.data(), router_output);
    const int selected_bucket = SelectNnueBenchBucket(router_output);
    return NnueBenchSelectedNetwork(selected_bucket)
        .Propagate(
            network_input.transformed.data(),
            network_input.diff_transformed.data(),
            network_input.abs_transformed.data(),
            network_input.material_bucket, network_buffer)[0];
  };

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const std::int32_t reference_final =
        final_from_main(sample, sample.expected_main);
    for (int variant = 0; variant < 4; ++variant) {
      switch (variant) {
        case 0:
          feature_transformer->BenchmarkTransformMainPairReuse<0>(
              *sample.position, candidate_main.data(), sample.material_bucket);
          break;
        case 1:
          feature_transformer->BenchmarkTransformMainPairReuse<1>(
              *sample.position, candidate_main.data(), sample.material_bucket);
          break;
        case 2:
          feature_transformer->BenchmarkTransformMainPairReuse<2>(
              *sample.position, candidate_main.data(), sample.material_bucket);
          break;
        case 3:
          feature_transformer->BenchmarkTransformMainPairReuse<3>(
              *sample.position, candidate_main.data(), sample.material_bucket);
          break;
      }
      const Color first = sample.position->side_to_move();
      constexpr std::size_t half =
          FeatureTransformer::kOutputDimensions / 2;
      for (std::size_t index = 0; index < candidate_main.size(); ++index) {
        MixNnueBenchChecksum(output_checksums[variant], candidate_main[index]);
        if (candidate_main[index] != sample.expected_main[index]) {
          ++packed_mismatches[variant];
          const Color perspective = index < half ? first : ~first;
          if (perspective == BLACK)
            ++black_mismatches[variant];
          else
            ++white_mismatches[variant];
        }
      }
      const std::int32_t candidate_final =
          final_from_main(sample, candidate_main);
      MixNnueBenchChecksum(final_checksums[variant], candidate_final);
      if (candidate_final != reference_final) {
        if (final_mismatches[variant] == 0) {
          first_final_sample[variant] = sample_index;
          first_final_reference[variant] = reference_final;
          first_final_candidate[variant] = candidate_final;
        }
        ++final_mismatches[variant];
      }
    }
  }

  std::cout << "[correctness]" << std::endl;
  for (int variant = 0; variant < 4; ++variant) {
    std::cout << "  " << kPairWeightReuseVariantNames[variant] << std::endl
              << "    BLACK output mismatch : "
              << black_mismatches[variant] << std::endl
              << "    WHITE output mismatch : "
              << white_mismatches[variant] << std::endl
              << "    packed Main mismatch  : "
              << packed_mismatches[variant] << std::endl
              << "    final Network mismatch: "
              << final_mismatches[variant] << std::endl
              << "    Main checksum : 0x" << std::hex
              << output_checksums[variant] << std::endl
              << "    final checksum: 0x" << final_checksums[variant]
              << std::dec << std::endl;
    if (final_mismatches[variant] != 0)
      std::cout << "    first final mismatch: sample="
                << first_final_sample[variant] << " A="
                << first_final_reference[variant] << " B="
                << first_final_candidate[variant] << std::endl;
  }
  std::cout << "[timing checksum match by stage]" << std::endl;
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    std::cout << "  " << kPairWeightReuseStageNames[stage] << " : ";
    bool match = true;
    for (int variant = 1; variant < 4; ++variant)
      match = match && checksums[stage][variant] == checksums[stage][0];
    std::cout << (match ? "yes" : "NO") << std::endl;
  }
}

struct alignas(kCacheLineSize) NetworkStageBenchCase {
  NetworkBenchCase input;
  Network::Buffer intermediate;
  float diff_sum_sq = 0.0f;
  float diff_inv_rms = 0.0f;
  std::array<std::int32_t, 32> abs_gated{};
  std::array<std::uint8_t, 32> diff_before_lca;
  // BenchmarkMainFc0() uses the AVX2 affine kernel, whose output stores require
  // the same SIMD alignment as Network::Buffer::fc_0_out.
  alignas(kCacheLineSize) std::array<std::int32_t, 32> main_before_gate;
  std::array<std::int32_t, 32> main_gate_q64;
  std::array<std::int32_t, 32> main_after_gate_before_clamp;
  std::array<float, 6> phase_values{};
  Network::BenchmarkPhaseScales phase_split_scales{};
  float lca_dot_product = 0.0f;
  float lca_attention_logit = 0.0f;
  float lca_attention_score = 0.0f;
  std::array<float, 32> lca_value_clamped{};
  std::array<float, 32> lca_value_correction{};
  std::array<float, 32> lca_output_before_narrow{};
  std::array<std::uint8_t, 32> lca_reconstructed_output{};
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
  for (IndexType j = 0; j < LCA_QK_SIZE; ++j)
    result.dot_product +=
        (static_cast<float>(query_output[j]) / 8128.0f)
        * (static_cast<float>(key_output[j]) / 8128.0f);
  result.attention_logit =
      (result.dot_product * 0.17677f) / selected_network.lca_temp;
  result.attention_score =
      1.0f / (1.0f + std::exp(-result.attention_logit));

  for (int j = 0; j < 32; ++j) {
    const float current_diff = static_cast<float>(diff_input[j]) / 127.0f;
    const float value = static_cast<float>(Network::LcaValueForDiffChannel(
        value_output, j)) / 8128.0f;
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
  std::cout << "  corpus build : network inputs..." << std::flush;
  const auto input_corpus = MakeNnueNetworkBenchCorpus();
  std::cout << "done (" << input_corpus.size() << "), stage captures..."
            << std::flush;
  std::vector<NetworkStageBenchCase> corpus;
  corpus.reserve(input_corpus.size());

  alignas(kCacheLineSize) Network::Buffer network_buffer{};
  std::size_t input_index = 0;
  for (const auto& input : input_corpus) {
    NetworkStageBenchCase sample{};
    sample.input = input;
    const Network& selected_network =
        NnueBenchSelectedNetwork(input.selected_bucket);
    const auto output = selected_network.Propagate(
        input.transformed.data(), input.diff_transformed.data(),
        input.abs_transformed.data(), input.material_bucket,
        reinterpret_cast<char*>(&network_buffer));
    sample.final_output = output[0];
    sample.intermediate = network_buffer;
#if defined(NNUE_COMPACT_PHASE5)
    // AbsSqr is not a production intermediate in the compact architecture.
    // Materialize it only for the retained diagnostic stage/checksum.
    selected_network.BenchmarkAbsSquared(
        sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out);
#endif

    sample.phase_scales = selected_network.BenchmarkPhaseScalesFromOutput(
        sample.intermediate.phase_out);
    selected_network.BenchmarkPhaseSigmoid(
        sample.intermediate.phase_out, sample.phase_values.data());
    sample.phase_split_scales = selected_network.BenchmarkPhaseChannelScales(
        sample.phase_values.data());
    selected_network.BenchmarkDiffRmsNorm(
        sample.intermediate.diff_fc_out, &sample.diff_sum_sq,
        &sample.diff_inv_rms);
    selected_network.BenchmarkAbsSigmoidGate(
        sample.intermediate.abs_fc_out, sample.abs_gated.data());
    std::copy_n(sample.intermediate.fm_cat_uint8, 32,
                sample.diff_before_lca.begin());
    selected_network.BenchmarkLcaDotAndLogit(
        sample.intermediate.lca_q_out, sample.intermediate.lca_k_out,
        &sample.lca_dot_product, &sample.lca_attention_logit);
    selected_network.BenchmarkLcaAttentionScore(
        sample.lca_attention_logit, &sample.lca_attention_score);
    selected_network.BenchmarkLcaValueClampAndCorrection(
        sample.intermediate.lca_v_out, sample.lca_attention_score,
        sample.lca_value_clamped.data(),
        sample.lca_value_correction.data());
    selected_network.BenchmarkLcaFinalAddAndQuantize(
        sample.diff_before_lca.data(), sample.lca_attention_score,
        sample.lca_value_correction.data(),
        sample.lca_output_before_narrow.data(),
        sample.lca_reconstructed_output.data());
    selected_network.BenchmarkMainFc0(
        input.transformed.data(), sample.main_before_gate.data());
    selected_network.BenchmarkMainGateSigmoid(
        sample.intermediate.diff_fc_out, sample.main_gate_q64.data());
    selected_network.BenchmarkMainGateApply(
        sample.main_before_gate.data(), sample.main_gate_q64.data(),
        sample.main_after_gate_before_clamp.data());
    selected_network.BenchmarkFc2(sample.intermediate.ac_1_out,
                                  &sample.fc2_before_blend);
    corpus.emplace_back(std::move(sample));
    ++input_index;
    if (input_index % 1000 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "done" << std::endl;
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
    MixNnueBenchRange(captured_checksums[1], sample.intermediate.phase_out,
                      PHASE_OUTPUT_SIZE);
    for (IndexType index = 0; index < PHASE_OUTPUT_SIZE; ++index)
      MixNnueBenchFloatBits(captured_checksums[2], sample.phase_values[index]);
    MixNnueBenchPhaseScales(captured_checksums[3],
                            sample.phase_split_scales);
    MixNnueBenchRange(captured_checksums[4], sample.intermediate.diff_fc_out,
                      64);
    MixNnueBenchRange(captured_checksums[5], sample.intermediate.abs_fc_out,
                      64);
    MixNnueBenchFloatBits(captured_checksums[6], sample.diff_sum_sq);
    MixNnueBenchFloatBits(captured_checksums[6], sample.diff_inv_rms);
    MixNnueBenchRange(captured_checksums[7], sample.diff_before_lca.data(), 32);
    MixNnueBenchRange(captured_checksums[8], sample.abs_gated.data(), 32);
    MixNnueBenchRange(captured_checksums[9], sample.intermediate.abs_ac_out,
                      32);
    MixNnueBenchRange(captured_checksums[10], sample.intermediate.abs_sqr_out,
                      32);
    MixNnueBenchRange(captured_checksums[11], sample.main_before_gate.data(),
                      32);
    MixNnueBenchRange(captured_checksums[12], sample.main_gate_q64.data(), 32);
    MixNnueBenchRange(captured_checksums[13],
                      sample.main_after_gate_before_clamp.data(), 32);
    MixNnueBenchRange(captured_checksums[14], sample.intermediate.fc_0_out, 32);
    MixNnueBenchRange(captured_checksums[15],
                      sample.intermediate.ac_sqr_0_out_temp,
                      32);
    MixNnueBenchRange(captured_checksums[16], sample.intermediate.ac_0_out, 32);
    MixNnueBenchRange(captured_checksums[17], sample.intermediate.fm_cat_uint8,
                      64);
    MixNnueBenchRange(captured_checksums[18], sample.intermediate.lca_q_out,
                      LCA_QK_SIZE);
    MixNnueBenchRange(captured_checksums[19], sample.intermediate.lca_k_out,
                      LCA_QK_SIZE);
    MixNnueBenchRange(captured_checksums[20], sample.intermediate.lca_v_out,
                      LCA_VALUE_SIZE);
    MixNnueBenchFloatBits(captured_checksums[21], sample.lca_dot_product);
    MixNnueBenchFloatBits(captured_checksums[21], sample.lca_attention_logit);
    MixNnueBenchFloatBits(captured_checksums[22], sample.lca_attention_score);
    for (int j = 0; j < 32; ++j) {
      MixNnueBenchFloatBits(captured_checksums[23],
                            sample.lca_value_clamped[j]);
      MixNnueBenchFloatBits(captured_checksums[23],
                            sample.lca_value_correction[j]);
      MixNnueBenchFloatBits(captured_checksums[24],
                            sample.lca_output_before_narrow[j]);
    }
    MixNnueBenchRange(captured_checksums[24],
                      sample.lca_reconstructed_output.data(), 32);
    MixNnueBenchRange(captured_checksums[25], sample.intermediate.cross_cat,
                      32);
    MixNnueBenchRange(captured_checksums[25], sample.intermediate.cross_fc_out,
                      CROSS_OUTPUT_SIZE);
    MixNnueBenchRange(captured_checksums[25], sample.intermediate.cross_feat,
                      CROSS_OUTPUT_SIZE);
    MixNnueBenchRange(captured_checksums[26], sample.intermediate.l2_input,
                      L2_REAL_SIZE);
    MixNnueBenchRange(captured_checksums[27], sample.intermediate.fc_1_out,
                      kHidden2Dims);
    MixNnueBenchRange(captured_checksums[28], sample.intermediate.ac_1_out,
                      kHidden2Dims);
    MixNnueBenchChecksum(captured_checksums[29], sample.fc2_before_blend);
    MixNnueBenchChecksum(captured_checksums[30], sample.final_output);

    selected_network.BenchmarkPhaseInputAssembly(
        sample.input.transformed.data(), sample.input.diff_transformed.data(),
        sample.input.abs_transformed.data(), sample.input.material_bucket,
        work.phase_input);
    MixNnueBenchRange(recomputed_checksums[0], work.phase_input, 384);
    CompareNnueNetworkStageRange(
        sample.intermediate.phase_input, work.phase_input, 384, sample_index,
        mismatches[0]);

    selected_network.BenchmarkPhaseProjection(
        sample.intermediate.phase_input, work.phase_out);
    MixNnueBenchRange(recomputed_checksums[1], work.phase_out,
                      PHASE_OUTPUT_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.phase_out, work.phase_out, PHASE_OUTPUT_SIZE,
        sample_index,
        mismatches[1]);

    float phase_values[PHASE_OUTPUT_SIZE];
    selected_network.BenchmarkPhaseSigmoid(
        sample.intermediate.phase_out, phase_values);
    for (IndexType index = 0; index < PHASE_OUTPUT_SIZE; ++index)
      MixNnueBenchFloatBits(recomputed_checksums[2], phase_values[index]);
    CompareNnueNetworkFloatRange(
        sample.phase_values.data(), phase_values, PHASE_OUTPUT_SIZE, sample_index,
        float_mismatches[2]);

    const auto phase_split_scales =
        selected_network.BenchmarkPhaseChannelScales(
            sample.phase_values.data());
    MixNnueBenchPhaseScales(recomputed_checksums[3], phase_split_scales);
    const auto captured_phase_split_scales =
        NnueBenchPhaseScalesToArray(sample.phase_split_scales);
    const auto recomputed_phase_split_scales =
        NnueBenchPhaseScalesToArray(phase_split_scales);
    CompareNnueNetworkFloatRange(
        captured_phase_split_scales.data(),
        recomputed_phase_split_scales.data(), 6, sample_index,
        float_mismatches[3]);

    selected_network.BenchmarkFcDiff(
        sample.input.diff_transformed.data(), work.diff_fc_out);
    MixNnueBenchRange(recomputed_checksums[4], work.diff_fc_out, 64);
    CompareNnueNetworkStageRange(
        sample.intermediate.diff_fc_out, work.diff_fc_out, 64, sample_index,
        mismatches[4]);
    selected_network.BenchmarkFcAbs(
        sample.input.abs_transformed.data(), work.abs_fc_out);
    MixNnueBenchRange(recomputed_checksums[5], work.abs_fc_out, 64);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_fc_out, work.abs_fc_out, 64, sample_index,
        mismatches[5]);

    float diff_sum_sq = 0.0f;
    float diff_inv_rms = 0.0f;
    selected_network.BenchmarkDiffRmsNorm(
        sample.intermediate.diff_fc_out, &diff_sum_sq, &diff_inv_rms);
    MixNnueBenchFloatBits(recomputed_checksums[6], diff_sum_sq);
    MixNnueBenchFloatBits(recomputed_checksums[6], diff_inv_rms);
    const float captured_diff_rms[2] = {
        sample.diff_sum_sq, sample.diff_inv_rms};
    const float recomputed_diff_rms[2] = {diff_sum_sq, diff_inv_rms};
    CompareNnueNetworkFloatRange(
        captured_diff_rms, recomputed_diff_rms, 2, sample_index,
        float_mismatches[6]);

    selected_network.BenchmarkDiffQuantize(
        sample.intermediate.diff_fc_out, sample.diff_inv_rms,
        work.diff_ac_out);
    MixNnueBenchRange(recomputed_checksums[7], work.diff_ac_out, 32);
    CompareNnueNetworkStageRange(
        sample.diff_before_lca.data(), work.diff_ac_out, 32, sample_index,
        mismatches[7]);

    std::int32_t abs_gated[32];
    selected_network.BenchmarkAbsSigmoidGate(
        sample.intermediate.abs_fc_out, abs_gated);
    MixNnueBenchRange(recomputed_checksums[8], abs_gated, 32);
    CompareNnueNetworkStageRange(
        sample.abs_gated.data(), abs_gated, 32, sample_index,
        mismatches[8]);

    selected_network.BenchmarkAbsGateQuantize(
        sample.abs_gated.data(), work.abs_ac_out);
    MixNnueBenchRange(recomputed_checksums[9], work.abs_ac_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_ac_out, work.abs_ac_out, 32, sample_index,
        mismatches[9]);

    selected_network.BenchmarkAbsSquared(
        sample.intermediate.abs_ac_out, work.abs_sqr_out);
    MixNnueBenchRange(recomputed_checksums[10], work.abs_sqr_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.abs_sqr_out, work.abs_sqr_out, 32, sample_index,
        mismatches[10]);

    selected_network.BenchmarkMainFc0(sample.input.transformed.data(),
                                      work.fc_0_out);
    MixNnueBenchRange(recomputed_checksums[11], work.fc_0_out, 32);
    CompareNnueNetworkStageRange(
        sample.main_before_gate.data(), work.fc_0_out, 32, sample_index,
        mismatches[11]);

    std::int32_t main_gate_q64[32];
    selected_network.BenchmarkMainGateSigmoid(
        sample.intermediate.diff_fc_out, main_gate_q64);
    MixNnueBenchRange(recomputed_checksums[12], main_gate_q64, 32);
    CompareNnueNetworkStageRange(
        sample.main_gate_q64.data(), main_gate_q64, 32, sample_index,
        mismatches[12]);

    selected_network.BenchmarkMainGateApply(
        sample.main_before_gate.data(), sample.main_gate_q64.data(),
        work.fc_0_out);
    MixNnueBenchRange(recomputed_checksums[13], work.fc_0_out, 32);
    CompareNnueNetworkStageRange(
        sample.main_after_gate_before_clamp.data(), work.fc_0_out, 32,
        sample_index, mismatches[13]);

    selected_network.BenchmarkMainGateClamp(work.fc_0_out, work.fc_0_out);
    MixNnueBenchRange(recomputed_checksums[14], work.fc_0_out, 32);
    CompareNnueNetworkStageRange(
        sample.intermediate.fc_0_out, work.fc_0_out, 32, sample_index,
        mismatches[14]);
    selected_network.BenchmarkMainSqrClippedRelu(
        sample.intermediate.fc_0_out, work.ac_sqr_0_out_temp);
    MixNnueBenchRange(recomputed_checksums[15],
                      work.ac_sqr_0_out_temp, 32);
    selected_network.BenchmarkMainClippedRelu(
        sample.intermediate.fc_0_out, work.ac_0_out);
    MixNnueBenchRange(recomputed_checksums[16], work.ac_0_out, 32);

    selected_network.BenchmarkLcaAssembleFmInput(
        sample.diff_before_lca.data(), sample.intermediate.abs_ac_out,
        work.fm_cat_uint8);
    MixNnueBenchRange(recomputed_checksums[17], work.fm_cat_uint8, 64);
    CompareNnueNetworkStageRange(
        sample.intermediate.fm_cat_uint8, work.fm_cat_uint8, 64,
        sample_index, mismatches[17]);

    selected_network.BenchmarkLcaQuery(
        sample.intermediate.ac_0_out, work.lca_q_out);
    MixNnueBenchRange(recomputed_checksums[18], work.lca_q_out, LCA_QK_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.lca_q_out, work.lca_q_out, LCA_QK_SIZE, sample_index,
        mismatches[18]);

    selected_network.BenchmarkLcaKey(
        sample.intermediate.fm_cat_uint8, work.lca_k_out);
    MixNnueBenchRange(recomputed_checksums[19], work.lca_k_out, LCA_QK_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.lca_k_out, work.lca_k_out, LCA_QK_SIZE, sample_index,
        mismatches[19]);

    selected_network.BenchmarkLcaValue(
        sample.intermediate.fm_cat_uint8, work.lca_v_out);
    MixNnueBenchRange(recomputed_checksums[20], work.lca_v_out, LCA_VALUE_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.lca_v_out, work.lca_v_out, LCA_VALUE_SIZE, sample_index,
        mismatches[20]);

    float lca_dot_product = 0.0f;
    float lca_attention_logit = 0.0f;
    selected_network.BenchmarkLcaDotAndLogit(
        sample.intermediate.lca_q_out, sample.intermediate.lca_k_out,
        &lca_dot_product, &lca_attention_logit);
    MixNnueBenchFloatBits(recomputed_checksums[21], lca_dot_product);
    MixNnueBenchFloatBits(recomputed_checksums[21], lca_attention_logit);
    const float captured_lca_dot_logit[2] = {
        sample.lca_dot_product, sample.lca_attention_logit};
    const float recomputed_lca_dot_logit[2] = {
        lca_dot_product, lca_attention_logit};
    CompareNnueNetworkFloatRange(
        captured_lca_dot_logit, recomputed_lca_dot_logit, 2, sample_index,
        float_mismatches[21]);

    float lca_attention_score = 0.0f;
    selected_network.BenchmarkLcaAttentionScore(
        sample.lca_attention_logit, &lca_attention_score);
    MixNnueBenchFloatBits(recomputed_checksums[22], lca_attention_score);
    CompareNnueNetworkFloatRange(
        &sample.lca_attention_score, &lca_attention_score, 1, sample_index,
        float_mismatches[22]);

    alignas(kCacheLineSize) float lca_value_clamped[32];
    alignas(kCacheLineSize) float lca_value_correction[32];
    selected_network.BenchmarkLcaValueClampAndCorrection(
        sample.intermediate.lca_v_out, sample.lca_attention_score,
        lca_value_clamped, lca_value_correction);
    for (int j = 0; j < 32; ++j) {
      MixNnueBenchFloatBits(recomputed_checksums[23], lca_value_clamped[j]);
      MixNnueBenchFloatBits(recomputed_checksums[23],
                            lca_value_correction[j]);
    }
    CompareNnueNetworkFloatRange(
        sample.lca_value_clamped.data(), lca_value_clamped, 32,
        sample_index, float_mismatches[23]);
    CompareNnueNetworkFloatRange(
        sample.lca_value_correction.data(), lca_value_correction, 32,
        sample_index, float_mismatches[23]);

    alignas(kCacheLineSize) float lca_output_before_narrow[32];
    selected_network.BenchmarkLcaFinalAddAndQuantize(
        sample.diff_before_lca.data(), sample.lca_attention_score,
        sample.lca_value_correction.data(), lca_output_before_narrow,
        work.diff_ac_out);
    for (int j = 0; j < 32; ++j)
      MixNnueBenchFloatBits(recomputed_checksums[24],
                            lca_output_before_narrow[j]);
    MixNnueBenchRange(recomputed_checksums[24], work.diff_ac_out, 32);
    CompareNnueNetworkFloatRange(
        sample.lca_output_before_narrow.data(), lca_output_before_narrow, 32,
        sample_index, float_mismatches[24]);
    CompareNnueNetworkStageRange(
        sample.lca_reconstructed_output.data(), work.diff_ac_out, 32,
        sample_index, mismatches[24]);

    selected_network.BenchmarkCross(
        sample.intermediate.ac_sqr_0_out_temp,
        sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
        sample.intermediate.abs_ac_out, work.cross_cat, work.cross_fc_out,
        work.cross_feat);
    MixNnueBenchRange(recomputed_checksums[25], work.cross_cat, 32);
    MixNnueBenchRange(recomputed_checksums[25], work.cross_fc_out,
                      CROSS_OUTPUT_SIZE);
    MixNnueBenchRange(recomputed_checksums[25], work.cross_feat,
                      CROSS_OUTPUT_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.cross_fc_out, work.cross_fc_out,
        CROSS_OUTPUT_SIZE, sample_index, mismatches[25]);
    CompareNnueNetworkStageRange(
        sample.intermediate.cross_feat, work.cross_feat,
        CROSS_OUTPUT_SIZE, sample_index, mismatches[25]);

#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
    std::int32_t production_scales_q23[PHASE_OUTPUT_SIZE];
    selected_network.BenchmarkPhaseFixedScalesQ23(
        sample.intermediate.phase_out, production_scales_q23);
    selected_network.BenchmarkL2AssemblyQ23(
        sample.intermediate.ac_sqr_0_out_temp,
        sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
        sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
        sample.intermediate.cross_feat, production_scales_q23,
        work.l2_input);
#else
    selected_network.BenchmarkL2Assembly(
        sample.intermediate.ac_sqr_0_out_temp,
        sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
        sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
        sample.intermediate.cross_feat, sample.phase_scales, work.l2_input);
#endif
    MixNnueBenchRange(recomputed_checksums[26], work.l2_input, L2_REAL_SIZE);
    CompareNnueNetworkStageRange(
        sample.intermediate.l2_input, work.l2_input, L2_REAL_SIZE, sample_index,
        mismatches[26]);
    CompareNnueNetworkStageRange(
        sample.intermediate.l2_input + L2_REAL_SIZE,
        work.l2_input + L2_REAL_SIZE, L2_PADDING_SIZE,
        sample_index, l2_padding_mismatch);

    selected_network.BenchmarkFc1(
        sample.intermediate.l2_input, work.fc_1_out);
    MixNnueBenchRange(recomputed_checksums[27], work.fc_1_out,
                      kHidden2Dims);
    CompareNnueNetworkStageRange(
        sample.intermediate.fc_1_out, work.fc_1_out, kHidden2Dims,
        sample_index, mismatches[27]);

    selected_network.BenchmarkAc1(
        sample.intermediate.fc_1_out, work.ac_1_out);
    MixNnueBenchRange(recomputed_checksums[28], work.ac_1_out,
                      kHidden2Dims);
    CompareNnueNetworkStageRange(
        sample.intermediate.ac_1_out, work.ac_1_out, kHidden2Dims,
        sample_index, mismatches[28]);

    selected_network.BenchmarkFc2(sample.intermediate.ac_1_out,
                                  work.fc_2_out);
    MixNnueBenchChecksum(recomputed_checksums[29], work.fc_2_out[0]);
    work.fc_2_out[0] = selected_network.BenchmarkBlend(
        sample.intermediate.fc_0_out[31], sample.fc2_before_blend);
    MixNnueBenchChecksum(recomputed_checksums[30], work.fc_2_out[0]);

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
        phase_output, sample.intermediate.phase_out, work.phase_out,
        PHASE_OUTPUT_SIZE,
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
        lca_query, sample.intermediate.lca_q_out, work.lca_q_out, LCA_QK_SIZE,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_key, sample.intermediate.lca_k_out, work.lca_k_out, LCA_QK_SIZE,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        lca_value, sample.intermediate.lca_v_out, work.lca_v_out, LCA_VALUE_SIZE,
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
  PrintNnueBenchIntegerDiagnostic(
      PHASE_OUTPUT_SIZE == 5 ? "phase.proj_output[5]"
                             : "phase.proj_output[6]",
      phase_output);
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
std::int32_t ComputeNnueNetworkStagedOutputFromPhaseInput(
    const NetworkStageBenchCase& sample, const int selected_bucket,
    const std::uint8_t* phase_input, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(selected_bucket);
  selected_network.BenchmarkPhaseProjection(phase_input, work.phase_out);
#if !defined(USE_NNUE_PHASE_L2_FIXED_C32)
  const auto scales =
      selected_network.BenchmarkPhaseScalesFromOutput(work.phase_out);
#endif
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
#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
  std::int32_t production_scales_q23[PHASE_OUTPUT_SIZE];
  selected_network.BenchmarkPhaseFixedScalesQ23(
      work.phase_out, production_scales_q23);
  selected_network.BenchmarkL2AssemblyQ23(
      work.ac_sqr_0_out_temp, work.ac_0_out, work.diff_ac_out,
      work.abs_ac_out, work.abs_sqr_out, work.cross_feat,
      production_scales_q23, work.l2_input);
#else
  selected_network.BenchmarkL2Assembly(
      work.ac_sqr_0_out_temp, work.ac_0_out, work.diff_ac_out,
      work.abs_ac_out, work.abs_sqr_out, work.cross_feat, scales,
      work.l2_input);
#endif
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

template<bool UseTiledFc1>
std::int32_t ComputeNnueNetworkStagedOutput(
    const NetworkStageBenchCase& sample, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(sample.input.selected_bucket);
  selected_network.BenchmarkPhaseInputAssembly(
      sample.input.transformed.data(), sample.input.diff_transformed.data(),
      sample.input.abs_transformed.data(), sample.input.material_bucket,
      work.phase_input);
  return ComputeNnueNetworkStagedOutputFromPhaseInput<UseTiledFc1>(
      sample, sample.input.selected_bucket, work.phase_input, work);
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

enum class RouterPhaseInputImplementation {
  Separate,
  Copy,
  Shared,
};

void AssembleNnueRouterPhaseCommonInput(
    const NetworkBenchCase& sample, std::uint8_t* output) {
  for (int index = 0; index < 128; ++index) {
    const int abs_value = sample.abs_transformed[index];
    output[index] = static_cast<std::uint8_t>(
        std::clamp((abs_value - 64) * 2, 0, 127));
    output[index + 128] = sample.diff_transformed[index];
    output[index + 256] = sample.transformed[index];
  }
}

template<RouterPhaseInputImplementation Implementation>
int PrepareNnueRouterAndPhaseInput(
    const NetworkBenchCase& sample, std::uint8_t* common_or_router_input,
    std::uint8_t* phase_input, std::int32_t* router_output) {
  AssembleNnueRouterPhaseCommonInput(sample, common_or_router_input);
  router->PropagatePrefix<12>(common_or_router_input, router_output);
  const int selected_bucket = SelectNnueBenchBucket(router_output);

  if constexpr (Implementation == RouterPhaseInputImplementation::Separate) {
    NnueBenchSelectedNetwork(selected_bucket).BenchmarkPhaseInputAssembly(
        sample.transformed.data(), sample.diff_transformed.data(),
        sample.abs_transformed.data(), sample.material_bucket, phase_input);
  } else {
    if constexpr (Implementation == RouterPhaseInputImplementation::Copy)
      std::memcpy(phase_input, common_or_router_input, 384);
    else
      phase_input = common_or_router_input;
    phase_input[127] =
        static_cast<std::uint8_t>((sample.material_bucket * 127) / 11);
  }
  return selected_bucket;
}

enum class RouterPhaseAssemblyOperation {
  Router,
  Phase,
};

template<RouterPhaseAssemblyOperation Operation>
NnueBenchTiming MeasureNnueRouterPhaseAssembly(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::uint8_t output[384];
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    if constexpr (Operation == RouterPhaseAssemblyOperation::Router)
      AssembleNnueRouterPhaseCommonInput(sample.input, output);
    else
      NnueBenchSelectedNetwork(sample.input.selected_bucket)
          .BenchmarkPhaseInputAssembly(
              sample.input.transformed.data(),
              sample.input.diff_transformed.data(),
              sample.input.abs_transformed.data(),
              sample.input.material_bucket, output);
    const std::size_t index = static_cast<std::size_t>(timing.calls) % 384;
    MixNnueBenchChecksum(checksum, output[index]);
    MixNnueBenchChecksum(checksum, output[(index + 193) % 384]);
    KeepNnueBenchObject(output);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

#if defined(NNUE_COMPACT_PHASE5)
template<IndexType OutputCount>
NnueBenchTiming MeasureNnuePhasePrefix(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    NnueBenchSelectedNetwork(sample.input.selected_bucket)
        .phase_proj.PropagatePrefix<OutputCount>(
            sample.intermediate.phase_input, output);
    // Compare the five live channels only. Phase6 row 5 is padding in a
    // Phase5 file and exists here solely as the pre-cleanup timing control.
    for (IndexType index = 0; index < PHASE_OUTPUT_SIZE; ++index)
      MixNnueBenchChecksum(checksum, output[index]);
    KeepNnueBenchObject(output);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<IndexType OutputCount>
NnueBenchTiming MeasureNnuePhasePrefixAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureNnuePhasePrefix<OutputCount>(corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureNnuePhasePrefix<OutputCount>(corpus, checksum);
}
#endif

template<RouterPhaseInputImplementation Implementation, bool FullNetwork>
NnueBenchTiming MeasureNnueRouterPhaseCandidate(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::uint8_t common_or_router_input[384];
  alignas(kCacheLineSize) std::uint8_t separate_phase_input[384];
  alignas(kCacheLineSize) std::int32_t router_output[32];
  alignas(kCacheLineSize) std::int32_t phase_output[32];
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    std::uint8_t* phase_input =
        Implementation == RouterPhaseInputImplementation::Shared
        ? common_or_router_input : separate_phase_input;
    const int selected_bucket = PrepareNnueRouterAndPhaseInput<Implementation>(
        sample.input, common_or_router_input, phase_input, router_output);
    if constexpr (FullNetwork) {
      const std::int32_t output =
          ComputeNnueNetworkStagedOutputFromPhaseInput<false>(
              sample, selected_bucket, phase_input, work);
      MixNnueBenchChecksum(checksum, output);
    } else {
      NnueBenchSelectedNetwork(selected_bucket).BenchmarkPhaseProjection(
          phase_input, phase_output);
      const std::size_t index =
          static_cast<std::size_t>(timing.calls) % PHASE_OUTPUT_SIZE;
      MixNnueBenchChecksum(checksum, selected_bucket);
      MixNnueBenchChecksum(checksum, router_output[index]);
      MixNnueBenchChecksum(checksum, phase_output[index]);
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<RouterPhaseInputImplementation Implementation>
NnueBenchTiming MeasureNnueRouterPhasePreparationCandidate(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::uint8_t common_or_router_input[384];
  alignas(kCacheLineSize) std::uint8_t separate_phase_input[384];
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    AssembleNnueRouterPhaseCommonInput(sample.input,
                                       common_or_router_input);
    // C must expose the Router-form input before byte 127 is overwritten.
    // This compiler barrier has no runtime instructions on clang/GCC.
    KeepNnueBenchObject(common_or_router_input);
    std::uint8_t* phase_input = separate_phase_input;
    if constexpr (Implementation == RouterPhaseInputImplementation::Separate) {
      NnueBenchSelectedNetwork(sample.input.selected_bucket)
          .BenchmarkPhaseInputAssembly(
              sample.input.transformed.data(),
              sample.input.diff_transformed.data(),
              sample.input.abs_transformed.data(),
              sample.input.material_bucket, phase_input);
    } else {
      if constexpr (Implementation == RouterPhaseInputImplementation::Copy)
        std::memcpy(phase_input, common_or_router_input, 384);
      else
        phase_input = common_or_router_input;
      phase_input[127] = static_cast<std::uint8_t>(
          (sample.input.material_bucket * 127) / 11);
    }
    const std::size_t index = static_cast<std::size_t>(timing.calls) % 384;
    MixNnueBenchChecksum(checksum, phase_input[index]);
    KeepNnueBenchObject(phase_input[index]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<RouterPhaseInputImplementation Implementation>
NnueBenchTiming MeasureNnueRouterPhasePreparationAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureNnueRouterPhasePreparationCandidate<Implementation>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureNnueRouterPhasePreparationCandidate<Implementation>(
      corpus, checksum);
}

template<RouterPhaseInputImplementation Implementation, bool FullNetwork>
NnueBenchTiming MeasureNnueRouterPhaseCandidateAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureNnueRouterPhaseCandidate<Implementation, FullNetwork>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureNnueRouterPhaseCandidate<Implementation, FullNetwork>(
      corpus, checksum);
}

void PrintNnueRouterPhaseComparison(
    const char* name, const std::array<NnueBenchSamples, 3>& samples) {
  std::cout << name << std::endl;
  constexpr std::array<const char*, 3> names = {
      "A. separate Router / Phase assembly",
      "B. common assembly + memcpy + overwrite",
      "C. shared buffer + post-Router overwrite"};
  const auto baseline = SummarizeNnueBenchSamples(samples[0]);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    PrintNnueBenchSamples(names[index], samples[index]);
    const auto summary = SummarizeNnueBenchSamples(samples[index]);
    double variance = 0.0;
    if (samples[index].ns_per_call.size() > 1) {
      for (const double value : samples[index].ns_per_call) {
        const double delta = value - summary.mean;
        variance += delta * delta;
      }
      variance /= static_cast<double>(samples[index].ns_per_call.size() - 1);
    }
    const double improvement = baseline.median == 0.0 ? 0.0
        : (baseline.median - summary.median) * 100.0 / baseline.median;
    std::cout << "  sample stdev ns/call: " << std::fixed
              << std::setprecision(1) << std::sqrt(variance) << std::endl
              << "  relative improvement: "
              << std::setprecision(2) << improvement << "%" << std::endl;
  }
}

void TestRouterPhaseInputBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Router / Phase shared input]" << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : A/B/C rotated by one per repeat" << std::endl
            << "  shared source: identical transformed/diff/abs arrays" << std::endl
            << "  only difference: Phase input[127] material-bucket overwrite"
            << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: NNUE Router/Phase benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  NnueBenchSamples router_assembly;
  NnueBenchSamples phase_assembly;
  std::array<NnueBenchSamples, 3> preparation;
  std::array<NnueBenchSamples, 3> combined;
  std::array<NnueBenchSamples, 3> full;
#if defined(NNUE_COMPACT_PHASE5)
  std::array<NnueBenchSamples, 2> phase_prefix;
  std::array<std::uint64_t, 2> phase_prefix_checksums;
  phase_prefix_checksums.fill(UINT64_C(14695981039346656037));
#endif
  std::uint64_t assembly_checksum = UINT64_C(14695981039346656037);
  std::array<std::uint64_t, 3> preparation_checksums;
  std::array<std::uint64_t, 3> combined_checksums;
  std::array<std::uint64_t, 3> full_checksums;
  preparation_checksums.fill(UINT64_C(14695981039346656037));
  combined_checksums.fill(UINT64_C(14695981039346656037));
  full_checksums.fill(UINT64_C(14695981039346656037));

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    MeasureNnueRouterPhaseAssembly<RouterPhaseAssemblyOperation::Router>(
        corpus, assembly_checksum);
    router_assembly.Add(
        MeasureNnueRouterPhaseAssembly<RouterPhaseAssemblyOperation::Router>(
            corpus, assembly_checksum));
    MeasureNnueRouterPhaseAssembly<RouterPhaseAssemblyOperation::Phase>(
        corpus, assembly_checksum);
    phase_assembly.Add(
        MeasureNnueRouterPhaseAssembly<RouterPhaseAssemblyOperation::Phase>(
            corpus, assembly_checksum));

#if defined(NNUE_COMPACT_PHASE5)
    if ((repeat & 1) == 0) {
      phase_prefix[0].Add(MeasureNnuePhasePrefixAfterWarmup<6>(
          corpus, phase_prefix_checksums[0]));
      phase_prefix[1].Add(MeasureNnuePhasePrefixAfterWarmup<5>(
          corpus, phase_prefix_checksums[1]));
    } else {
      phase_prefix[1].Add(MeasureNnuePhasePrefixAfterWarmup<5>(
          corpus, phase_prefix_checksums[1]));
      phase_prefix[0].Add(MeasureNnuePhasePrefixAfterWarmup<6>(
          corpus, phase_prefix_checksums[0]));
    }
#endif

    for (std::size_t offset = 0; offset < 3; ++offset) {
      const std::size_t candidate = (repeat + offset) % 3;
      if (candidate == 0) {
        preparation[0].Add(MeasureNnueRouterPhasePreparationAfterWarmup<
            RouterPhaseInputImplementation::Separate>(
                corpus, preparation_checksums[0]));
        combined[0].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Separate, false>(
                corpus, combined_checksums[0]));
        full[0].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Separate, true>(
                corpus, full_checksums[0]));
      } else if (candidate == 1) {
        preparation[1].Add(MeasureNnueRouterPhasePreparationAfterWarmup<
            RouterPhaseInputImplementation::Copy>(
                corpus, preparation_checksums[1]));
        combined[1].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Copy, false>(
                corpus, combined_checksums[1]));
        full[1].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Copy, true>(
                corpus, full_checksums[1]));
      } else {
        preparation[2].Add(MeasureNnueRouterPhasePreparationAfterWarmup<
            RouterPhaseInputImplementation::Shared>(
                corpus, preparation_checksums[2]));
        combined[2].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Shared, false>(
                corpus, combined_checksums[2]));
        full[2].Add(MeasureNnueRouterPhaseCandidateAfterWarmup<
            RouterPhaseInputImplementation::Shared, true>(
                corpus, full_checksums[2]));
      }
    }
  }

  PrintNnueBenchSamples("Router input assembly", router_assembly);
  PrintNnueBenchSamples("Phase input assembly", phase_assembly);
#if defined(NNUE_COMPACT_PHASE5)
  std::cout << "Phase projection cleanup A/B" << std::endl;
  PrintNnueBenchSamples("A. legacy Prefix<6>", phase_prefix[0]);
  PrintNnueBenchSamples("B. Phase5 Prefix<5>", phase_prefix[1]);
  const auto prefix6_summary = SummarizeNnueBenchSamples(phase_prefix[0]);
  const auto prefix5_summary = SummarizeNnueBenchSamples(phase_prefix[1]);
  const double prefix_improvement = prefix6_summary.median == 0.0 ? 0.0
      : (prefix6_summary.median - prefix5_summary.median) * 100.0
          / prefix6_summary.median;
  std::cout << "  improvement ns/call : " << std::fixed
            << std::setprecision(1)
            << prefix6_summary.median - prefix5_summary.median << std::endl
            << "  improvement         : " << std::setprecision(2)
            << prefix_improvement << "%" << std::endl
            << "  live-channel checksum match: "
            << (phase_prefix_checksums[0] == phase_prefix_checksums[1]
                    ? "yes" : "NO")
            << std::endl;
#endif
  PrintNnueRouterPhaseComparison(
      "combined Router + Phase input preparation", preparation);
  PrintNnueRouterPhaseComparison(
      "Router prefix + Phase prefix end-to-end", combined);
  PrintNnueRouterPhaseComparison("full staged Network", full);

  std::array<std::uint64_t, 3> router_checksums;
  std::array<std::uint64_t, 3> phase_checksums;
  std::array<std::uint64_t, 3> final_checksums;
  router_checksums.fill(UINT64_C(14695981039346656037));
  phase_checksums.fill(UINT64_C(14695981039346656037));
  final_checksums.fill(UINT64_C(14695981039346656037));
  std::array<std::uint64_t, 3> router_mismatches{};
  std::array<std::uint64_t, 3> phase_mismatches{};
  std::array<std::uint64_t, 3> final_mismatches{};
  std::uint64_t source_mismatches = 0;
  std::uint64_t normal_final_checksum = UINT64_C(14695981039346656037);

  alignas(kCacheLineSize) std::uint8_t common[384];
  alignas(kCacheLineSize) std::uint8_t phase[384];
  alignas(kCacheLineSize) std::int32_t router_out[32];
  alignas(kCacheLineSize) std::int32_t phase_out[32];
  alignas(kCacheLineSize) Network::Buffer work{};
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    AssembleNnueRouterPhaseCommonInput(sample.input, common);
    for (int i = 0; i < 384; ++i)
      source_mismatches += common[i] != sample.input.router_input[i];
    MixNnueBenchChecksum(normal_final_checksum, sample.final_output);
    alignas(kCacheLineSize) std::int32_t captured_router[32];
    router->PropagatePrefix<12>(sample.input.router_input.data(),
                                captured_router);

    for (int candidate = 0; candidate < 3; ++candidate) {
      const int selected_bucket = candidate == 0
          ? PrepareNnueRouterAndPhaseInput<
                RouterPhaseInputImplementation::Separate>(
                sample.input, common, phase, router_out)
          : candidate == 1
          ? PrepareNnueRouterAndPhaseInput<
                RouterPhaseInputImplementation::Copy>(
                sample.input, common, phase, router_out)
          : PrepareNnueRouterAndPhaseInput<
                RouterPhaseInputImplementation::Shared>(
                sample.input, common, common, router_out);
      const std::uint8_t* candidate_phase = candidate == 2 ? common : phase;
      NnueBenchSelectedNetwork(selected_bucket).BenchmarkPhaseProjection(
          candidate_phase, phase_out);
      for (int i = 0; i < 12; ++i) {
        MixNnueBenchChecksum(router_checksums[candidate], router_out[i]);
        router_mismatches[candidate] +=
            router_out[i] != captured_router[i];
      }

      for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
        MixNnueBenchChecksum(phase_checksums[candidate], phase_out[i]);
        phase_mismatches[candidate] +=
            phase_out[i] != sample.intermediate.phase_out[i];
      }
      const std::int32_t output =
          ComputeNnueNetworkStagedOutputFromPhaseInput<false>(
              sample, selected_bucket, candidate_phase, work);
      MixNnueBenchChecksum(final_checksums[candidate], output);
      final_mismatches[candidate] += output != sample.final_output;
    }
  }

  std::cout << "[correctness]" << std::endl
            << "  common input vs captured Router input mismatches: "
            << source_mismatches << std::endl;
  constexpr std::array<const char*, 3> labels = {"A", "B", "C"};
  for (std::size_t candidate = 0; candidate < 3; ++candidate) {
    std::cout << "  " << labels[candidate] << " Router output checksum: 0x"
              << std::hex << router_checksums[candidate] << std::dec
              << ", mismatches: " << router_mismatches[candidate]
              << std::endl
              << "  " << labels[candidate] << " Phase output checksum : 0x"
              << std::hex << phase_checksums[candidate] << std::dec
              << ", mismatches: " << phase_mismatches[candidate]
              << std::endl
              << "  " << labels[candidate] << " final output checksum : 0x"
              << std::hex << final_checksums[candidate] << std::dec
              << ", mismatches: " << final_mismatches[candidate]
              << std::endl;
  }
  std::cout << "  normal final checksum: 0x" << std::hex
            << normal_final_checksum << std::dec << std::endl
            << "  A/B/C timing checksum match: "
            << (preparation_checksums[0] == preparation_checksums[1]
                    && preparation_checksums[0] == preparation_checksums[2]
                    && combined_checksums[0] == combined_checksums[1]
                    && combined_checksums[0] == combined_checksums[2]
                    && full_checksums[0] == full_checksums[1]
                    && full_checksums[0] == full_checksums[2]
                ? "yes" : "NO")
            << std::endl
            << "  note: combined timing includes Router/Phase prefix calls so"
            << std::endl
            << "        C's pre-overwrite byte is observably consumed by Router."
            << std::endl;
}

enum class PhaseL2FixedCandidate {
  CurrentFloat,
  FloatToQ23,
  FixedWidth256,
  FixedWidth128,
  FixedWidth64,
  FixedWidth32,
};

enum class PhaseL2FixedOperation {
  PhaseSigmoidValue,
  ScaleGeneration,
  L2Assembly,
  CombinedPhaseL2,
  FullStagedNetwork,
};

constexpr std::int32_t kPhaseFixedRawMin = -65536;
constexpr std::int32_t kPhaseFixedRawMax = 65536;

template<std::int32_t RawStep>
const auto& NnueBenchPhaseSigmoidQ15Lut() {
  constexpr std::size_t kLutSize =
      (kPhaseFixedRawMax - kPhaseFixedRawMin) / RawStep + 1;
  static const auto lut = [] {
    std::array<std::uint16_t, kLutSize> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::int32_t raw = kPhaseFixedRawMin
          + static_cast<std::int32_t>(i) * RawStep;
      const float logit = (static_cast<float>(raw) / 8128.0f) * 3.0f + 1.0f;
      const float sigmoid = 1.0f / (1.0f + std::exp(-logit));
      values[i] = static_cast<std::uint16_t>(std::clamp(
          static_cast<int>(std::lround(sigmoid * 32768.0f)), 0, 32768));
    }
    return values;
  }();
  return lut;
}

template<std::int32_t RawStep>
void NnueBenchPhaseFixedSigmoidQ15(const std::int32_t* phase_output,
                                   std::uint16_t* sigmoid_q15) {
  const auto& lut = NnueBenchPhaseSigmoidQ15Lut<RawStep>();
  for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
    const std::int32_t raw = phase_output[i];
    if (raw <= kPhaseFixedRawMin)
      sigmoid_q15[i] = lut.front();
    else if (raw >= kPhaseFixedRawMax)
      sigmoid_q15[i] = lut.back();
    else {
      const std::int32_t index =
          (raw - kPhaseFixedRawMin + RawStep / 2) / RawStep;
      sigmoid_q15[i] = lut[static_cast<std::size_t>(index)];
    }
  }
}

void NnueBenchPhaseFixedScalesQ23(const std::uint16_t* sigmoid_q15,
                                  std::int32_t* scales_q23) {
  // scale = factor * (0.55 + 0.45 * sigmoid).  Both terms are kept in Q15;
  // Q15*Q15 is then rounded directly to Q23.  The largest intermediate is
  // below 1.5 * 32768^2, hence it remains within signed int32.
  constexpr std::int32_t kBaseQ15 = 18022;  // round(0.55 * 32768)
  constexpr std::int32_t kGainQ15 = 14746;  // round(0.45 * 32768)
#if defined(NNUE_COMPACT_PHASE5)
  constexpr std::array<std::int32_t, PHASE_OUTPUT_SIZE> kFactorQ15 = {
      42598, 49152, 32768, 22938, 49152};
#else
  constexpr std::array<std::int32_t, PHASE_OUTPUT_SIZE> kFactorQ15 = {
      42598, 49152, 32768, 22938, 28836, 49152};
#endif
  for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
    const std::int32_t base_q15 = kBaseQ15
        + (static_cast<std::int32_t>(sigmoid_q15[i]) * kGainQ15 + (1 << 14))
            / (1 << 15);
    scales_q23[i] =
        (base_q15 * kFactorQ15[i] + 64) >> 7;
  }
}

void NnueBenchPhaseFloatScalesQ23(
    const Network::BenchmarkPhaseScales& scales, std::int32_t* scales_q23) {
  // Multiplication by 2^23 is exact for the finite float scales produced by
  // the current Phase path.  This preserves the historical B candidate:
  // current float sigmoid/scales followed by integer-only L2 assembly.
  constexpr float kQ23 = 8388608.0f;
  scales_q23[0] = static_cast<std::int32_t>(scales.main_sqr * kQ23);
  scales_q23[1] = static_cast<std::int32_t>(scales.main_raw * kQ23);
  scales_q23[2] = static_cast<std::int32_t>(scales.diff * kQ23);
  scales_q23[3] = static_cast<std::int32_t>(scales.abs_raw * kQ23);
#if !defined(NNUE_COMPACT_PHASE5)
  scales_q23[4] = static_cast<std::int32_t>(scales.abs_sqr * kQ23);
#endif
  scales_q23[PHASE_CROSS_INDEX] =
      static_cast<std::int32_t>(scales.cross * kQ23);
}

struct PhaseL2FixedBenchAux {
  std::array<std::int32_t, PHASE_OUTPUT_SIZE> float_scales_q23{};
  std::array<std::array<std::uint16_t, PHASE_OUTPUT_SIZE>, 4>
      fixed_sigmoid_q15{};
  std::array<std::array<std::int32_t, PHASE_OUTPUT_SIZE>, 4>
      fixed_scales_q23{};
};

std::vector<PhaseL2FixedBenchAux> MakePhaseL2FixedBenchAux(
    const std::vector<NetworkStageBenchCase>& corpus) {
  std::vector<PhaseL2FixedBenchAux> auxiliary(corpus.size());
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    NnueBenchPhaseFloatScalesQ23(
        corpus[i].phase_scales, auxiliary[i].float_scales_q23.data());
    NnueBenchPhaseFixedSigmoidQ15<256>(
        corpus[i].intermediate.phase_out, auxiliary[i].fixed_sigmoid_q15[0].data());
    NnueBenchPhaseFixedSigmoidQ15<128>(
        corpus[i].intermediate.phase_out, auxiliary[i].fixed_sigmoid_q15[1].data());
    NnueBenchPhaseFixedSigmoidQ15<64>(
        corpus[i].intermediate.phase_out, auxiliary[i].fixed_sigmoid_q15[2].data());
    NnueBenchPhaseFixedSigmoidQ15<32>(
        corpus[i].intermediate.phase_out, auxiliary[i].fixed_sigmoid_q15[3].data());
    for (std::size_t width = 0; width < 4; ++width)
      NnueBenchPhaseFixedScalesQ23(
          auxiliary[i].fixed_sigmoid_q15[width].data(),
          auxiliary[i].fixed_scales_q23[width].data());
  }
  return auxiliary;
}

template<PhaseL2FixedCandidate Candidate>
constexpr std::int32_t NnueBenchPhaseFixedRawStep() {
  if constexpr (Candidate == PhaseL2FixedCandidate::FixedWidth256)
    return 256;
  else if constexpr (Candidate == PhaseL2FixedCandidate::FixedWidth128)
    return 128;
  else if constexpr (Candidate == PhaseL2FixedCandidate::FixedWidth64)
    return 64;
  else
    return 32;
}

template<PhaseL2FixedCandidate Candidate>
constexpr std::size_t NnueBenchPhaseFixedAuxIndex() {
  return static_cast<std::size_t>(Candidate) - 2;
}

template<PhaseL2FixedCandidate Candidate>
void NnueBenchComputePhaseScales(
    const Network& selected_network, const std::int32_t* phase_output,
    Network::BenchmarkPhaseScales& float_scales,
    std::int32_t* scales_q23) {
  if constexpr (Candidate == PhaseL2FixedCandidate::CurrentFloat) {
    float_scales = selected_network.BenchmarkPhaseScalesFromOutput(phase_output);
  } else if constexpr (Candidate == PhaseL2FixedCandidate::FloatToQ23) {
    float_scales = selected_network.BenchmarkPhaseScalesFromOutput(phase_output);
    NnueBenchPhaseFloatScalesQ23(float_scales, scales_q23);
  } else {
    std::uint16_t sigmoid_q15[PHASE_OUTPUT_SIZE];
    NnueBenchPhaseFixedSigmoidQ15<NnueBenchPhaseFixedRawStep<Candidate>()>(
        phase_output, sigmoid_q15);
    NnueBenchPhaseFixedScalesQ23(sigmoid_q15, scales_q23);
  }
}

template<PhaseL2FixedCandidate Candidate>
void NnueBenchAssembleCandidateL2(
    const Network& selected_network, const Network::Buffer& source,
    const Network::BenchmarkPhaseScales& float_scales,
    const std::int32_t* scales_q23, std::uint8_t* output) {
  if constexpr (Candidate == PhaseL2FixedCandidate::CurrentFloat) {
    selected_network.BenchmarkL2Assembly(
        source.ac_sqr_0_out_temp, source.ac_0_out, source.diff_ac_out,
        source.abs_ac_out, source.abs_sqr_out, source.cross_feat,
        float_scales, output);
  } else {
    selected_network.BenchmarkL2AssemblyQ23(
        source.ac_sqr_0_out_temp, source.ac_0_out, source.diff_ac_out,
        source.abs_ac_out, source.abs_sqr_out, source.cross_feat,
        scales_q23, output);
  }
}

template<PhaseL2FixedCandidate Candidate>
std::int32_t ComputeNnueNetworkPhaseL2Candidate(
    const NetworkStageBenchCase& sample, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(sample.input.selected_bucket);
  selected_network.BenchmarkPhaseInputAssembly(
      sample.input.transformed.data(), sample.input.diff_transformed.data(),
      sample.input.abs_transformed.data(), sample.input.material_bucket,
      work.phase_input);
  selected_network.BenchmarkPhaseProjection(work.phase_input, work.phase_out);
  Network::BenchmarkPhaseScales float_scales{};
  std::int32_t scales_q23[PHASE_OUTPUT_SIZE]{};
  NnueBenchComputePhaseScales<Candidate>(
      selected_network, work.phase_out, float_scales, scales_q23);
  selected_network.BenchmarkFmAffine(
      sample.input.diff_transformed.data(), sample.input.abs_transformed.data(),
      work.diff_fc_out, work.abs_fc_out);
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
  NnueBenchAssembleCandidateL2<Candidate>(
      selected_network, work, float_scales, scales_q23, work.l2_input);
  selected_network.BenchmarkFc1(work.l2_input, work.fc_1_out);
  selected_network.BenchmarkAc1(work.fc_1_out, work.ac_1_out);
  selected_network.BenchmarkFc2(work.ac_1_out, work.fc_2_out);
  return selected_network.BenchmarkBlend(work.fc_0_out[31], work.fc_2_out[0]);
}

template<PhaseL2FixedCandidate Candidate, PhaseL2FixedOperation Operation>
NnueBenchTiming MeasurePhaseL2FixedCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    const std::vector<PhaseL2FixedBenchAux>& auxiliary,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const auto& aux = auxiliary[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    std::uint64_t representative = 0;
    if constexpr (Operation == PhaseL2FixedOperation::PhaseSigmoidValue) {
      if constexpr (Candidate != PhaseL2FixedCandidate::CurrentFloat
                    && Candidate != PhaseL2FixedCandidate::FloatToQ23) {
        std::uint16_t values[PHASE_OUTPUT_SIZE];
        NnueBenchPhaseFixedSigmoidQ15<
            NnueBenchPhaseFixedRawStep<Candidate>()>(
                sample.intermediate.phase_out, values);
        KeepNnueBenchObject(values);
        representative = values[timing.calls % PHASE_OUTPUT_SIZE];
      } else {
        float values[PHASE_OUTPUT_SIZE];
        selected_network.BenchmarkPhaseSigmoid(
            sample.intermediate.phase_out, values);
        KeepNnueBenchObject(values);
        representative =
            NnueBenchFloatBits(values[timing.calls % PHASE_OUTPUT_SIZE]);
      }
    } else if constexpr (Operation == PhaseL2FixedOperation::ScaleGeneration) {
      if constexpr (Candidate == PhaseL2FixedCandidate::CurrentFloat) {
        const auto scales = selected_network.BenchmarkPhaseChannelScales(
            sample.phase_values.data());
        KeepNnueBenchObject(scales);
        const IndexType channel = timing.calls % PHASE_OUTPUT_SIZE;
        const IndexType semantic_channel =
            PHASE_OUTPUT_SIZE == 5 && channel == PHASE_CROSS_INDEX ? 5 : channel;
        representative = NnueBenchFloatBits(
            NnueBenchPhaseScalesToArray(scales)[semantic_channel]);
      } else if constexpr (Candidate == PhaseL2FixedCandidate::FloatToQ23) {
        const auto scales = selected_network.BenchmarkPhaseChannelScales(
            sample.phase_values.data());
        std::int32_t q23[PHASE_OUTPUT_SIZE];
        NnueBenchPhaseFloatScalesQ23(scales, q23);
        KeepNnueBenchObject(q23);
        representative = static_cast<std::uint32_t>(
            q23[timing.calls % PHASE_OUTPUT_SIZE]);
      } else {
        std::int32_t q23[PHASE_OUTPUT_SIZE];
        NnueBenchPhaseFixedScalesQ23(
            aux.fixed_sigmoid_q15[NnueBenchPhaseFixedAuxIndex<Candidate>()].data(),
            q23);
        KeepNnueBenchObject(q23);
        representative = static_cast<std::uint32_t>(
            q23[timing.calls % PHASE_OUTPUT_SIZE]);
      }
    } else if constexpr (Operation == PhaseL2FixedOperation::L2Assembly) {
      if constexpr (Candidate == PhaseL2FixedCandidate::CurrentFloat) {
        selected_network.BenchmarkL2Assembly(
            sample.intermediate.ac_sqr_0_out_temp,
            sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
            sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
            sample.intermediate.cross_feat, sample.phase_scales,
            work.l2_input);
      } else {
        const std::int32_t* q23;
        if constexpr (Candidate == PhaseL2FixedCandidate::FloatToQ23)
          q23 = aux.float_scales_q23.data();
        else
          q23 = aux.fixed_scales_q23[
              NnueBenchPhaseFixedAuxIndex<Candidate>()].data();
        selected_network.BenchmarkL2AssemblyQ23(
            sample.intermediate.ac_sqr_0_out_temp,
            sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
            sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
            sample.intermediate.cross_feat, q23, work.l2_input);
      }
      KeepNnueBenchObject(work.l2_input);
      representative = work.l2_input[timing.calls % L2_INPUT_SIZE];
    } else if constexpr (Operation == PhaseL2FixedOperation::CombinedPhaseL2) {
      Network::BenchmarkPhaseScales float_scales{};
      std::int32_t q23[PHASE_OUTPUT_SIZE]{};
      NnueBenchComputePhaseScales<Candidate>(
          selected_network, sample.intermediate.phase_out, float_scales, q23);
      NnueBenchAssembleCandidateL2<Candidate>(
          selected_network, sample.intermediate, float_scales, q23,
          work.l2_input);
      KeepNnueBenchObject(work.l2_input);
      representative = work.l2_input[timing.calls % L2_INPUT_SIZE];
    } else {
      const std::int32_t output =
          ComputeNnueNetworkPhaseL2Candidate<Candidate>(sample, work);
      KeepNnueBenchObject(work);
      representative = static_cast<std::uint32_t>(output);
    }
    MixNnueBenchChecksum(checksum, representative);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<PhaseL2FixedCandidate Candidate, PhaseL2FixedOperation Operation>
NnueBenchTiming MeasurePhaseL2FixedCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    const std::vector<PhaseL2FixedBenchAux>& auxiliary,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasurePhaseL2FixedCorpus<Candidate, Operation>(
      corpus, auxiliary, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasurePhaseL2FixedCorpus<Candidate, Operation>(
      corpus, auxiliary, checksum);
}

template<PhaseL2FixedCandidate Candidate>
void AddPhaseL2FixedCandidateSamples(
    const std::vector<NetworkStageBenchCase>& corpus,
    const std::vector<PhaseL2FixedBenchAux>& auxiliary,
    std::array<NnueBenchSamples, 5>& samples,
    std::array<std::uint64_t, 5>& checksums) {
  samples[0].Add(MeasurePhaseL2FixedCorpusAfterWarmup<
      Candidate, PhaseL2FixedOperation::PhaseSigmoidValue>(
          corpus, auxiliary, checksums[0]));
  samples[1].Add(MeasurePhaseL2FixedCorpusAfterWarmup<
      Candidate, PhaseL2FixedOperation::ScaleGeneration>(
          corpus, auxiliary, checksums[1]));
  samples[2].Add(MeasurePhaseL2FixedCorpusAfterWarmup<
      Candidate, PhaseL2FixedOperation::L2Assembly>(
          corpus, auxiliary, checksums[2]));
  samples[3].Add(MeasurePhaseL2FixedCorpusAfterWarmup<
      Candidate, PhaseL2FixedOperation::CombinedPhaseL2>(
          corpus, auxiliary, checksums[3]));
  samples[4].Add(MeasurePhaseL2FixedCorpusAfterWarmup<
      Candidate, PhaseL2FixedOperation::FullStagedNetwork>(
          corpus, auxiliary, checksums[4]));
}

double NnueBenchPhaseL2Percentile(const std::vector<double>& sorted,
                                  const double percentile) {
  if (sorted.empty())
    return 0.0;
  const std::size_t index = static_cast<std::size_t>(std::ceil(
      percentile * static_cast<double>(sorted.size()))) - 1;
  return sorted[std::min(index, sorted.size() - 1)];
}

template<bool UseFixedPhaseL2>
NnueBenchTiming MeasureProductionPhaseL2NetworkCorpus(
    const std::vector<NetworkBenchCase>& corpus, std::uint64_t& checksum) {
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;
  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    const auto output = selected_network.Propagate<true, false,
        UseFixedPhaseL2>(
            sample.transformed.data(), sample.diff_transformed.data(),
            sample.abs_transformed.data(), sample.material_bucket,
            reinterpret_cast<char*>(&work));
    MixNnueBenchChecksum(checksum, output[0]);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<bool UseFixedPhaseL2>
NnueBenchTiming MeasureProductionPhaseL2NetworkCorpusAfterWarmup(
    const std::vector<NetworkBenchCase>& corpus, std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureProductionPhaseL2NetworkCorpus<UseFixedPhaseL2>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureProductionPhaseL2NetworkCorpus<UseFixedPhaseL2>(
      corpus, checksum);
}

double NnueBenchSampleStandardDeviation(const NnueBenchSamples& samples) {
  if (samples.ns_per_call.size() < 2)
    return 0.0;
  const auto summary = SummarizeNnueBenchSamples(samples);
  double sum = 0.0;
  for (const double value : samples.ns_per_call) {
    const double difference = value - summary.mean;
    sum += difference * difference;
  }
  return std::sqrt(sum /
      static_cast<double>(samples.ns_per_call.size() - 1));
}

void TestPhaseL2FixedBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Phase scale + L2 fixed-point]" << std::endl
            << "  architecture : Phase" << PHASE_OUTPUT_SIZE
            << " / L2 " << L2_INPUT_SIZE << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : A/B/C32 rotated per repeat"
            << std::endl
            << "  fixed L2     : uint8 * Q23, shift 23, clamp [0,127]"
            << std::endl
            << "  Phase LUT    : Q15, raw [-65536,65536], nearest, no interpolation"
            << std::endl
            << "  C32 LUT bytes: 8194"
            << std::endl;
  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: NNUE Phase/L2 benchmark corpus is empty" << std::endl;
    return;
  }
  const auto auxiliary = MakePhaseL2FixedBenchAux(corpus);
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  L2 values    : " << corpus.size() * L2_INPUT_SIZE << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  std::array<std::array<NnueBenchSamples, 5>, 3> samples;
  std::array<std::array<std::uint64_t, 5>, 3> timing_checksums;
  for (auto& candidate : timing_checksums)
    candidate.fill(UINT64_C(14695981039346656037));
  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      const std::size_t candidate = (repeat + offset) % 3;
      if (candidate == 0)
        AddPhaseL2FixedCandidateSamples<PhaseL2FixedCandidate::CurrentFloat>(
            corpus, auxiliary, samples[0], timing_checksums[0]);
      else if (candidate == 1)
        AddPhaseL2FixedCandidateSamples<
            PhaseL2FixedCandidate::FloatToQ23>(
                corpus, auxiliary, samples[1], timing_checksums[1]);
      else
        AddPhaseL2FixedCandidateSamples<
            PhaseL2FixedCandidate::FixedWidth32>(
                corpus, auxiliary, samples[2], timing_checksums[2]);
    }
  }

  constexpr std::array<const char*, 3> candidate_names = {
      "A. current float Phase + float L2",
      "B. current float Phase -> Q23 + integer L2",
      "C32. Q15 nearest width 32 + Q23 L2"};
  constexpr std::array<const char*, 5> operation_names = {
      "phase sigmoid / value", "scale generation", "L2 assembly",
      "combined Phase scale + L2", "full staged Network"};
  for (std::size_t operation = 0; operation < operation_names.size();
       ++operation) {
    std::cout << operation_names[operation] << std::endl;
    const auto baseline = SummarizeNnueBenchSamples(samples[0][operation]);
    for (std::size_t candidate = 0; candidate < 3; ++candidate) {
      PrintNnueBenchSamples(candidate_names[candidate],
                            samples[candidate][operation]);
      const auto summary =
          SummarizeNnueBenchSamples(samples[candidate][operation]);
      const double improvement = baseline.median == 0.0 ? 0.0
          : (baseline.median - summary.median) * 100.0 / baseline.median;
      std::cout << "  improvement vs A : " << std::fixed
                << std::setprecision(2) << improvement << "%" << std::endl
                << "  timing checksum   : 0x" << std::hex
                << timing_checksums[candidate][operation] << std::dec
                << std::endl;
    }
  }

  struct Accuracy {
    std::uint64_t l2_mismatches = 0;
    int l2_max_diff = 0;
    std::uint64_t final_mismatches = 0;
    std::int64_t final_max_raw_diff = 0;
    std::uint64_t sign_flips = 0;
    std::uint64_t l2_checksum = UINT64_C(14695981039346656037);
    std::uint64_t final_checksum = UINT64_C(14695981039346656037);
    double cp_sum = 0.0;
    std::vector<double> cp_differences;
  };
  std::array<Accuracy, 3> accuracy;
  for (auto& value : accuracy)
    value.cp_differences.reserve(corpus.size());
  std::vector<std::int32_t> phase_raw;
  phase_raw.reserve(corpus.size() * PHASE_OUTPUT_SIZE);
  float scale_min = std::numeric_limits<float>::max();
  float scale_max = std::numeric_limits<float>::lowest();
  std::array<float, PHASE_OUTPUT_SIZE> channel_scale_min;
  std::array<float, PHASE_OUTPUT_SIZE> channel_scale_max;
  channel_scale_min.fill(std::numeric_limits<float>::max());
  channel_scale_max.fill(std::numeric_limits<float>::lowest());

  alignas(kCacheLineSize) Network::Buffer work{};
  alignas(kCacheLineSize) std::uint8_t l2[L2_INPUT_SIZE];
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    const auto float_scale_values =
        NnueBenchPhaseScalesToArray(sample.phase_scales);
    for (IndexType channel = 0; channel < PHASE_OUTPUT_SIZE; ++channel) {
      phase_raw.push_back(sample.intermediate.phase_out[channel]);
      const IndexType semantic_channel =
          PHASE_OUTPUT_SIZE == 5 && channel == PHASE_CROSS_INDEX ? 5 : channel;
      scale_min = std::min(scale_min, float_scale_values[semantic_channel]);
      scale_max = std::max(scale_max, float_scale_values[semantic_channel]);
      channel_scale_min[channel] =
          std::min(channel_scale_min[channel],
                   float_scale_values[semantic_channel]);
      channel_scale_max[channel] =
          std::max(channel_scale_max[channel],
                   float_scale_values[semantic_channel]);
    }

    for (std::size_t candidate = 0; candidate < 3; ++candidate) {
      std::int32_t output;
      if (candidate == 0) {
        selected_network.BenchmarkL2Assembly(
            sample.intermediate.ac_sqr_0_out_temp,
            sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
            sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
            sample.intermediate.cross_feat, sample.phase_scales, l2);
        output = ComputeNnueNetworkPhaseL2Candidate<
            PhaseL2FixedCandidate::CurrentFloat>(sample, work);
      } else {
        const std::int32_t* q23 = candidate == 1
            ? auxiliary[sample_index].float_scales_q23.data()
            : auxiliary[sample_index].fixed_scales_q23[3].data();
        selected_network.BenchmarkL2AssemblyQ23(
            sample.intermediate.ac_sqr_0_out_temp,
            sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
            sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
            sample.intermediate.cross_feat, q23, l2);
        if (candidate == 1)
          output = ComputeNnueNetworkPhaseL2Candidate<
              PhaseL2FixedCandidate::FloatToQ23>(sample, work);
        else
          output = ComputeNnueNetworkPhaseL2Candidate<
              PhaseL2FixedCandidate::FixedWidth32>(sample, work);
      }
      for (IndexType i = 0; i < L2_INPUT_SIZE; ++i) {
        MixNnueBenchChecksum(accuracy[candidate].l2_checksum, l2[i]);
        const int difference = std::abs(
            static_cast<int>(l2[i])
            - static_cast<int>(sample.intermediate.l2_input[i]));
        accuracy[candidate].l2_mismatches += difference != 0;
        accuracy[candidate].l2_max_diff =
            std::max(accuracy[candidate].l2_max_diff, difference);
      }
      MixNnueBenchChecksum(accuracy[candidate].final_checksum, output);
      const std::int64_t raw_difference = std::abs(
          static_cast<std::int64_t>(output) - sample.final_output);
      accuracy[candidate].final_mismatches += raw_difference != 0;
      accuracy[candidate].final_max_raw_diff =
          std::max(accuracy[candidate].final_max_raw_diff, raw_difference);
      accuracy[candidate].sign_flips +=
          (output < 0) != (sample.final_output < 0);
      const double cp_difference =
          static_cast<double>(raw_difference) / static_cast<double>(FV_SCALE);
      accuracy[candidate].cp_sum += cp_difference;
      accuracy[candidate].cp_differences.push_back(cp_difference);
    }
  }

  std::sort(phase_raw.begin(), phase_raw.end());
  auto raw_percentile = [&](const double p) {
    const std::size_t index = static_cast<std::size_t>(std::ceil(
        p * static_cast<double>(phase_raw.size()))) - 1;
    return phase_raw[std::min(index, phase_raw.size() - 1)];
  };
  constexpr std::array<double, 7> percentiles = {
      0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0};
  constexpr std::array<const char*, 7> percentile_names = {
      "min", "p1", "p10", "median", "p90", "p99", "max"};
  std::cout << "[phase_out / logit distribution, all "
            << PHASE_OUTPUT_SIZE << " channels]" << std::endl;
  for (std::size_t i = 0; i < percentiles.size(); ++i) {
    const std::int32_t raw = percentiles[i] == 0.0 ? phase_raw.front()
        : percentiles[i] == 1.0 ? phase_raw.back()
        : raw_percentile(percentiles[i]);
    const double logit = (static_cast<double>(raw) / 8128.0) * 3.0 + 1.0;
    std::cout << "  " << std::setw(6) << percentile_names[i]
              << " raw=" << std::setw(9) << raw
              << " logit=" << std::fixed << std::setprecision(6) << logit
              << std::endl;
  }
  std::cout << "  current float scale range: [" << scale_min << ", "
            << scale_max << "]" << std::endl;
  for (IndexType channel = 0; channel < PHASE_OUTPUT_SIZE; ++channel)
    std::cout << "  channel " << channel << " scale range: ["
              << channel_scale_min[channel] << ", "
              << channel_scale_max[channel] << "]" << std::endl;

  std::cout << "[accuracy vs current production]" << std::endl;
  const double l2_total =
      static_cast<double>(corpus.size() * L2_INPUT_SIZE);
  for (std::size_t candidate = 0; candidate < 3; ++candidate) {
    auto& result = accuracy[candidate];
    std::sort(result.cp_differences.begin(), result.cp_differences.end());
    const double final_total = static_cast<double>(corpus.size());
    std::cout << candidate_names[candidate] << std::endl
              << "  L2 checksum           : 0x" << std::hex
              << result.l2_checksum << std::dec << std::endl
              << "  L2 byte mismatch      : " << result.l2_mismatches << " / "
              << static_cast<std::uint64_t>(l2_total) << " ("
              << std::fixed << std::setprecision(6)
              << 100.0 * result.l2_mismatches / l2_total << "%)" << std::endl
              << "  L2 max byte diff      : " << result.l2_max_diff << std::endl
              << "  final checksum        : 0x" << std::hex
              << result.final_checksum << std::dec << std::endl
              << "  final mismatch        : " << result.final_mismatches << " / "
              << corpus.size() << " (" << std::fixed << std::setprecision(6)
              << 100.0 * result.final_mismatches / final_total << "%)"
              << std::endl
              << "  final max raw diff    : " << result.final_max_raw_diff
              << std::endl
              << "  sign flip             : " << result.sign_flips << " / "
              << corpus.size() << " (" << std::fixed << std::setprecision(6)
              << 100.0 * result.sign_flips / final_total << "%)" << std::endl
              << "  eval/cp abs diff mean : "
              << result.cp_sum / final_total << std::endl
              << "  eval/cp abs diff median: "
              << NnueBenchPhaseL2Percentile(result.cp_differences, 0.50)
              << std::endl
              << "  eval/cp abs diff max  : "
              << result.cp_differences.back() << std::endl
              << "  eval/cp abs diff p90  : "
              << NnueBenchPhaseL2Percentile(result.cp_differences, 0.90) << std::endl
              << "  eval/cp abs diff p95  : "
              << NnueBenchPhaseL2Percentile(result.cp_differences, 0.95) << std::endl
              << "  eval/cp abs diff p99  : "
              << NnueBenchPhaseL2Percentile(result.cp_differences, 0.99) << std::endl
              << "  eval/cp abs diff p99.9: "
              << NnueBenchPhaseL2Percentile(result.cp_differences, 0.999) << std::endl;
  }

  std::cout << "[actual Network::Propagate A / C32]" << std::endl;
  const auto production_corpus = MakeNnueNetworkBenchCorpus();
  NnueBenchSamples production_float_samples;
  NnueBenchSamples production_fixed_samples;
  std::uint64_t production_float_timing_checksum =
      UINT64_C(14695981039346656037);
  std::uint64_t production_fixed_timing_checksum =
      UINT64_C(14695981039346656037);
  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      production_float_samples.Add(
          MeasureProductionPhaseL2NetworkCorpusAfterWarmup<false>(
              production_corpus, production_float_timing_checksum));
      production_fixed_samples.Add(
          MeasureProductionPhaseL2NetworkCorpusAfterWarmup<true>(
              production_corpus, production_fixed_timing_checksum));
    } else {
      production_fixed_samples.Add(
          MeasureProductionPhaseL2NetworkCorpusAfterWarmup<true>(
              production_corpus, production_fixed_timing_checksum));
      production_float_samples.Add(
          MeasureProductionPhaseL2NetworkCorpusAfterWarmup<false>(
              production_corpus, production_float_timing_checksum));
    }
  }
  PrintNnueBenchSamples("A. production float", production_float_samples);
  std::cout << "  SD ns/call : "
            << NnueBenchSampleStandardDeviation(production_float_samples)
            << std::endl;
  PrintNnueBenchSamples("C32. production fixed", production_fixed_samples);
  std::cout << "  SD ns/call : "
            << NnueBenchSampleStandardDeviation(production_fixed_samples)
            << std::endl;
  const auto production_float_summary =
      SummarizeNnueBenchSamples(production_float_samples);
  const auto production_fixed_summary =
      SummarizeNnueBenchSamples(production_fixed_samples);
  std::cout << "  median improvement: " << std::fixed << std::setprecision(2)
            << (production_float_summary.median
                    - production_fixed_summary.median)
                * 100.0 / production_float_summary.median
            << "%" << std::endl
            << "  timing checksum match: "
            << (production_float_timing_checksum
                    == production_fixed_timing_checksum
                ? "yes" : "NO (expected for approximate C32)")
            << std::endl;

  std::uint64_t production_l2_mismatches = 0;
  std::uint64_t production_final_mismatches = 0;
  int production_l2_max_diff = 0;
  std::int64_t production_final_max_diff = 0;
  alignas(kCacheLineSize) Network::Buffer float_work{};
  alignas(kCacheLineSize) Network::Buffer fixed_work{};
  for (const auto& sample : production_corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.selected_bucket);
    const auto float_output = selected_network.Propagate<true, false, false>(
        sample.transformed.data(), sample.diff_transformed.data(),
        sample.abs_transformed.data(), sample.material_bucket,
        reinterpret_cast<char*>(&float_work));
    const auto fixed_output = selected_network.Propagate<true, false, true>(
        sample.transformed.data(), sample.diff_transformed.data(),
        sample.abs_transformed.data(), sample.material_bucket,
        reinterpret_cast<char*>(&fixed_work));
    for (IndexType i = 0; i < L2_INPUT_SIZE; ++i) {
      const int difference = std::abs(
          static_cast<int>(float_work.l2_input[i])
          - static_cast<int>(fixed_work.l2_input[i]));
      production_l2_mismatches += difference != 0;
      production_l2_max_diff = std::max(production_l2_max_diff, difference);
    }
    const std::int64_t difference = std::abs(
        static_cast<std::int64_t>(float_output[0]) - fixed_output[0]);
    production_final_mismatches += difference != 0;
    production_final_max_diff =
        std::max(production_final_max_diff, difference);
  }
  std::cout << "  production L2 mismatch       : "
            << production_l2_mismatches << " / "
            << production_corpus.size() * L2_INPUT_SIZE << std::endl
            << "  production L2 max byte diff : "
            << production_l2_max_diff << std::endl
            << "  production final mismatch   : "
            << production_final_mismatches << " / "
            << production_corpus.size() << std::endl
            << "  production max raw diff     : "
            << production_final_max_diff << std::endl
            << "  matches reconstructed C32 counts: "
            << (production_l2_mismatches == accuracy[2].l2_mismatches
                    && production_final_mismatches
                        == accuracy[2].final_mismatches
                    && production_l2_max_diff == accuracy[2].l2_max_diff
                    && production_final_max_diff
                        == accuracy[2].final_max_raw_diff
                ? "yes" : "NO")
            << std::endl;
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

    if constexpr (Operation ==
                  NetworkStageBenchOperation::PhaseInputAssembly) {
      selected_network.BenchmarkPhaseInputAssembly(
          sample.input.transformed.data(),
          sample.input.diff_transformed.data(),
          sample.input.abs_transformed.data(), sample.input.material_bucket,
          work.phase_input);
      representative = work.phase_input[index % 384]
          ^ (static_cast<std::uint64_t>(
                 work.phase_input[(index + 193) % 384]) << 8);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::PhaseProjection) {
      selected_network.BenchmarkPhaseProjection(
          sample.intermediate.phase_input, work.phase_out);
      representative = static_cast<std::uint32_t>(
                           work.phase_out[index % PHASE_OUTPUT_SIZE])
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.phase_out[(index + 3) % PHASE_OUTPUT_SIZE])) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::PhaseSigmoid) {
      float phase_values[PHASE_OUTPUT_SIZE];
      selected_network.BenchmarkPhaseSigmoid(
          sample.intermediate.phase_out, phase_values);
      representative = NnueBenchFloatBits(
                           phase_values[index % PHASE_OUTPUT_SIZE])
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(
                 phase_values[(index + 3) % PHASE_OUTPUT_SIZE])) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::PhaseChannelScales) {
      const auto scales = selected_network.BenchmarkPhaseChannelScales(
          sample.phase_values.data());
      const auto scale_values = NnueBenchPhaseScalesToArray(scales);
      representative = NnueBenchFloatBits(scale_values[index % 6])
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(
                 scale_values[(index + 3) % 6])) << 32);
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
      representative = static_cast<std::uint32_t>(
          work.lca_q_out[index % LCA_QK_SIZE]);
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
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::MainGateSigmoid) {
      selected_network.BenchmarkMainGateSigmoid(
          sample.intermediate.diff_fc_out, work.lca_q_out);
      representative = static_cast<std::uint32_t>(
          work.lca_q_out[index % LCA_QK_SIZE])
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.lca_q_out[(index + 17) % LCA_QK_SIZE])) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::MainGateApply) {
      selected_network.BenchmarkMainGateApply(
          sample.main_before_gate.data(), sample.main_gate_q64.data(),
          work.fc_0_out);
      representative = static_cast<std::uint32_t>(work.fc_0_out[index % 32])
          ^ (static_cast<std::uint64_t>(
                 static_cast<std::uint32_t>(work.fc_0_out[(index + 17) % 32]))
             << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::MainGateClamp) {
      selected_network.BenchmarkMainGateClamp(
          sample.main_after_gate_before_clamp.data(), work.fc_0_out);
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
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaFmInput) {
      selected_network.BenchmarkLcaAssembleFmInput(
          sample.diff_before_lca.data(), sample.intermediate.abs_ac_out,
          work.fm_cat_uint8);
      representative = work.fm_cat_uint8[index % 64]
          ^ (static_cast<std::uint64_t>(
                 work.fm_cat_uint8[(index + 33) % 64]) << 8);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaQuery) {
      selected_network.BenchmarkLcaQuery(
          sample.intermediate.ac_0_out, work.lca_q_out);
      representative = static_cast<std::uint32_t>(
          work.lca_q_out[index % LCA_QK_SIZE]);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaKey) {
      selected_network.BenchmarkLcaKey(
          sample.intermediate.fm_cat_uint8, work.lca_k_out);
      representative = static_cast<std::uint32_t>(
          work.lca_k_out[index % LCA_QK_SIZE]);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaValue) {
      selected_network.BenchmarkLcaValue(
          sample.intermediate.fm_cat_uint8, work.lca_v_out);
      representative = static_cast<std::uint32_t>(
          work.lca_v_out[index % LCA_VALUE_SIZE]);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaDotAndLogit) {
      float dot_product = 0.0f;
      float attention_logit = 0.0f;
      selected_network.BenchmarkLcaDotAndLogit(
          sample.intermediate.lca_q_out, sample.intermediate.lca_k_out,
          &dot_product, &attention_logit);
      representative = NnueBenchFloatBits(dot_product)
          ^ (static_cast<std::uint64_t>(
                 NnueBenchFloatBits(attention_logit)) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::LcaAttentionScore) {
      float attention_score = 0.0f;
      selected_network.BenchmarkLcaAttentionScore(
          sample.lca_attention_logit, &attention_score);
      representative = NnueBenchFloatBits(attention_score);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::
                             LcaValueClampAndCorrection) {
      alignas(kCacheLineSize) float value_clamped[32];
      alignas(kCacheLineSize) float value_correction[32];
      selected_network.BenchmarkLcaValueClampAndCorrection(
          sample.intermediate.lca_v_out, sample.lca_attention_score,
          value_clamped, value_correction);
      representative = NnueBenchFloatBits(value_clamped[index % 32])
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(
                 value_correction[(index + 17) % 32])) << 32);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::
                             LcaFinalAddAndQuantize) {
      alignas(kCacheLineSize) float output_before_narrow[32];
      selected_network.BenchmarkLcaFinalAddAndQuantize(
          sample.diff_before_lca.data(), sample.lca_attention_score,
          sample.lca_value_correction.data(), output_before_narrow,
          work.diff_ac_out);
      representative = work.diff_ac_out[index % 32]
          ^ (static_cast<std::uint64_t>(NnueBenchFloatBits(
                 output_before_narrow[(index + 17) % 32])) << 8);
    } else if constexpr (Operation == NetworkStageBenchOperation::Cross) {
      selected_network.BenchmarkCross(
          sample.intermediate.ac_sqr_0_out_temp,
          sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
          sample.intermediate.abs_ac_out, work.cross_cat,
          work.cross_fc_out, work.cross_feat);
      representative = work.cross_cat[index % 32]
          ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                 work.cross_fc_out[index % CROSS_OUTPUT_SIZE])) << 8)
          ^ (static_cast<std::uint64_t>(
                 work.cross_feat[index % CROSS_OUTPUT_SIZE]) << 48);
    } else if constexpr (Operation ==
                         NetworkStageBenchOperation::L2Assembly) {
#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
      std::int32_t production_scales_q23[PHASE_OUTPUT_SIZE];
      selected_network.BenchmarkPhaseFixedScalesQ23(
          sample.intermediate.phase_out, production_scales_q23);
      selected_network.BenchmarkL2AssemblyQ23(
          sample.intermediate.ac_sqr_0_out_temp,
          sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
          sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
          sample.intermediate.cross_feat, production_scales_q23,
          work.l2_input);
#else
      selected_network.BenchmarkL2Assembly(
          sample.intermediate.ac_sqr_0_out_temp,
          sample.intermediate.ac_0_out, sample.intermediate.diff_ac_out,
          sample.intermediate.abs_ac_out, sample.intermediate.abs_sqr_out,
          sample.intermediate.cross_feat, sample.phase_scales,
          work.l2_input);
#endif
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
    case NetworkStageBenchOperation::PhaseInputAssembly:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::PhaseInputAssembly>(corpus, checksum);
    case NetworkStageBenchOperation::PhaseProjection:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::PhaseProjection>(corpus, checksum);
    case NetworkStageBenchOperation::PhaseSigmoid:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::PhaseSigmoid>(corpus, checksum);
    case NetworkStageBenchOperation::PhaseChannelScales:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::PhaseChannelScales>(corpus, checksum);
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
    case NetworkStageBenchOperation::MainGateSigmoid:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainGateSigmoid>(corpus, checksum);
    case NetworkStageBenchOperation::MainGateApply:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainGateApply>(corpus, checksum);
    case NetworkStageBenchOperation::MainGateClamp:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainGateClamp>(corpus, checksum);
    case NetworkStageBenchOperation::MainSqrClippedRelu:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainSqrClippedRelu>(corpus, checksum);
    case NetworkStageBenchOperation::MainClippedRelu:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::MainClippedRelu>(corpus, checksum);
    case NetworkStageBenchOperation::LcaFmInput:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaFmInput>(corpus, checksum);
    case NetworkStageBenchOperation::LcaQuery:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaQuery>(corpus, checksum);
    case NetworkStageBenchOperation::LcaKey:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaKey>(corpus, checksum);
    case NetworkStageBenchOperation::LcaValue:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaValue>(corpus, checksum);
    case NetworkStageBenchOperation::LcaDotAndLogit:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaDotAndLogit>(corpus, checksum);
    case NetworkStageBenchOperation::LcaAttentionScore:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaAttentionScore>(corpus, checksum);
    case NetworkStageBenchOperation::LcaValueClampAndCorrection:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaValueClampAndCorrection>(
              corpus, checksum);
    case NetworkStageBenchOperation::LcaFinalAddAndQuantize:
      return MeasureNnueNetworkStageCorpus<
          NetworkStageBenchOperation::LcaFinalAddAndQuantize>(
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
    if ((stage >= 2 && stage <= 3) || stage == 6
        || (stage >= 21 && stage <= 24)) {
      if (float_mismatches[stage].count == 0)
        std::cout << "  float bit mismatch count    : 0" << std::endl;
    }
    if (stage == 24)
      std::cout
          << "  note: this stage compares the split benchmark reconstruction;"
          << std::endl
          << "        normal-call-site output differences are reported in the"
          << std::endl
          << "        separate LCA diagnostics below and do not fail this stage."
          << std::endl;
    if (stage == 26) {
      std::cout << "  L2 real range         : [0, " << L2_REAL_SIZE << ")"
                << std::endl
                << "  L2 padding range      : [" << L2_REAL_SIZE << ", "
                << L2_INPUT_SIZE << ")" << std::endl
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

enum class MainGateBenchImplementation {
  Combined,
  Reconstructed,
};

template<MainGateBenchImplementation Implementation>
NnueBenchTiming MeasureMainGateCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    if constexpr (Implementation == MainGateBenchImplementation::Combined)
      selected_network.BenchmarkMainGateCombined(
          sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
          output);
    else
      selected_network.BenchmarkMainGateReconstructed(
          sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
          output);

    const std::size_t index = static_cast<std::size_t>(timing.calls) & 31;
    const std::uint64_t representative =
        static_cast<std::uint32_t>(output[index])
        ^ (static_cast<std::uint64_t>(
               static_cast<std::uint32_t>(output[(index + 17) & 31]))
           << 32);
    MixNnueBenchChecksum(checksum, representative);
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<MainGateBenchImplementation Implementation>
NnueBenchTiming MeasureMainGateCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureMainGateCorpus<Implementation>(corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureMainGateCorpus<Implementation>(corpus, checksum);
}

void TestMainGateBenchmarkCompare(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Main gate combined / reconstructed]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : even=A,B odd=B,A" << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: Main gate benchmark corpus is empty" << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  compared values: " << corpus.size() * 32 << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  NnueBenchSamples combined_samples;
  NnueBenchSamples reconstructed_samples;
  std::uint64_t combined_timing_checksum =
      UINT64_C(14695981039346656037);
  std::uint64_t reconstructed_timing_checksum =
      UINT64_C(14695981039346656037);

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
    if ((repeat & 1) == 0) {
      combined_samples.Add(MeasureMainGateCorpusAfterWarmup<
          MainGateBenchImplementation::Combined>(
              corpus, combined_timing_checksum));
      reconstructed_samples.Add(MeasureMainGateCorpusAfterWarmup<
          MainGateBenchImplementation::Reconstructed>(
              corpus, reconstructed_timing_checksum));
    } else {
      reconstructed_samples.Add(MeasureMainGateCorpusAfterWarmup<
          MainGateBenchImplementation::Reconstructed>(
              corpus, reconstructed_timing_checksum));
      combined_samples.Add(MeasureMainGateCorpusAfterWarmup<
          MainGateBenchImplementation::Combined>(
              corpus, combined_timing_checksum));
    }
  }

  PrintNnueBenchSamples("A. combined normal-order loop", combined_samples);
  PrintNnueBenchSamples("B. reconstructed sigmoid -> apply -> clamp",
                        reconstructed_samples);

  const NnueBenchSummary combined_summary =
      SummarizeNnueBenchSamples(combined_samples);
  const NnueBenchSummary reconstructed_summary =
      SummarizeNnueBenchSamples(reconstructed_samples);
  const double difference =
      reconstructed_summary.median - combined_summary.median;
  const double improvement = combined_summary.median == 0.0
      ? 0.0
      : (combined_summary.median - reconstructed_summary.median)
            * 100.0 / combined_summary.median;

  alignas(kCacheLineSize) std::int32_t combined_output[32];
  alignas(kCacheLineSize) std::int32_t reconstructed_output[32];
  std::uint64_t combined_checksum = UINT64_C(14695981039346656037);
  std::uint64_t reconstructed_checksum = UINT64_C(14695981039346656037);
  NnueNetworkStageMismatch mismatch{};
  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    selected_network.BenchmarkMainGateCombined(
        sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
        combined_output);
    selected_network.BenchmarkMainGateReconstructed(
        sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
        reconstructed_output);
    MixNnueBenchRange(combined_checksum, combined_output, 32);
    MixNnueBenchRange(reconstructed_checksum, reconstructed_output, 32);
    CompareNnueNetworkStageRange(
        combined_output, reconstructed_output, 32, sample_index, mismatch);
  }

  std::cout << "  difference (B - A) : " << std::fixed
            << std::setprecision(1) << difference << " ns/call" << std::endl
            << "  improvement        : " << std::setprecision(2)
            << improvement << "%" << std::endl
            << "  A checksum : 0x" << std::hex << combined_checksum
            << std::endl
            << "  B checksum : 0x" << reconstructed_checksum << std::dec
            << std::endl
            << "  checksum match : "
            << (combined_checksum == reconstructed_checksum ? "yes" : "NO")
            << std::endl
            << "  mismatch count : " << mismatch.count << std::endl;
  if (mismatch.count != 0) {
    std::cout << "  first mismatch sample : " << mismatch.first_sample
              << std::endl
              << "  first mismatch output : " << mismatch.first_element
              << std::endl
              << "  A value               : " << mismatch.first_captured
              << std::endl
              << "  B value               : " << mismatch.first_recomputed
              << std::endl
              << "  max abs diff          : " << mismatch.max_abs_diff
              << std::endl;
  }
  std::cout << "  A timing checksum : 0x" << std::hex
            << combined_timing_checksum << std::endl
            << "  B timing checksum : 0x" << reconstructed_timing_checksum
            << std::dec << std::endl
            << "  timing checksum match: "
            << (combined_timing_checksum == reconstructed_timing_checksum
                    ? "yes"
                    : "NO")
            << std::endl;
}

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

using SigmoidBenchImplementation = Network::BenchmarkSigmoidImplementation;

enum class SigmoidBenchOperation {
  MainSigmoid,
  AbsSigmoidAndValueGate,
  MainAndAbs,
  FullNetwork,
};

template<SigmoidBenchImplementation MainImplementation,
         SigmoidBenchImplementation AbsImplementation>
std::int32_t ComputeNnueNetworkSigmoidBenchOutput(
    const NetworkBenchCase& input, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(input.selected_bucket);
  const auto scales = selected_network.BenchmarkPhase(
      input.transformed.data(), input.diff_transformed.data(),
      input.abs_transformed.data(), input.material_bucket,
      work.phase_input, work.phase_out);

  selected_network.BenchmarkFmAffine(
      input.diff_transformed.data(), input.abs_transformed.data(),
      work.diff_fc_out, work.abs_fc_out);
  float diff_sum_sq = 0.0f;
  float diff_inv_rms = 0.0f;
  selected_network.BenchmarkDiffRmsNorm(
      work.diff_fc_out, &diff_sum_sq, &diff_inv_rms);
  selected_network.BenchmarkDiffQuantize(
      work.diff_fc_out, diff_inv_rms, work.diff_ac_out);
  alignas(kCacheLineSize) std::int32_t abs_gated[32];
  selected_network
      .template BenchmarkAbsSigmoidGateCandidate<AbsImplementation>(
          work.abs_fc_out, abs_gated);
  selected_network.BenchmarkAbsGateQuantize(abs_gated, work.abs_ac_out);
  selected_network.BenchmarkAbsSquared(work.abs_ac_out, work.abs_sqr_out);

  selected_network.BenchmarkMainFc0(
      input.transformed.data(), work.fc_0_out);
  selected_network.template BenchmarkMainGateCandidate<MainImplementation>(
      work.fc_0_out, work.diff_fc_out, work.fc_0_out);
  selected_network.BenchmarkMainSqrClippedRelu(
      work.fc_0_out, work.ac_sqr_0_out_temp);
  selected_network.BenchmarkMainClippedRelu(work.fc_0_out, work.ac_0_out);

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
  selected_network.BenchmarkFc1(work.l2_input, work.fc_1_out);
  selected_network.BenchmarkAc1(work.fc_1_out, work.ac_1_out);
  selected_network.BenchmarkFc2(work.ac_1_out, work.fc_2_out);
  return selected_network.BenchmarkBlend(work.fc_0_out[31],
                                         work.fc_2_out[0]);
}

#if defined(USE_NNUE_APPROX_SIGMOID_LUT)

enum class MainGateIntegerBenchImplementation {
  CurrentFloatLut,
  ProductionCompact64,
};

enum class MainGateIntegerBenchOperation {
  SigmoidOnly,
  CompleteGate,
  FullNetwork,
};

template<MainGateIntegerBenchImplementation Implementation>
void ComputeMainGateIntegerBench(
    const Network& selected_network, const std::int32_t* main_input,
    const std::int32_t* diff_fc_output, std::int32_t* output) {
  if constexpr (Implementation ==
                MainGateIntegerBenchImplementation::CurrentFloatLut)
    selected_network.BenchmarkMainGateFloatLut(
        main_input, diff_fc_output, output);
  else
    selected_network.BenchmarkMainGateReconstructed(
        main_input, diff_fc_output, output);
}

template<MainGateIntegerBenchImplementation Implementation>
void ComputeMainGateIntegerBenchQ64(
    const Network& selected_network, const std::int32_t* diff_fc_output,
    std::int32_t* gate_q64) {
  if constexpr (Implementation ==
                MainGateIntegerBenchImplementation::CurrentFloatLut)
    selected_network.BenchmarkMainGateFloatLutSigmoid(
        diff_fc_output, gate_q64);
  else
    selected_network.BenchmarkMainGateSigmoid(diff_fc_output, gate_q64);
}

template<MainGateIntegerBenchImplementation Implementation>
std::int32_t ComputeNnueNetworkMainGateIntegerBenchOutput(
    const NetworkBenchCase& input, Network::Buffer& work) {
  const Network& selected_network =
      NnueBenchSelectedNetwork(input.selected_bucket);
  const auto scales = selected_network.BenchmarkPhase(
      input.transformed.data(), input.diff_transformed.data(),
      input.abs_transformed.data(), input.material_bucket,
      work.phase_input, work.phase_out);

  selected_network.BenchmarkFmAffine(
      input.diff_transformed.data(), input.abs_transformed.data(),
      work.diff_fc_out, work.abs_fc_out);
  selected_network.BenchmarkFmActivation(
      work.diff_fc_out, work.abs_fc_out, work.diff_ac_out,
      work.abs_ac_out, work.abs_sqr_out);

  selected_network.BenchmarkMainFc0(
      input.transformed.data(), work.fc_0_out);
  ComputeMainGateIntegerBench<Implementation>(
      selected_network, work.fc_0_out, work.diff_fc_out, work.fc_0_out);
  selected_network.BenchmarkMainSqrClippedRelu(
      work.fc_0_out, work.ac_sqr_0_out_temp);
  selected_network.BenchmarkMainClippedRelu(work.fc_0_out, work.ac_0_out);

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
  selected_network.BenchmarkFc1(work.l2_input, work.fc_1_out);
  selected_network.BenchmarkAc1(work.fc_1_out, work.ac_1_out);
  selected_network.BenchmarkFc2(work.ac_1_out, work.fc_2_out);
  return selected_network.BenchmarkBlend(work.fc_0_out[31],
                                         work.fc_2_out[0]);
}

template<MainGateIntegerBenchImplementation Implementation,
         MainGateIntegerBenchOperation Operation>
NnueBenchTiming MeasureMainGateIntegerCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t output[32];
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    if constexpr (Operation == MainGateIntegerBenchOperation::SigmoidOnly) {
      ComputeMainGateIntegerBenchQ64<Implementation>(
          selected_network, sample.intermediate.diff_fc_out, output);
      const auto index = static_cast<std::size_t>(timing.calls) & 31;
      MixNnueBenchChecksum(checksum, output[index]);
    } else if constexpr (
        Operation == MainGateIntegerBenchOperation::CompleteGate) {
      ComputeMainGateIntegerBench<Implementation>(
          selected_network, sample.main_before_gate.data(),
          sample.intermediate.diff_fc_out, output);
      const auto index = static_cast<std::size_t>(timing.calls) & 31;
      MixNnueBenchChecksum(checksum, output[index]);
      MixNnueBenchChecksum(checksum, output[(index + 17) & 31]);
    } else {
      const auto value =
          ComputeNnueNetworkMainGateIntegerBenchOutput<Implementation>(
              sample.input, work);
      MixNnueBenchChecksum(checksum, value);
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<MainGateIntegerBenchImplementation Implementation,
         MainGateIntegerBenchOperation Operation>
NnueBenchTiming MeasureMainGateIntegerCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureMainGateIntegerCorpus<Implementation, Operation>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureMainGateIntegerCorpus<Implementation, Operation>(
      corpus, checksum);
}

struct MainGateIntegerValidation {
  NnueBenchIntegerDiagnostic gate_q64;
  NnueBenchIntegerDiagnostic gate_applied;
  NnueBenchIntegerDiagnostic gate_output;
  NnueBenchIntegerDiagnostic final_output;
  std::uint64_t reference_gate_checksum =
      UINT64_C(14695981039346656037);
  std::uint64_t candidate_gate_checksum =
      UINT64_C(14695981039346656037);
  std::uint64_t reference_final_checksum =
      UINT64_C(14695981039346656037);
  std::uint64_t candidate_final_checksum =
      UINT64_C(14695981039346656037);
};

template<MainGateIntegerBenchImplementation Candidate>
MainGateIntegerValidation ValidateMainGateIntegerCandidate(
    const std::vector<NetworkStageBenchCase>& corpus) {
  MainGateIntegerValidation validation;
  alignas(kCacheLineSize) std::int32_t reference_q64[32];
  alignas(kCacheLineSize) std::int32_t candidate_q64[32];
  alignas(kCacheLineSize) std::int32_t reference_applied[32];
  alignas(kCacheLineSize) std::int32_t candidate_applied[32];
  alignas(kCacheLineSize) std::int32_t reference_output[32];
  alignas(kCacheLineSize) std::int32_t candidate_output[32];
  alignas(kCacheLineSize) Network::Buffer reference_work{};
  alignas(kCacheLineSize) Network::Buffer candidate_work{};

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    ComputeMainGateIntegerBenchQ64<
        MainGateIntegerBenchImplementation::CurrentFloatLut>(
            selected_network, sample.intermediate.diff_fc_out,
            reference_q64);
    ComputeMainGateIntegerBenchQ64<
        Candidate>(
            selected_network, sample.intermediate.diff_fc_out,
            candidate_q64);
    selected_network.BenchmarkMainGateApply(
        sample.main_before_gate.data(), reference_q64, reference_applied);
    selected_network.BenchmarkMainGateApply(
        sample.main_before_gate.data(), candidate_q64, candidate_applied);
    selected_network.BenchmarkMainGateClamp(
        reference_applied, reference_output);
    selected_network.BenchmarkMainGateClamp(
        candidate_applied, candidate_output);

    MixNnueBenchRange(validation.reference_gate_checksum, reference_q64, 32);
    MixNnueBenchRange(validation.candidate_gate_checksum, candidate_q64, 32);
    AddNnueBenchIntegerDiagnostic(
        validation.gate_q64, reference_q64, candidate_q64, 32,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        validation.gate_applied, reference_applied, candidate_applied, 32,
        sample_index);
    AddNnueBenchIntegerDiagnostic(
        validation.gate_output, reference_output, candidate_output, 32,
        sample_index);

    const auto reference_final =
        ComputeNnueNetworkMainGateIntegerBenchOutput<
            MainGateIntegerBenchImplementation::CurrentFloatLut>(
                sample.input, reference_work);
    const auto candidate_final =
        ComputeNnueNetworkMainGateIntegerBenchOutput<
            Candidate>(
                sample.input, candidate_work);
    MixNnueBenchChecksum(validation.reference_final_checksum,
                         reference_final);
    MixNnueBenchChecksum(validation.candidate_final_checksum,
                         candidate_final);
    AddNnueBenchIntegerDiagnostic(
        validation.final_output, &reference_final, &candidate_final, 1,
        sample_index);
  }
  return validation;
}

template<MainGateIntegerBenchOperation Operation>
NnueBenchTiming DispatchMainGateIntegerMeasurement(
    const MainGateIntegerBenchImplementation implementation,
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  switch (implementation) {
    case MainGateIntegerBenchImplementation::CurrentFloatLut:
      return MeasureMainGateIntegerCorpusAfterWarmup<
          MainGateIntegerBenchImplementation::CurrentFloatLut, Operation>(
              corpus, checksum);
    case MainGateIntegerBenchImplementation::ProductionCompact64:
      return MeasureMainGateIntegerCorpusAfterWarmup<
          MainGateIntegerBenchImplementation::ProductionCompact64, Operation>(
              corpus, checksum);
  }
  return {};
}

NnueBenchTiming DispatchMainGateIntegerOperation(
    const MainGateIntegerBenchImplementation implementation,
    const MainGateIntegerBenchOperation operation,
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  switch (operation) {
    case MainGateIntegerBenchOperation::SigmoidOnly:
      return DispatchMainGateIntegerMeasurement<
          MainGateIntegerBenchOperation::SigmoidOnly>(
              implementation, corpus, checksum);
    case MainGateIntegerBenchOperation::CompleteGate:
      return DispatchMainGateIntegerMeasurement<
          MainGateIntegerBenchOperation::CompleteGate>(
              implementation, corpus, checksum);
    case MainGateIntegerBenchOperation::FullNetwork:
      return DispatchMainGateIntegerMeasurement<
          MainGateIntegerBenchOperation::FullNetwork>(
              implementation, corpus, checksum);
  }
  return {};
}

template<MainGateIntegerBenchImplementation Implementation>
std::int32_t MainGateIntegerValue(const std::int32_t raw) {
  if constexpr (Implementation ==
                MainGateIntegerBenchImplementation::CurrentFloatLut)
    return Network::sigmoid_gate_slow(raw, 64);
  else
    return Network::MainGateCompactQ64Value(raw);
}

template<MainGateIntegerBenchImplementation Candidate>
NnueBenchIntegerDiagnostic ValidateMainGateIntegerAllRawInputs() {
  NnueBenchIntegerDiagnostic result;
  std::size_t sample = 0;
  for (std::int32_t raw = Network::kMainGateCompactRawMinimum;
       raw <= Network::kMainGateCompactRawMaximum; ++raw, ++sample) {
    const auto reference = MainGateIntegerValue<
        MainGateIntegerBenchImplementation::CurrentFloatLut>(raw);
    const auto candidate = MainGateIntegerValue<Candidate>(raw);
    AddNnueBenchIntegerDiagnostic(
        result, &reference, &candidate, 1, sample);
  }
  const std::array<std::int32_t, 4> saturation_inputs = {
      Network::kMainGateCompactRawMinimum - 1,
      std::numeric_limits<std::int32_t>::min(),
      Network::kMainGateCompactRawMaximum + 1,
      std::numeric_limits<std::int32_t>::max()};
  for (const auto raw : saturation_inputs) {
    const auto reference = MainGateIntegerValue<
        MainGateIntegerBenchImplementation::CurrentFloatLut>(raw);
    const auto candidate = MainGateIntegerValue<Candidate>(raw);
    AddNnueBenchIntegerDiagnostic(
        result, &reference, &candidate, 1, sample++);
  }
  return result;
}

void PrintMainGateCompactTableInfo() {
  const auto& table = Network::MainGateCompactQ64Lut();
  std::uint64_t transition_bins = 0;
  for (const auto& entry : table)
    transition_bins +=
        entry.threshold <= Network::kMainGateCompactCoarseWidth;
  std::cout << "B. production C64-v2 branchless compact LUT" << std::endl
            << "  coarse width       : "
            << Network::kMainGateCompactCoarseWidth << " raw values"
            << std::endl
            << "  coarse bins        : " << table.size() << std::endl
            << "  entry layout       : uint8 threshold + uint8 base"
            << std::endl
            << "  table bytes        : " << sizeof(table) << std::endl
            << "  lookup             : clamp -> one 2-byte entry -> base + compare"
            << std::endl
            << "  threshold compares : exactly 1 (branchless)" << std::endl
            << "  transition/no-transition bins: " << transition_bins << " / "
            << table.size() - transition_bins << std::endl
            << "  no-transition sentinel: "
            << Network::kMainGateCompactCoarseWidth + 1 << std::endl;
}

void PrintMainGateIntegerCandidateValidation(
    const char* const name, const MainGateIntegerValidation& validation) {
  std::cout << name << std::endl;
  PrintNnueBenchIntegerDiagnostic("  gate_q64[32]", validation.gate_q64);
  PrintNnueBenchIntegerDiagnostic(
      "  Main gate applied before clamp[32]", validation.gate_applied);
  PrintNnueBenchIntegerDiagnostic(
      "  fc_0_out[32]", validation.gate_output);
  PrintNnueBenchIntegerDiagnostic(
      "  final Network output", validation.final_output);
}

void TestMainGateIntegerBenchmarkCompare(const std::uint64_t repeat_count) {
  // Build the production table before corpus generation and timed regions.
  Network::MainGateCompactQ64Lut();

  std::cout << "[NNUE benchmark: Main gate exact compact Q64 LUTs]"
            << std::endl
            << "  baseline     : production float32 LUT [-10,8], step 1/32"
            << std::endl
            << "  candidate    : production C64-v2 branchless compact LUT"
            << std::endl
            << "  raw range    : ["
            << Network::kMainGateCompactRawMinimum << ", "
            << Network::kMainGateCompactRawMaximum << "]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  order        : implementation and stage order rotated per repeat"
            << std::endl
            << "[table layout]" << std::endl;
  PrintMainGateCompactTableInfo();

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: Main gate integer benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  compared gate values: " << corpus.size() * 32 << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  constexpr std::array<MainGateIntegerBenchImplementation, 2>
      implementations = {
          MainGateIntegerBenchImplementation::CurrentFloatLut,
          MainGateIntegerBenchImplementation::ProductionCompact64};
  constexpr std::array<const char*, 2> implementation_names = {
      "A. previous production float LUT",
      "B. production C64-v2 compact LUT"};
  constexpr std::array<MainGateIntegerBenchOperation, 3> operations = {
      MainGateIntegerBenchOperation::SigmoidOnly,
      MainGateIntegerBenchOperation::CompleteGate,
      MainGateIntegerBenchOperation::FullNetwork};
  constexpr std::array<const char*, 3> operation_names = {
      "Main sigmoid -> gate_q64[32]",
      "Main gate sigmoid + apply + clamp",
      "full staged Network (Main candidate only)"};
  std::array<std::array<NnueBenchSamples, implementations.size()>,
             operations.size()> samples;
  std::array<std::array<std::uint64_t, implementations.size()>,
             operations.size()> timing_checksums;
  for (auto& checksums : timing_checksums)
    checksums.fill(UINT64_C(14695981039346656037));

  for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat)
    for (std::size_t operation_offset = 0;
         operation_offset < operations.size(); ++operation_offset) {
      const auto operation_index =
          (operation_offset + static_cast<std::size_t>(repeat))
          % operations.size();
      for (std::size_t implementation_offset = 0;
           implementation_offset < implementations.size();
           ++implementation_offset) {
        const auto implementation_index =
            (implementation_offset + static_cast<std::size_t>(repeat))
            % implementations.size();
        samples[operation_index][implementation_index].Add(
            DispatchMainGateIntegerOperation(
                implementations[implementation_index],
                operations[operation_index], corpus,
                timing_checksums[operation_index][implementation_index]));
      }
    }

  for (std::size_t operation = 0; operation < operations.size();
       ++operation) {
    std::cout << operation_names[operation] << std::endl;
    const auto reference =
        SummarizeNnueBenchSamples(samples[operation][0]);
    for (std::size_t implementation = 0;
         implementation < implementations.size(); ++implementation) {
      PrintNnueBenchSamples(implementation_names[implementation],
                            samples[operation][implementation]);
      const auto summary = SummarizeNnueBenchSamples(
          samples[operation][implementation]);
      const double improvement = reference.median == 0.0
          ? 0.0
          : (reference.median - summary.median) * 100.0
                / reference.median;
      std::cout << "  relative improvement vs A: " << std::fixed
                << std::setprecision(2) << improvement << "%" << std::endl
                << "  timing checksum: 0x" << std::hex
                << timing_checksums[operation][implementation] << std::dec
                << std::endl
                << "  checksum match vs A: "
                << (timing_checksums[operation][implementation]
                            == timing_checksums[operation][0]
                        ? "yes" : "NO")
                << std::endl;
    }
  }

  std::cout << "[exhaustive raw-input validation]" << std::endl
            << "  in-range values : "
            << Network::kMainGateCompactRawCount << std::endl
            << "  saturation probes: 4" << std::endl;
  const auto raw_production = ValidateMainGateIntegerAllRawInputs<
      MainGateIntegerBenchImplementation::ProductionCompact64>();
  PrintNnueBenchIntegerDiagnostic("B. production vs A", raw_production);

  std::cout << "[corpus correctness]" << std::endl;
  PrintMainGateIntegerCandidateValidation(
      "B. production vs A",
      ValidateMainGateIntegerCandidate<
          MainGateIntegerBenchImplementation::ProductionCompact64>(corpus));
}

#endif  // defined(USE_NNUE_APPROX_SIGMOID_LUT)

template<SigmoidBenchImplementation Implementation,
         SigmoidBenchOperation Operation>
NnueBenchTiming MeasureSigmoidApproximationCorpus(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  alignas(kCacheLineSize) std::int32_t main_gate_q64[32];
  alignas(kCacheLineSize) std::int32_t abs_gated[32];
  alignas(kCacheLineSize) Network::Buffer work{};
  NnueBenchTiming timing;

  const auto begin = NnueBenchClock::now();
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    if constexpr (Operation == SigmoidBenchOperation::MainSigmoid
                  || Operation == SigmoidBenchOperation::MainAndAbs) {
      selected_network
          .template BenchmarkMainGateSigmoidCandidate<Implementation>(
              sample.intermediate.diff_fc_out, main_gate_q64);
    }
    if constexpr (Operation == SigmoidBenchOperation::AbsSigmoidAndValueGate
                  || Operation == SigmoidBenchOperation::MainAndAbs) {
      selected_network
          .template BenchmarkAbsSigmoidGateCandidate<Implementation>(
              sample.intermediate.abs_fc_out, abs_gated);
    }

    if constexpr (Operation == SigmoidBenchOperation::FullNetwork) {
      const std::int32_t output =
          ComputeNnueNetworkSigmoidBenchOutput<Implementation,
                                                Implementation>(sample.input,
                                                                work);
      MixNnueBenchChecksum(checksum, output);
    } else {
      const std::size_t index = static_cast<std::size_t>(timing.calls) & 31;
      if constexpr (Operation == SigmoidBenchOperation::MainSigmoid)
        MixNnueBenchChecksum(checksum, main_gate_q64[index]);
      else if constexpr (
          Operation == SigmoidBenchOperation::AbsSigmoidAndValueGate)
        MixNnueBenchChecksum(checksum, abs_gated[index]);
      else {
        MixNnueBenchChecksum(checksum, main_gate_q64[index]);
        MixNnueBenchChecksum(checksum, abs_gated[(index + 17) & 31]);
      }
    }
    ++timing.calls;
  }
  const auto end = NnueBenchClock::now();
  timing.nanoseconds =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return timing;
}

template<SigmoidBenchImplementation Implementation,
         SigmoidBenchOperation Operation>
NnueBenchTiming MeasureSigmoidApproximationCorpusAfterWarmup(
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  std::uint64_t warmup_checksum = UINT64_C(14695981039346656037);
  MeasureSigmoidApproximationCorpus<Implementation, Operation>(
      corpus, warmup_checksum);
  MixNnueBenchChecksum(checksum, warmup_checksum);
  return MeasureSigmoidApproximationCorpus<Implementation, Operation>(
      corpus, checksum);
}

template<SigmoidBenchOperation Operation>
NnueBenchTiming DispatchSigmoidApproximationMeasurement(
    const SigmoidBenchImplementation implementation,
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  switch (implementation) {
    case SigmoidBenchImplementation::StdExp:
      return MeasureSigmoidApproximationCorpusAfterWarmup<
          SigmoidBenchImplementation::StdExp, Operation>(corpus, checksum);
    case SigmoidBenchImplementation::FloatLutMinus8To8Step16:
      return MeasureSigmoidApproximationCorpusAfterWarmup<
          SigmoidBenchImplementation::FloatLutMinus8To8Step16, Operation>(
              corpus, checksum);
    case SigmoidBenchImplementation::FloatLutMinus10To8Step16:
      return MeasureSigmoidApproximationCorpusAfterWarmup<
          SigmoidBenchImplementation::FloatLutMinus10To8Step16, Operation>(
              corpus, checksum);
    case SigmoidBenchImplementation::FloatLutMinus8To8Step32:
      return MeasureSigmoidApproximationCorpusAfterWarmup<
          SigmoidBenchImplementation::FloatLutMinus8To8Step32, Operation>(
              corpus, checksum);
    case SigmoidBenchImplementation::FloatLutMinus10To8Step32:
      return MeasureSigmoidApproximationCorpusAfterWarmup<
          SigmoidBenchImplementation::FloatLutMinus10To8Step32, Operation>(
              corpus, checksum);
  }
  return {};
}

NnueBenchTiming DispatchSigmoidApproximationOperation(
    const SigmoidBenchImplementation implementation,
    const SigmoidBenchOperation operation,
    const std::vector<NetworkStageBenchCase>& corpus,
    std::uint64_t& checksum) {
  switch (operation) {
    case SigmoidBenchOperation::MainSigmoid:
      return DispatchSigmoidApproximationMeasurement<
          SigmoidBenchOperation::MainSigmoid>(
              implementation, corpus, checksum);
    case SigmoidBenchOperation::AbsSigmoidAndValueGate:
      return DispatchSigmoidApproximationMeasurement<
          SigmoidBenchOperation::AbsSigmoidAndValueGate>(
              implementation, corpus, checksum);
    case SigmoidBenchOperation::MainAndAbs:
      return DispatchSigmoidApproximationMeasurement<
          SigmoidBenchOperation::MainAndAbs>(
              implementation, corpus, checksum);
    case SigmoidBenchOperation::FullNetwork:
      return DispatchSigmoidApproximationMeasurement<
          SigmoidBenchOperation::FullNetwork>(
              implementation, corpus, checksum);
  }
  return {};
}

struct SigmoidApproximationError {
  double maximum = 0.0;
  double sum = 0.0;
  std::uint64_t count = 0;

  void Add(const float reference, const float candidate) {
    const double error = std::abs(
        static_cast<double>(reference) - static_cast<double>(candidate));
    maximum = std::max(maximum, error);
    sum += error;
    ++count;
  }

  double Mean() const {
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
  }
};

struct SigmoidIntegerDifference {
  std::uint64_t count = 0;
  std::int64_t maximum = 0;

  void Add(const std::int32_t reference, const std::int32_t candidate) {
    const std::int64_t difference = std::abs(
        static_cast<std::int64_t>(reference) - candidate);
    if (difference != 0)
      ++count;
    maximum = std::max(maximum, difference);
  }
};

double NnueBenchPercentile(std::vector<double> values,
                           const double percentile) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::floor((values.size() - 1) * percentile));
  return values[index];
}

void PrintSigmoidDistribution(const char* const name,
                              const std::vector<double>& values) {
  if (values.empty())
    return;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  double mean = 0.0;
  for (const double value : sorted)
    mean += value;
  mean /= static_cast<double>(sorted.size());
  const auto percentile = [&](const double p) {
    return sorted[static_cast<std::size_t>(
        std::floor((sorted.size() - 1) * p))];
  };
  std::cout << name << std::endl
            << "  min    : " << sorted.front() << std::endl
            << "  p1     : " << percentile(0.01) << std::endl
            << "  p10    : " << percentile(0.10) << std::endl
            << "  median : " << percentile(0.50) << std::endl
            << "  mean   : " << mean << std::endl
            << "  p90    : " << percentile(0.90) << std::endl
            << "  p99    : " << percentile(0.99) << std::endl
            << "  max    : " << sorted.back() << std::endl;
}

void PrintSigmoidTimingComparison(
    const char* const name, const std::array<NnueBenchSamples, 5>& samples) {
  constexpr std::array<const char*, 5> names = {
      "Reference. std::exp",
      "A. float32 LUT [-8, 8], step 1/16",
      "B. float32 LUT [-10, 8], step 1/16",
      "C. float32 LUT [-8, 8], step 1/32",
      "D. float32 LUT [-10, 8], step 1/32"};
  const auto baseline = SummarizeNnueBenchSamples(samples[0]);
  std::cout << name << std::endl;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto summary = SummarizeNnueBenchSamples(samples[i]);
    const double improvement = baseline.median == 0.0
        ? 0.0
        : (baseline.median - summary.median) * 100.0 / baseline.median;
    std::cout << "  " << names[i] << std::endl
              << "    median ns/call : " << std::fixed << std::setprecision(1)
              << summary.median << std::endl
              << "    mean ns/call   : " << summary.mean << std::endl
              << "    min ns/call    : " << summary.minimum << std::endl
              << "    max ns/call    : " << summary.maximum << std::endl
              << "    improvement    : " << std::setprecision(2)
              << improvement << "%" << std::endl;
  }
}

std::int32_t NnueBenchNetworkOutputToEval(const std::int32_t output) {
  return std::clamp(output / FV_SCALE, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);
}

void PrintEvalDifference(const char* const name,
                         const std::vector<double>& absolute_differences) {
  double mean = 0.0;
  double maximum = 0.0;
  for (const double difference : absolute_differences) {
    mean += difference;
    maximum = std::max(maximum, difference);
  }
  if (!absolute_differences.empty())
    mean /= static_cast<double>(absolute_differences.size());
  std::cout << name << std::endl
            << "  max abs eval diff  : " << maximum << std::endl
            << "  mean abs eval diff : " << mean << std::endl
            << "  p50 / p90 / p95 / p99 : "
            << NnueBenchPercentile(absolute_differences, 0.50) << " / "
            << NnueBenchPercentile(absolute_differences, 0.90) << " / "
            << NnueBenchPercentile(absolute_differences, 0.95) << " / "
            << NnueBenchPercentile(absolute_differences, 0.99) << std::endl;
}

template<SigmoidBenchImplementation Candidate>
void ValidateSigmoidApproximationCandidate(
    const char* const name, const std::vector<NetworkStageBenchCase>& corpus,
    const std::vector<std::int32_t>& baseline_outputs) {
  SigmoidApproximationError main_sigmoid_error;
  SigmoidIntegerDifference main_q64_difference;
  SigmoidIntegerDifference main_gate_output_difference;
  NnueNetworkStageMismatch main_final_output_difference{};
  std::vector<double> main_eval_differences;
  main_eval_differences.reserve(corpus.size());

  SigmoidApproximationError abs_sigmoid_error;
  SigmoidIntegerDifference abs_raw_difference;
  SigmoidIntegerDifference abs_byte_difference;
  NnueNetworkStageMismatch abs_final_output_difference{};
  NnueNetworkStageMismatch combined_final_output_difference{};
  std::vector<double> combined_eval_differences;
  combined_eval_differences.reserve(corpus.size());

  alignas(kCacheLineSize) std::int32_t main_q64_a[32];
  alignas(kCacheLineSize) std::int32_t main_q64_b[32];
  alignas(kCacheLineSize) std::int32_t main_gate_a[32];
  alignas(kCacheLineSize) std::int32_t main_gate_b[32];
  alignas(kCacheLineSize) std::int32_t abs_raw_a[32];
  alignas(kCacheLineSize) std::int32_t abs_raw_b[32];
  alignas(kCacheLineSize) std::uint8_t abs_byte_a[32];
  alignas(kCacheLineSize) std::uint8_t abs_byte_b[32];
  alignas(kCacheLineSize) Network::Buffer work{};

  for (std::size_t sample_index = 0; sample_index < corpus.size();
       ++sample_index) {
    const auto& sample = corpus[sample_index];
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    selected_network.template BenchmarkMainGateSigmoidCandidate<
        SigmoidBenchImplementation::StdExp>(
            sample.intermediate.diff_fc_out, main_q64_a);
    selected_network.template BenchmarkMainGateSigmoidCandidate<Candidate>(
        sample.intermediate.diff_fc_out, main_q64_b);
    selected_network.template BenchmarkMainGateCandidate<
        SigmoidBenchImplementation::StdExp>(
            sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
            main_gate_a);
    selected_network.template BenchmarkMainGateCandidate<Candidate>(
        sample.main_before_gate.data(), sample.intermediate.diff_fc_out,
        main_gate_b);

    selected_network.template BenchmarkAbsSigmoidGateCandidate<
        SigmoidBenchImplementation::StdExp>(
            sample.intermediate.abs_fc_out, abs_raw_a);
    selected_network.template BenchmarkAbsSigmoidGateCandidate<Candidate>(
        sample.intermediate.abs_fc_out, abs_raw_b);
    selected_network.BenchmarkAbsGateQuantize(abs_raw_a, abs_byte_a);
    selected_network.BenchmarkAbsGateQuantize(abs_raw_b, abs_byte_b);

    for (int j = 0; j < 32; ++j) {
      const std::int32_t main_raw_x =
          sample.intermediate.diff_fc_out[j] - 2438;
      main_sigmoid_error.Add(
          Network::BenchmarkSigmoidValue<
              SigmoidBenchImplementation::StdExp>(main_raw_x),
          Network::BenchmarkSigmoidValue<Candidate>(main_raw_x));
      main_q64_difference.Add(main_q64_a[j], main_q64_b[j]);
      main_gate_output_difference.Add(main_gate_a[j], main_gate_b[j]);

      const std::int32_t abs_raw_x = sample.intermediate.abs_fc_out[j];
      abs_sigmoid_error.Add(
          Network::BenchmarkSigmoidValue<
              SigmoidBenchImplementation::StdExp>(abs_raw_x),
          Network::BenchmarkSigmoidValue<Candidate>(abs_raw_x));
      abs_raw_difference.Add(abs_raw_a[j], abs_raw_b[j]);
      abs_byte_difference.Add(abs_byte_a[j], abs_byte_b[j]);
    }

    const std::int32_t main_only_output =
        ComputeNnueNetworkSigmoidBenchOutput<Candidate,
            SigmoidBenchImplementation::StdExp>(sample.input, work);
    CompareNnueNetworkStageRange(
        &baseline_outputs[sample_index], &main_only_output, 1, sample_index,
        main_final_output_difference);
    main_eval_differences.push_back(std::abs(
        NnueBenchNetworkOutputToEval(main_only_output)
        - NnueBenchNetworkOutputToEval(baseline_outputs[sample_index])));

    const std::int32_t abs_only_output =
        ComputeNnueNetworkSigmoidBenchOutput<
            SigmoidBenchImplementation::StdExp, Candidate>(sample.input,
                                                            work);
    CompareNnueNetworkStageRange(
        &baseline_outputs[sample_index], &abs_only_output, 1, sample_index,
        abs_final_output_difference);

    const std::int32_t combined_output =
        ComputeNnueNetworkSigmoidBenchOutput<Candidate, Candidate>(
            sample.input, work);
    CompareNnueNetworkStageRange(
        &baseline_outputs[sample_index], &combined_output, 1, sample_index,
        combined_final_output_difference);
    combined_eval_differences.push_back(std::abs(
        NnueBenchNetworkOutputToEval(combined_output)
        - NnueBenchNetworkOutputToEval(baseline_outputs[sample_index])));
  }

  const double main_q64_rate = main_q64_difference.count * 100.0 /
      static_cast<double>(corpus.size() * 32);
  const double abs_byte_rate = abs_byte_difference.count * 100.0 /
      static_cast<double>(corpus.size() * 32);
  std::cout << "[" << name << " correctness / error]" << std::endl
            << "Main sigmoid" << std::endl
            << "  max / mean abs error : " << std::scientific
            << main_sigmoid_error.maximum << " / "
            << main_sigmoid_error.Mean() << std::fixed << std::endl
            << "  q64 mismatch count/rate : "
            << main_q64_difference.count << " / "
            << std::setprecision(6) << main_q64_rate << "%" << std::endl
            << "  q64 max integer diff    : "
            << main_q64_difference.maximum << std::endl
            << "  gate-applied int32 mismatch : "
            << main_gate_output_difference.count << std::endl
            << "  gate-applied max diff       : "
            << main_gate_output_difference.maximum << std::endl
            << "  final Network output mismatch: "
            << main_final_output_difference.count << std::endl
            << "FM Abs sigmoid + value gate" << std::endl
            << "  max / mean abs error : " << std::scientific
            << abs_sigmoid_error.maximum << " / "
            << abs_sigmoid_error.Mean() << std::fixed << std::endl
            << "  a_gated raw mismatch : " << abs_raw_difference.count
            << std::endl
            << "  a_gated max diff     : " << abs_raw_difference.maximum
            << std::endl
            << "  abs_ac_out mismatch/rate : " << abs_byte_difference.count
            << " / " << std::setprecision(6) << abs_byte_rate << "%"
            << std::endl
            << "  abs_ac_out max diff       : "
            << abs_byte_difference.maximum << std::endl
            << "  final Network output mismatch: "
            << abs_final_output_difference.count << std::endl
            << "Main + FM combined final Network output mismatch: "
            << combined_final_output_difference.count << std::endl;
  PrintEvalDifference("Main-only eval difference", main_eval_differences);
  PrintEvalDifference("Main+FM combined eval difference",
                      combined_eval_differences);
}

constexpr std::uint64_t kSigmoidCandidateValidationGames = 1000;
constexpr int kSigmoidCandidateValidationMaxPly = 256;

std::size_t SigmoidCandidateEvalHistogramBin(const std::int32_t difference) {
  if (difference == 0) return 0;
  if (difference == 1) return 1;
  if (difference == 2) return 2;
  if (difference <= 4) return 3;
  if (difference <= 8) return 4;
  if (difference <= 16) return 5;
  if (difference <= 32) return 6;
  if (difference <= 64) return 7;
  return 8;
}

void PrintSigmoidCandidateEvalHistogram(
    const std::array<std::uint64_t, 9>& histogram,
    const std::uint64_t total_positions) {
  constexpr std::array<const char*, 9> labels = {
      "0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"};
  std::cout << "[absolute eval/cp difference histogram]" << std::endl;
  for (std::size_t index = 0; index < histogram.size(); ++index) {
    const double rate = total_positions == 0
        ? 0.0
        : 100.0 * static_cast<double>(histogram[index])
            / static_cast<double>(total_positions);
    std::cout << "  " << std::setw(5) << labels[index] << " : "
              << std::setw(8) << histogram[index] << " ("
              << std::fixed << std::setprecision(6) << rate << "%)"
              << std::endl;
  }
}

void ValidateProductionSigmoidCandidate() {
  constexpr auto kReference = SigmoidBenchImplementation::StdExp;
  constexpr auto kCandidate =
      SigmoidBenchImplementation::FloatLutMinus10To8Step32;

  std::cout << "[NNUE validation: production sigmoid candidate]" << std::endl
            << "  reference    : current std::exp" << std::endl
            << "  candidate    : float32 LUT [-10, 8], step 1/32"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kSigmoidCandidateValidationGames
            << std::endl
            << "  max ply/game : " << kSigmoidCandidateValidationMaxPly
            << std::endl
            << "  generation   : fixed-seed random legal games, streamed"
            << std::endl;

  Position pos;
  StateInfo root_state;
  std::vector<StateInfo> states(kSigmoidCandidateValidationMaxPly);
  PRNG prng(kNnueBenchSeed);
  alignas(kCacheLineSize) std::int32_t router_output[32];
  alignas(kCacheLineSize) std::int32_t main_q64_reference[32];
  alignas(kCacheLineSize) std::int32_t main_q64_candidate[32];
  alignas(kCacheLineSize) std::int32_t abs_gated_reference[32];
  alignas(kCacheLineSize) std::int32_t abs_gated_candidate[32];
  alignas(kCacheLineSize) std::uint8_t abs_output_reference[32];
  alignas(kCacheLineSize) std::uint8_t abs_output_candidate[32];
  alignas(kCacheLineSize) std::int32_t diff_fc_output[64];
  alignas(kCacheLineSize) std::int32_t abs_fc_output[64];
  alignas(kCacheLineSize) char normal_network_buffer[Network::kBufferSize];
  alignas(kCacheLineSize) Network::Buffer reference_work{};
  alignas(kCacheLineSize) Network::Buffer candidate_work{};

  SigmoidIntegerDifference main_q64_difference;
  SigmoidIntegerDifference abs_output_difference;
  NnueNetworkStageMismatch reference_vs_normal{};
  NnueNetworkStageMismatch candidate_vs_normal{};
  NnueNetworkStageMismatch candidate_final_difference{};
  std::vector<double> eval_differences;
  std::array<std::uint64_t, 9> eval_histogram{};
  std::uint64_t total_positions = 0;
  std::uint64_t reference_checksum = UINT64_C(14695981039346656037);
  std::uint64_t candidate_checksum = UINT64_C(14695981039346656037);

  std::cout << "  processing   : " << std::flush;
  for (std::uint64_t game = 0; game < kSigmoidCandidateValidationGames;
       ++game) {
    pos.set_hirate(&root_state);
    for (int ply = 0; ply < kSigmoidCandidateValidationMaxPly; ++ply) {
      MoveList<LEGAL_ALL> moves(pos);
      if (moves.size() == 0)
        break;

      const Move move = moves.begin()[prng.rand(moves.size())];
      pos.do_move(move, states[ply]);

      NetworkBenchCase input{};
      input.material_bucket = NnueBenchMaterialBucket(pos);
      feature_transformer->Transform(
          pos, input.transformed.data(), input.diff_transformed.data(),
          input.abs_transformed.data(), false, input.material_bucket);
      FillNnueBenchRouterInput(input);
      router->PropagatePrefix<12>(input.router_input.data(), router_output);
      input.selected_bucket = SelectNnueBenchBucket(router_output);

      const Network& selected_network =
          NnueBenchSelectedNetwork(input.selected_bucket);
      selected_network.BenchmarkFmAffine(
          input.diff_transformed.data(), input.abs_transformed.data(),
          diff_fc_output, abs_fc_output);

      selected_network.template BenchmarkMainGateSigmoidCandidate<kReference>(
          diff_fc_output, main_q64_reference);
      selected_network.template BenchmarkMainGateSigmoidCandidate<kCandidate>(
          diff_fc_output, main_q64_candidate);
      selected_network.template BenchmarkAbsSigmoidGateCandidate<kReference>(
          abs_fc_output, abs_gated_reference);
      selected_network.template BenchmarkAbsSigmoidGateCandidate<kCandidate>(
          abs_fc_output, abs_gated_candidate);
      selected_network.BenchmarkAbsGateQuantize(
          abs_gated_reference, abs_output_reference);
      selected_network.BenchmarkAbsGateQuantize(
          abs_gated_candidate, abs_output_candidate);

      for (int index = 0; index < 32; ++index) {
        main_q64_difference.Add(main_q64_reference[index],
                                main_q64_candidate[index]);
        abs_output_difference.Add(abs_output_reference[index],
                                  abs_output_candidate[index]);
      }

      const std::int32_t reference_output =
          ComputeNnueNetworkSigmoidBenchOutput<kReference, kReference>(
              input, reference_work);
      const std::int32_t candidate_output =
          ComputeNnueNetworkSigmoidBenchOutput<kCandidate, kCandidate>(
              input, candidate_work);
      const std::int32_t normal_output = selected_network.Propagate(
          input.transformed.data(), input.diff_transformed.data(),
          input.abs_transformed.data(), input.material_bucket,
          normal_network_buffer)[0];
      CompareNnueNetworkStageRange(
          &normal_output, &reference_output, 1, total_positions,
          reference_vs_normal);
      CompareNnueNetworkStageRange(
          &normal_output, &candidate_output, 1, total_positions,
          candidate_vs_normal);
      CompareNnueNetworkStageRange(
          &reference_output, &candidate_output, 1, total_positions,
          candidate_final_difference);

      const std::int32_t reference_eval =
          NnueBenchNetworkOutputToEval(reference_output);
      const std::int32_t candidate_eval =
          NnueBenchNetworkOutputToEval(candidate_output);
      const std::int32_t eval_difference = static_cast<std::int32_t>(
          std::abs(static_cast<std::int64_t>(candidate_eval)
                   - reference_eval));
      eval_differences.push_back(eval_difference);
      ++eval_histogram[SigmoidCandidateEvalHistogramBin(eval_difference)];
      MixNnueBenchChecksum(reference_checksum, reference_output);
      MixNnueBenchChecksum(candidate_checksum, candidate_output);
      ++total_positions;
    }
    if (game < 10 || (game + 1) % 10 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << std::endl;

  if (total_positions == 0) {
    std::cout << "error: sigmoid candidate validation corpus is empty"
              << std::endl;
    return;
  }

  double eval_sum = 0.0;
  double eval_maximum = 0.0;
  for (const double difference : eval_differences) {
    eval_sum += difference;
    eval_maximum = std::max(eval_maximum, difference);
  }
  const double eval_mean = eval_sum / eval_differences.size();
  const std::uint64_t gate_values = total_positions * 32;
  const auto rate = [](const std::uint64_t count,
                       const std::uint64_t total) {
    return total == 0
        ? 0.0
        : 100.0 * static_cast<double>(count) / static_cast<double>(total);
  };

  std::cout << "[validation results]" << std::endl
            << "  positions                  : " << total_positions
            << std::endl
            << "  gate values                : " << gate_values << std::endl
            << "  reference staged vs normal : "
            << reference_vs_normal.count << " mismatches" << std::endl
            << "  candidate staged vs normal : "
            << candidate_vs_normal.count << " mismatches" << std::endl
            << "  final Network mismatch     : "
            << candidate_final_difference.count << " / " << total_positions
            << " (" << std::fixed << std::setprecision(6)
            << rate(candidate_final_difference.count, total_positions)
            << "%)" << std::endl
            << "  final Network max raw diff : "
            << candidate_final_difference.max_abs_diff << std::endl
            << "  Main Q64 mismatch          : "
            << main_q64_difference.count << " / " << gate_values << " ("
            << rate(main_q64_difference.count, gate_values) << "%)"
            << std::endl
            << "  Main Q64 max diff          : "
            << main_q64_difference.maximum << std::endl
            << "  FM abs_ac_out mismatch     : "
            << abs_output_difference.count << " / " << gate_values << " ("
            << rate(abs_output_difference.count, gate_values) << "%)"
            << std::endl
            << "  FM abs_ac_out max diff     : "
            << abs_output_difference.maximum << std::endl
            << "[absolute eval/cp difference]" << std::endl
            << "  max   : " << eval_maximum << std::endl
            << "  mean  : " << eval_mean << std::endl
            << "  p90   : " << NnueBenchPercentile(eval_differences, 0.90)
            << std::endl
            << "  p95   : " << NnueBenchPercentile(eval_differences, 0.95)
            << std::endl
            << "  p99   : " << NnueBenchPercentile(eval_differences, 0.99)
            << std::endl
            << "  p99.9 : " << NnueBenchPercentile(eval_differences, 0.999)
            << std::endl;
  PrintSigmoidCandidateEvalHistogram(eval_histogram, total_positions);
  std::cout << "[checksums]" << std::endl
            << "  reference : 0x" << std::hex << reference_checksum
            << std::endl
            << "  candidate : 0x" << candidate_checksum << std::dec
            << std::endl;
}

void TestSigmoidApproximationBenchmark(const std::uint64_t repeat_count) {
  std::cout << "[NNUE benchmark: Main / FM sigmoid approximations]"
            << std::endl
            << "  seed         : " << kNnueBenchSeed << std::endl
            << "  corpus games : " << kNnueBenchMeasuredGames << std::endl
            << "  max ply/game : " << kNnueBenchMaxPly << std::endl
            << "  repeats      : " << repeat_count << std::endl
            << "  candidates   : A [-8,8] /16 (257 points)" << std::endl
            << "                 B [-10,8] /16 (289 points)" << std::endl
            << "                 C [-8,8] /32 (513 points)" << std::endl
            << "                 D [-10,8] /32 (577 points)" << std::endl
            << "  outside LUT  : saturate at each endpoint" << std::endl
            << "  lookup       : scalar float32 + linear interpolation" << std::endl
            << "  order        : reference/A/B/C/D rotated per repeat" << std::endl;

  const auto corpus = MakeNnueNetworkStageBenchCorpus();
  if (corpus.empty()) {
    std::cout << "error: sigmoid approximation benchmark corpus is empty"
              << std::endl;
    return;
  }
  std::cout << "  corpus calls : " << corpus.size() << std::endl
            << "  compared gate values: " << corpus.size() * 32 << std::endl
            << "  warm-up calls: " << corpus.size()
            << " before every timed sample" << std::endl;

  std::vector<double> main_logits;
  std::vector<double> main_q64;
  main_logits.reserve(corpus.size() * 32);
  main_q64.reserve(corpus.size() * 32);
  std::vector<std::int32_t> baseline_outputs;
  baseline_outputs.reserve(corpus.size());
  NnueNetworkStageMismatch baseline_vs_normal{};
  alignas(kCacheLineSize) std::int32_t q64[32];
  alignas(kCacheLineSize) Network::Buffer work{};
  for (const auto& sample : corpus) {
    const Network& selected_network =
        NnueBenchSelectedNetwork(sample.input.selected_bucket);
    selected_network.template BenchmarkMainGateSigmoidCandidate<
        SigmoidBenchImplementation::StdExp>(
            sample.intermediate.diff_fc_out, q64);
    for (int j = 0; j < 32; ++j) {
      main_logits.push_back(
          static_cast<double>(sample.intermediate.diff_fc_out[j] - 2438)
          / 8128.0);
      main_q64.push_back(q64[j]);
    }
    const std::int32_t baseline_output =
        ComputeNnueNetworkSigmoidBenchOutput<
            SigmoidBenchImplementation::StdExp,
            SigmoidBenchImplementation::StdExp>(sample.input, work);
    CompareNnueNetworkStageRange(
        &sample.final_output, &baseline_output, 1, baseline_outputs.size(),
        baseline_vs_normal);
    baseline_outputs.push_back(baseline_output);
  }
  PrintSigmoidDistribution("[Main normalized logit z distribution]",
                           main_logits);
  PrintSigmoidDistribution("[Main gate Q64 distribution]", main_q64);
  std::cout << "[benchmark staged-path validation]" << std::endl
            << "  Reference std::exp staged vs normal Propagate mismatch: "
            << baseline_vs_normal.count << std::endl
            << "  max integer diff: " << baseline_vs_normal.max_abs_diff
            << std::endl;

  ValidateSigmoidApproximationCandidate<
      SigmoidBenchImplementation::FloatLutMinus8To8Step16>(
          "A. float32 LUT [-8, 8], step 1/16", corpus,
          baseline_outputs);
  ValidateSigmoidApproximationCandidate<
      SigmoidBenchImplementation::FloatLutMinus10To8Step16>(
          "B. float32 LUT [-10, 8], step 1/16", corpus,
          baseline_outputs);
  ValidateSigmoidApproximationCandidate<
      SigmoidBenchImplementation::FloatLutMinus8To8Step32>(
          "C. float32 LUT [-8, 8], step 1/32", corpus,
          baseline_outputs);
  ValidateSigmoidApproximationCandidate<
      SigmoidBenchImplementation::FloatLutMinus10To8Step32>(
          "D. float32 LUT [-10, 8], step 1/32", corpus,
          baseline_outputs);

  constexpr std::array<SigmoidBenchOperation, 4> operations = {
      SigmoidBenchOperation::MainSigmoid,
      SigmoidBenchOperation::AbsSigmoidAndValueGate,
      SigmoidBenchOperation::MainAndAbs,
      SigmoidBenchOperation::FullNetwork};
  constexpr std::array<const char*, 4> operation_names = {
      "Main sigmoid only", "FM Abs sigmoid + value gate",
      "Main + FM combined", "Network::Propagate-equivalent staged path"};
  constexpr std::array<SigmoidBenchImplementation, 5> implementations = {
      SigmoidBenchImplementation::StdExp,
      SigmoidBenchImplementation::FloatLutMinus8To8Step16,
      SigmoidBenchImplementation::FloatLutMinus10To8Step16,
      SigmoidBenchImplementation::FloatLutMinus8To8Step32,
      SigmoidBenchImplementation::FloatLutMinus10To8Step32};
  std::array<std::array<NnueBenchSamples, 5>, 4> samples;
  std::array<std::array<std::uint64_t, 5>, 4> checksums;
  for (auto& operation_checksums : checksums)
    operation_checksums.fill(UINT64_C(14695981039346656037));

  for (std::size_t operation_index = 0;
       operation_index < operations.size(); ++operation_index) {
    for (std::uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
      for (std::size_t offset = 0; offset < implementations.size(); ++offset) {
        const std::size_t implementation_index =
            (static_cast<std::size_t>(repeat) + offset)
            % implementations.size();
        samples[operation_index][implementation_index].Add(
            DispatchSigmoidApproximationOperation(
                implementations[implementation_index],
                operations[operation_index], corpus,
                checksums[operation_index][implementation_index]));
      }
    }
  }

  std::cout << "[timing]" << std::endl;
  for (std::size_t operation_index = 0;
       operation_index < operations.size(); ++operation_index) {
    PrintSigmoidTimingComparison(operation_names[operation_index],
                                 samples[operation_index]);
    std::cout << "  checksums reference/A/B/C/D: 0x" << std::hex
              << checksums[operation_index][0] << " / 0x"
              << checksums[operation_index][1] << " / 0x"
              << checksums[operation_index][2] << " / 0x"
              << checksums[operation_index][3] << " / 0x"
              << checksums[operation_index][4] << std::dec << std::endl;
  }
  std::cout << "  note: the full-network timing uses the same benchmark-only"
            << std::endl
            << "        staged path for reference/A/B/C/D; production Propagate()"
            << std::endl
            << "        is unchanged."
            << std::endl;
}

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
constexpr std::size_t kTraceBucketInputDimensions = L2_INPUT_SIZE;
constexpr std::size_t kTraceBucketHiddenDimensions = kHidden2Dims;

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
  std::array<std::int32_t, kTracePhaseDimensions> channel_scale_q23;
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
  std::copy_n(lca_buffer.lca_q_out, LCA_QK_SIZE,
              lca.query_preact.begin());
  std::copy_n(lca_buffer.lca_k_out, LCA_QK_SIZE,
              lca.key_preact.begin());
  std::copy_n(lca_buffer.lca_v_out, LCA_VALUE_SIZE,
              lca.value_preact.begin());

  float lca_dot_product = 0.0f;
  for (std::size_t index = 0; index < LCA_QK_SIZE; ++index) {
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
    const float value = static_cast<float>(Network::LcaValueForDiffChannel(
        lca_buffer.lca_v_out, static_cast<IndexType>(index))) / 8128.0f;
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

  selected_network->phase_proj.PropagatePrefix<PHASE_OUTPUT_SIZE>(
      lca_buffer.phase_input, lca_buffer.phase_out);
  float channel_scales[kTracePhaseDimensions];
#if defined(NNUE_COMPACT_PHASE5)
  constexpr float scale_multipliers[PHASE_OUTPUT_SIZE] = {
      1.3f, 1.5f, 1.0f, 0.7f, 1.5f};
  constexpr std::size_t semantic_channels[PHASE_OUTPUT_SIZE] = {
      0, 1, 2, 3, 5};
#else
  constexpr float scale_multipliers[kTracePhaseDimensions] = {
      1.3f, 1.5f, 1.0f, 0.7f, 0.88f, 1.5f};
#endif
  std::fill_n(channel_scales, kTracePhaseDimensions, 0.0f);
  for (std::size_t index = 0; index < PHASE_OUTPUT_SIZE; ++index) {
    const float phase_logit =
        (static_cast<float>(lca_buffer.phase_out[index]) / 8128.0f)
        * 3.0f + 1.0f;
    const float phase_sigmoid =
        1.0f / (1.0f + std::exp(-phase_logit));
    const float phase_value = 0.1f + 0.9f * phase_sigmoid;
    const float channel_scale =
        (0.5f + 0.5f * phase_value) * scale_multipliers[index];

#if defined(NNUE_COMPACT_PHASE5)
    const std::size_t semantic_index = semantic_channels[index];
#else
    const std::size_t semantic_index = index;
#endif
    deep_path.phase_preact[semantic_index] = lca_buffer.phase_out[index];
    deep_path.phase_logit_f32_bits[semantic_index] = TraceFloatBits(phase_logit);
    deep_path.phase_sigmoid_f32_bits[semantic_index] = TraceFloatBits(phase_sigmoid);
    deep_path.phase_value_f32_bits[semantic_index] = TraceFloatBits(phase_value);
    deep_path.channel_scale_f32_bits[semantic_index] = TraceFloatBits(channel_scale);
    channel_scales[semantic_index] = channel_scale;
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
  selected_network->fc_cross.PropagatePrefix<CROSS_OUTPUT_SIZE>(
      lca_buffer.cross_cat, lca_buffer.cross_fc_out);
  selected_network->PropagateCrossActivation(lca_buffer.cross_fc_out,
                                              lca_buffer.cross_feat);
  std::fill(deep_path.cross_preact.begin(), deep_path.cross_preact.end(), 0);
  std::fill(deep_path.cross_output.begin(), deep_path.cross_output.end(), 0);
  std::copy_n(lca_buffer.cross_fc_out, CROSS_OUTPUT_SIZE,
              deep_path.cross_preact.begin());
  std::copy_n(lca_buffer.cross_feat, CROSS_OUTPUT_SIZE,
              deep_path.cross_output.begin());

#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
  std::int32_t phase_scales_q23[PHASE_OUTPUT_SIZE];
  selected_network->BenchmarkPhaseFixedScalesQ23(
      lca_buffer.phase_out, phase_scales_q23);
  std::fill(deep_path.channel_scale_q23.begin(),
            deep_path.channel_scale_q23.end(), 0);
  for (std::size_t index = 0; index < PHASE_OUTPUT_SIZE; ++index) {
#if defined(NNUE_COMPACT_PHASE5)
    const std::size_t semantic_index = semantic_channels[index];
#else
    const std::size_t semantic_index = index;
#endif
    deep_path.channel_scale_q23[semantic_index] = phase_scales_q23[index];
  }
  selected_network->BenchmarkL2AssemblyQ23(
      lca_buffer.ac_sqr_0_out_temp, lca_buffer.ac_0_out,
      lca_buffer.diff_ac_out, lca_buffer.abs_ac_out,
      lca_buffer.abs_sqr_out, lca_buffer.cross_feat,
      phase_scales_q23, lca_buffer.l2_input);
#else
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
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
    lca_buffer.l2_input[126 + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.abs_sqr_out[index] * channel_scales[4],
                        0, 127));
#endif
  }
  for (IndexType index = 0; index < CROSS_OUTPUT_SIZE; ++index)
    lca_buffer.l2_input[L2_CROSS_OFFSET + index] = static_cast<std::uint8_t>(
        std::clamp<int>(lca_buffer.cross_feat[index] * channel_scales[5],
                        0, 127));
  std::memset(lca_buffer.l2_input + L2_REAL_SIZE, 0, L2_PADDING_SIZE);
#endif
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
  WriteTraceArray(output, "deep.phase.channel_scale_q23", "i32",
                  snapshot.deep_path.channel_scale_q23);
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

#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL)
void TestHaoRiskHeads(const Position& pos) {
  std::uint8_t activation[64];
  std::cout << std::setprecision(10)
            << "[NNUE Hao search-risk heads self-test]" << std::endl;
  for (std::size_t bucket = 0; bucket < kLayerStacks; ++bucket) {
    for (std::size_t lane = 0; lane < 64; ++lane)
      activation[lane] = static_cast<std::uint8_t>(
          (lane * 37 + bucket * 11 + 3) & 127);
    NnueSignalSnapshot sample{};
    const int game_ply = static_cast<int>(bucket * 19 + 7);
    const int material = static_cast<int>(bucket * 731) - 3500;
    const int static_eval = static_cast<int>(bucket * 613) - 3000;
    network[bucket]->ComputeHaoSearchRiskSignals(
      activation, game_ply, material, static_eval, &sample);
    std::cout << "bucket=" << bucket << " game_ply=" << game_ply
              << " material=" << material << " static_eval=" << static_eval;
    for (std::size_t kind = 0; kind < NnueSignalSnapshot::HaoRiskCount; ++kind)
      std::cout << " logit" << kind << '=' << sample.hao_risk_logit[kind]
                << " probability" << kind << '=' << sample.hao_risk_probability[kind]
                << " q8_" << kind << '=' << static_cast<unsigned>(sample.hao_risk_q8[kind]);
    std::cout << std::endl;
  }
  const Value value = Eval::evaluate(pos);
  const auto& access = LastNnueSignalAccess();
  std::cout << "current_position eval=" << value
            << " source=" << static_cast<unsigned>(access.source)
            << " valid=" << (access.signal.valid ? 1 : 0);
  if (access.signal.valid) {
    std::cout << " bucket=" << access.signal.selected_bucket;
    for (std::size_t kind = 0; kind < NnueSignalSnapshot::HaoRiskCount; ++kind)
      std::cout << " q8_" << kind << '='
                << static_cast<unsigned>(access.signal.hao_risk_q8[kind]);
  }
  std::cout << std::endl;
}
#endif

#if defined(ENABLE_NNUE_UNCERTAINTY_SIGNAL)
void TestUncertaintyHead(const Position& pos) {
  std::uint8_t activation[64];
  std::cout << std::setprecision(10)
            << "[NNUE uncertainty head self-test]" << std::endl;
  for (std::size_t bucket = 0; bucket < kLayerStacks; ++bucket) {
    for (std::size_t lane = 0; lane < 64; ++lane)
      activation[lane] = static_cast<std::uint8_t>(
          (lane * 37 + bucket * 11 + 3) & 127);
    float logit = 0.0f;
    float probability = 0.0f;
    const auto q8 = network[bucket]->ComputeUncertaintySignal(
        activation, &logit, &probability);
    std::cout << "bucket=" << bucket << " logit=" << logit
              << " probability=" << probability
              << " q8=" << static_cast<unsigned>(q8) << std::endl;
  }

  const Value value = Eval::evaluate(pos);
  const auto& access = LastNnueSignalAccess();
  const bool valid = access.signal.valid;
  std::cout << "current_position eval=" << value
            << " source=" << static_cast<unsigned>(access.source)
            << " valid=" << (valid ? 1 : 0);
  if (valid)
    std::cout << " bucket=" << access.signal.selected_bucket
              << " logit=" << access.signal.uncertainty_logit
              << " probability=" << access.signal.uncertainty_probability
              << " q8=" << static_cast<unsigned>(access.signal.uncertainty_q8);
  std::cout << std::endl;
}
#endif

#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
void TestSideInput(const Position& pos, std::uint64_t repeats) {
  repeats = std::max<std::uint64_t>(repeats, 1);
  const std::uint16_t mask = NnueSideInput::safe_escape_mask16(pos);
  std::array<float, Network::kSideInputDimensions> encoded{};
  std::array<float, L2_INPUT_SIZE> residual{};
  network[0]->ComputeSideInputResidual(
      mask, encoded.data(), residual.data());
  std::cout << std::setprecision(10)
            << "side_input mask16=" << mask << " encoded=";
  for (const float value : encoded)
    std::cout << value << ',';
  std::cout << " residual=";
  for (const float value : residual)
    std::cout << value << ',';
  std::cout << std::endl;

  volatile std::uint64_t mask_checksum = 0;
  auto begin = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < repeats; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(&pos) : "memory");
#endif
    mask_checksum += NnueSideInput::safe_escape_mask16(pos);
  }
  auto middle = std::chrono::steady_clock::now();
  volatile float projection_checksum = 0.0f;
  for (std::uint64_t i = 0; i < repeats; ++i) {
    network[0]->ComputeSideInputResidual(
        mask, encoded.data(), residual.data());
    projection_checksum += residual[i % residual.size()];
  }
  auto end = std::chrono::steady_clock::now();
  const double mask_ns = std::chrono::duration<double, std::nano>(
      middle - begin).count() / repeats;
  const double projection_ns = std::chrono::duration<double, std::nano>(
      end - middle).count() / repeats;
  std::cout << "side_input_cost repeats=" << repeats
            << " mask_ns=" << mask_ns
            << " projection_ns=" << projection_ns
            << " total_ns=" << (mask_ns + projection_ns)
            << " checksum=" << mask_checksum << ':' << projection_checksum
            << std::endl;
}

void TestSideInputRecord(std::istream& stream) {
  std::string input_name;
  std::uint64_t record_index = 0;
  stream >> std::quoted(input_name) >> record_index;
  std::ifstream input(input_name, std::ios::binary);
  if (!input) {
    std::cout << "error: side_input_record cannot open " << input_name
              << std::endl;
    return;
  }
  input.seekg(static_cast<std::streamoff>(
      record_index * sizeof(MoveAccuracyRecord)), std::ios::beg);
  MoveAccuracyRecord record{};
  if (!input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    std::cout << "error: side_input_record cannot read record" << std::endl;
    return;
  }
  Position position;
  StateInfo state;
  if (position.set_from_packed_sfen(
          record.sfen, &state, false, record.game_ply).is_not_ok()) {
    std::cout << "error: side_input_record decode failed" << std::endl;
    return;
  }
  const auto mask = NnueSideInput::safe_escape_mask16(position);
  const auto value = Eval::evaluate(position);
  std::cout << "side_input_record index=" << record_index
            << " mask16=" << mask << " eval=" << value
            << " sfen=" << position.sfen() << std::endl;
}

void FindSideInputRecord(std::istream& stream) {
  std::string input_name;
  int wanted_score = 0;
  int wanted_ply = 0;
  unsigned wanted_mask = 0;
  stream >> std::quoted(input_name) >> wanted_score >> wanted_ply >> wanted_mask;
  std::ifstream input(input_name, std::ios::binary);
  MoveAccuracyRecord record{};
  std::uint64_t index = 0;
  while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    if (record.score != wanted_score || record.game_ply != wanted_ply) {
      ++index;
      continue;
    }
    Position position;
    StateInfo state;
    if (position.set_from_packed_sfen(
            record.sfen, &state, false, record.game_ply).is_ok()
        && NnueSideInput::safe_escape_mask16(position) == wanted_mask) {
      std::cout << "side_input_find index=" << index
                << " mask16=" << wanted_mask
                << " eval=" << Eval::evaluate(position)
                << " score=" << record.score << " ply=" << record.game_ply
                << std::endl;
      return;
    }
    ++index;
  }
  std::cout << "error: side_input_find no matching record" << std::endl;
}
#endif

#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
void TestPairRelation(const Position& pos, const std::uint64_t repeats) {
  std::array<std::uint16_t, NnuePairRelation::MaxRelations> indices{};
  const auto generated = NnuePairRelation::generate(
      pos, indices.data(), indices.size());
  const auto count = std::min(generated, indices.size());
  std::cout << "pair_relation count=" << count;
  if (generated > indices.size())
    std::cout << " truncated_from=" << generated;
  std::cout << " indices=";
  for (std::size_t i = 0; i < count; ++i)
    std::cout << (i ? "," : "") << indices[i];
  std::cout << std::endl;

  volatile std::uint64_t checksum = 0;
  const auto generation_start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < repeats; ++i)
    checksum += NnuePairRelation::generate(
        pos, indices.data(), indices.size());
  const auto generation_end = std::chrono::steady_clock::now();
  std::array<float, 64> delta{};
  const auto projection_start = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < repeats; ++i) {
    network[0]->ComputePairRelationDelta(indices.data(), count, delta.data());
    checksum += static_cast<std::uint64_t>(std::abs(delta[i % 64]) * 1000.0f);
  }
  const auto projection_end = std::chrono::steady_clock::now();
  const auto generation_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      generation_end - generation_start).count();
  const auto projection_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      projection_end - projection_start).count();
  std::cout << "pair_relation_cost repeats=" << repeats
            << " generation_ns=" << double(generation_ns) / repeats
            << " aggregate_ln_proj_ns=" << double(projection_ns) / repeats
            << " checksum=" << checksum << std::endl;
}

template <typename T, std::size_t N>
void PrintPairVector(const char* name, const std::array<T, N>& values) {
  std::cout << "pair_stage " << name << "=";
  std::cout << std::setprecision(9);
  for (std::size_t i = 0; i < N; ++i)
    std::cout << (i ? "," : "") << +values[i];
  std::cout << std::endl;
}

void TestPairRelationStages(const Position& pos) {
  std::array<std::uint16_t, NnuePairRelation::MaxRelations> indices{};
  const auto generated = NnuePairRelation::generate(
      pos, indices.data(), indices.size());
  const auto count = std::min(generated, indices.size());
  std::array<float, 32> pooled{};
  std::array<float, 32> normalized{};
  std::array<float, 64> projected{};
  std::array<float, 64> gated{};
  float gate = 0.0f;
  network[0]->ComputePairRelationStages(
      indices.data(), count, pooled.data(), normalized.data(),
      projected.data(), gated.data(), &gate);
  std::cout << "pair_stage count=" << count << " indices=";
  for (std::size_t i = 0; i < count; ++i)
    std::cout << (i ? "," : "") << indices[i];
  std::cout << std::endl;
  PrintPairVector("embedding_sum", pooled);
  PrintPairVector("layer_norm", normalized);
  PrintPairVector("projection", projected);
  std::cout << "pair_stage sigmoid_gate=" << std::setprecision(9) << gate
            << std::endl;
  PrintPairVector("pair_delta", gated);
  std::array<std::int32_t, 64> injected_pre_activation{};
  std::array<std::uint8_t, 64> injected_activation{};
  constexpr float kPairPreActivationScale = 127.0f * 64.0f;
  for (std::size_t i = 0; i < injected_pre_activation.size(); ++i) {
    // A deterministic affine baseline spanning the unclipped and clipped
    // regions makes the injection boundary independently testable.
    const std::int32_t baseline = (static_cast<std::int32_t>(i) - 16) * 512;
    injected_pre_activation[i] = baseline + static_cast<std::int32_t>(
        std::round(gated[i] * kPairPreActivationScale));
  }
  for (std::size_t i = 0; i < injected_activation.size(); ++i)
    injected_activation[i] = static_cast<std::uint8_t>(std::clamp(
        injected_pre_activation[i] >> 6, 0, 127));
  PrintPairVector("injected_pre_activation", injected_pre_activation);
  PrintPairVector("activation_after_single_clip", injected_activation);
  std::cout << "pair_stage final_eval_cp=" << Eval::evaluate(pos) << std::endl;
}

// Diagnostic-only corpus sweep for checking whether the very small learned
// float residual survives conversion to the int32 FC1 pre-activation domain.
// It does not alter the network, position, or search behavior.
void TestPairRelationQuantization(std::istream& stream) {
  std::string file_name;
  stream >> file_name;
  std::ifstream input(file_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open sfenpack file: " << file_name << std::endl;
    return;
  }

  constexpr std::array<int, 6> kScales = {1, 2, 4, 8, 16, 32};
  struct ScaleStats {
    std::uint64_t all_zero_positions = 0;
    std::uint64_t nonzero_channels = 0;
    std::uint64_t hypothetical_full_range_clips = 0;
    std::array<std::uint64_t, 65> nonzero_histogram{};
    std::int32_t max_abs = 0;
  };
  std::array<ScaleStats, kScales.size()> scale_stats{};
  std::uint64_t positions = 0;
  std::uint64_t total_channels = 0;
  long double raw_abs_sum = 0.0;
  std::vector<float> raw_abs_delta;
#if defined(ENABLE_NNUE_PAIR_RELATION_POST_ACTIVATION_DIAGNOSTIC)
  PairPostActivationDiagnostic post_activation{};
#endif
  constexpr float kPairPreActivationScale = 127.0f * 64.0f;
  MoveAccuracyRecord record{};
  while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    if (30000 < std::abs(static_cast<int>(record.score))
        || record.game_result == 0)
      continue;
    Position pos;
    StateInfo state;
    if (pos.set_from_packed_sfen(record.sfen, &state, false).is_not_ok())
      continue;
    if (MoveList<LEGAL>(pos).size() == 0)
      continue;

    std::array<std::uint16_t, NnuePairRelation::MaxRelations> indices{};
    const auto generated = NnuePairRelation::generate(
        pos, indices.data(), indices.size());
    const auto count = std::min(generated, indices.size());
    std::array<float, 64> delta{};
    network[0]->ComputePairRelationDelta(
        indices.data(), count, delta.data());
    for (const float value : delta) {
      const float magnitude = std::abs(value);
      raw_abs_sum += magnitude;
      raw_abs_delta.push_back(magnitude);
      ++total_channels;
    }
    for (std::size_t scale_index = 0; scale_index < kScales.size();
         ++scale_index) {
      auto& stats = scale_stats[scale_index];
      std::uint32_t position_nonzero = 0;
      for (const float value : delta) {
        const auto quantized = static_cast<std::int32_t>(std::round(
            value * kPairPreActivationScale * kScales[scale_index]));
        const auto magnitude = static_cast<std::int32_t>(std::abs(quantized));
        position_nonzero += quantized != 0;
        stats.nonzero_channels += quantized != 0;
        // The production injection is int32 and has no clip of its own.  This
        // diagnostic counter asks whether the residual alone exceeds the
        // entire 0..1 FC1 activation range (127 activation levels * Q6).
        stats.hypothetical_full_range_clips +=
            magnitude > static_cast<std::int32_t>(kPairPreActivationScale);
        stats.max_abs = std::max(stats.max_abs, magnitude);
      }
      stats.all_zero_positions += position_nonzero == 0;
      ++stats.nonzero_histogram[position_nonzero];
    }
#if defined(ENABLE_NNUE_PAIR_RELATION_POST_ACTIVATION_DIAGNOSTIC)
    pair_post_activation_diagnostic = &post_activation;
    (void) Eval::compute_eval(pos);
    pair_post_activation_diagnostic = nullptr;
#endif
    ++positions;
  }
  if (positions == 0) {
    std::cout << "error: no positions passed pair quantization filters"
              << std::endl;
    return;
  }
  std::sort(raw_abs_delta.begin(), raw_abs_delta.end());
  const auto raw_percentile = [&](const double percentile) {
    const auto index = static_cast<std::size_t>(std::floor(
        percentile * static_cast<double>(raw_abs_delta.size() - 1)));
    return raw_abs_delta[index];
  };
  const auto histogram_percentile = [&](const ScaleStats& stats,
                                        const double percentile) {
    const auto target = static_cast<std::uint64_t>(std::ceil(
        percentile * static_cast<double>(positions)));
    std::uint64_t cumulative = 0;
    for (std::size_t value = 0; value < stats.nonzero_histogram.size(); ++value) {
      cumulative += stats.nonzero_histogram[value];
      if (cumulative >= target)
        return value;
    }
    return stats.nonzero_histogram.size() - 1;
  };
  std::cout << std::fixed << std::setprecision(9)
            << "pair_quant positions=" << positions << std::endl
            << "pair_quant channels=" << total_channels << std::endl
            << "pair_quant preactivation_fixed_lsb_python="
            << 1.0 / kPairPreActivationScale << std::endl
            << "pair_quant activation_output_lsb_python=" << 1.0 / 127.0
            << std::endl
            << "pair_quant injection_has_explicit_clip=0" << std::endl
            << "pair_delta_abs mean="
            << static_cast<double>(raw_abs_sum / total_channels)
            << " p50=" << raw_percentile(0.50)
            << " p90=" << raw_percentile(0.90)
            << " p99=" << raw_percentile(0.99)
            << " max=" << raw_abs_delta.back() << std::endl;
  for (std::size_t scale_index = 0; scale_index < kScales.size();
       ++scale_index) {
    const auto& stats = scale_stats[scale_index];
    std::cout << "pair_scale S=" << kScales[scale_index]
              << " all64_zero=" << stats.all_zero_positions
              << " all64_zero_rate="
              << double(stats.all_zero_positions) / positions
              << " mean_nonzero="
              << double(stats.nonzero_channels) / positions
              << " p50_nonzero=" << histogram_percentile(stats, 0.50)
              << " p90_nonzero=" << histogram_percentile(stats, 0.90)
              << " p99_nonzero=" << histogram_percentile(stats, 0.99)
              << " max_abs_fixed=" << stats.max_abs
              << " hypothetical_full_range_clip_rate="
              << double(stats.hypothetical_full_range_clips) / total_channels
              << std::endl;
  }
#if defined(ENABLE_NNUE_PAIR_RELATION_POST_ACTIVATION_DIAGNOSTIC)
  const auto print_post_activation = [&](const int scale,
      const PairPostActivationScaleStats& stats) {
    const auto percentile = [&](const double p) {
      const auto target = static_cast<std::uint64_t>(
          std::ceil(p * static_cast<double>(stats.positions)));
      std::uint64_t cumulative = 0;
      for (std::size_t value = 0; value < stats.changed_histogram.size(); ++value) {
        cumulative += stats.changed_histogram[value];
        if (cumulative >= target)
          return value;
      }
      return stats.changed_histogram.size() - 1;
    };
    const auto channels = stats.positions * 64;
    std::cout << "pair_post_activation S=" << scale
              << " positions=" << stats.positions
              << " changed_positions=" << stats.changed_positions
              << " changed_position_rate="
              << double(stats.changed_positions) / stats.positions
              << " mean_changed_channels="
              << double(stats.changed_channels) / stats.positions
              << " p50_changed=" << percentile(0.50)
              << " p90_changed=" << percentile(0.90)
              << " p99_changed=" << percentile(0.99)
              << " max_changed=" << percentile(1.0)
              << " positive_changes=" << stats.positive_changes
              << " negative_changes=" << stats.negative_changes
              << " zero_boundary_changes=" << stats.zero_boundary_changes
              << " high_boundary_changes=" << stats.high_boundary_changes
              << " clip_zero_rate="
              << double(stats.clipped_zero_channels) / channels
              << " clip_127_rate="
              << double(stats.clipped_high_channels) / channels
              << std::endl;
  };
  print_post_activation(1, post_activation.current);
  const auto baseline_channels = post_activation.current.positions * 64;
  std::cout << "pair_post_activation baseline"
            << " clip_zero_rate="
            << double(post_activation.baseline_clipped_zero_channels) /
                   baseline_channels
            << " clip_127_rate="
            << double(post_activation.baseline_clipped_high_channels) /
                   baseline_channels
            << std::endl;
  print_post_activation(8, post_activation.scale8);
  print_post_activation(16, post_activation.scale16);
#endif
}
#endif

void TestFreshEvaluateCost(const Position& pos, const std::uint64_t repeats) {
  volatile std::int64_t checksum = 0;
  const auto started = std::chrono::steady_clock::now();
  for (std::uint64_t i = 0; i < repeats; ++i)
    checksum += static_cast<std::int64_t>(Eval::compute_eval(pos));
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "nnue_fresh_eval_cost repeats=" << repeats
            << " ns_per_eval=" << double(elapsed) / repeats
            << " checksum=" << checksum << std::endl;
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
  } else if (sub_command == "incremental_eval_checksum") {
    TestIncrementalEvalChecksum();
  } else if (sub_command == "fresh_eval_cost") {
    std::uint64_t repeats = 100000;
    stream >> repeats;
    TestFreshEvaluateCost(pos, repeats);
#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
  } else if (sub_command == "side_input_selftest") {
    std::uint64_t repeats = 100000;
    stream >> repeats;
    TestSideInput(pos, repeats);
  } else if (sub_command == "side_input_record") {
    TestSideInputRecord(stream);
  } else if (sub_command == "side_input_find") {
    FindSideInputRecord(stream);
#endif
#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
  } else if (sub_command == "pair_relation_selftest") {
    std::uint64_t repeats = 100000;
    stream >> repeats;
    TestPairRelation(pos, repeats);
  } else if (sub_command == "pair_relation_stages") {
    TestPairRelationStages(pos);
  } else if (sub_command == "pair_relation_quantization") {
    TestPairRelationQuantization(stream);
#endif
#if defined(ENABLE_NNUE_DECISION_RISK_LMR_COUNTERS)
  } else if (sub_command == "decision_risk_lmr_counters_reset") {
    Search::NnueDecisionRiskLmrCounters::Reset();
    std::cout << "NNUE decision-risk restricted LMR counters reset." << std::endl;
  } else if (sub_command == "decision_risk_lmr_counters_report") {
    Search::NnueDecisionRiskLmrCounters::Report(std::cout);
#endif
#if defined(ENABLE_NNUE_ASPIRATION_DIAGNOSTIC)
  } else if (sub_command == "aspiration_log_reset") {
    Search::NnueAspirationLog::Reset();
    std::cout << "NNUE aspiration diagnostics reset." << std::endl;
  } else if (sub_command == "aspiration_log_report") {
    std::string csv_file;
    stream >> std::quoted(csv_file);
    Search::NnueAspirationLog::Report(std::cout);
    if (!csv_file.empty()) {
      if (Search::NnueAspirationLog::WriteCsv(csv_file.c_str()))
        std::cout << "NNUE aspiration CSV written: " << csv_file << std::endl;
      else
        std::cout << "Failed to write aspiration CSV: " << csv_file << std::endl;
    }
#endif
#if defined(ENABLE_NNUE_ADAPTIVE_ASPIRATION_COUNTERS)
  } else if (sub_command == "adaptive_aspiration_counters_reset") {
    Search::AdaptiveAspirationCounters::Reset();
    std::cout << "Adaptive aspiration counters reset." << std::endl;
  } else if (sub_command == "adaptive_aspiration_counters_report") {
    Search::AdaptiveAspirationCounters::Report(std::cout);
#endif
#if defined(ENABLE_NNUE_TT_REUSE_DIAGNOSTIC)
  } else if (sub_command == "tt_reuse_log_reset") {
    Search::NnueTtReuseLog::Reset();
    std::cout << "NNUE TT reuse diagnostics reset." << std::endl;
  } else if (sub_command == "tt_reuse_log_report") {
    std::string csv_file;
    stream >> std::quoted(csv_file);
    Search::NnueTtReuseLog::Report(std::cout);
    if (!csv_file.empty() && Search::NnueTtReuseLog::WriteCsv(csv_file.c_str()))
      std::cout << "NNUE TT reuse CSV written: " << csv_file << std::endl;
#endif
#if defined(ENABLE_ROOT_MOVE_HISTORY_DIAGNOSTIC)
  } else if (sub_command == "root_move_history_reset") {
    Search::RootMoveHistoryLog::Reset();
    std::cout << "Root move history diagnostics reset." << std::endl;
  } else if (sub_command == "root_move_history_report") {
    std::string csv_file;
    stream >> std::quoted(csv_file);
    Search::RootMoveHistoryLog::Report(std::cout);
    if (!csv_file.empty() && Search::RootMoveHistoryLog::WriteCsv(csv_file.c_str()))
      std::cout << "Root move history CSV written: " << csv_file << std::endl;
#endif
#if defined(ENABLE_ROOT_TIME_RISK_BUDGET)
  } else if (sub_command == "root_time_risk_budget_reset") {
    Search::RootTimeRiskBudget::Reset();
    std::cout << "Root time risk budget counters reset." << std::endl;
  } else if (sub_command == "root_time_risk_budget_report") {
    std::string csv_file;
    stream >> std::quoted(csv_file);
    Search::RootTimeRiskBudget::Report(std::cout);
    if (!csv_file.empty() && Search::RootTimeRiskBudget::WriteCsv(csv_file.c_str()))
      std::cout << "Root time risk budget CSV written: " << csv_file << std::endl;
#endif
  } else if (sub_command == "info") {
    PrintInfo(stream);
  } else if (sub_command == "accuracy") {
    TestMoveAccuracy(engine, stream, false);
  } else if (sub_command == "accuracy_detail") {
    TestMoveAccuracy(engine, stream, true);
#if defined(ENABLE_NNUE_UNCERTAINTY_SIGNAL)
  } else if (sub_command == "uncertainty_head_selftest") {
    TestUncertaintyHead(pos);
#endif
#if defined(ENABLE_NNUE_POLICY_SHADOW)
  } else if (sub_command == "policy_probe_selftest") {
    PolicyProbeSelftest(stream);
#endif
#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL)
  } else if (sub_command == "hao_risk_head_selftest") {
    TestHaoRiskHeads(pos);
  } else if (sub_command == "hao_root_diagnostic_reset") {
    Search::NnueHaoRiskLog::ResetRootCompact();
    std::cout << "NNUE Hao root diagnostics reset." << std::endl;
  } else if (sub_command == "hao_root_diagnostic_summary") {
    Search::NnueHaoRiskLog::ReportRootSummary(std::cout);
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
  } else if (sub_command == "signal_log_reset") {
    Search::NnueSignalLog::Reset();
    std::cout << "NNUE search signal diagnostics reset." << std::endl;
  } else if (sub_command == "signal_log_report") {
    Search::NnueSignalLog::Report(std::cout);
    std::string output_file;
    stream >> std::quoted(output_file);
    if (!output_file.empty()) {
      std::ofstream output(output_file);
      if (!output)
        std::cout << "Failed to open signal report: " << output_file << std::endl;
      else {
        Search::NnueSignalLog::Report(output);
        std::cout << "NNUE search signal report written: " << output_file << std::endl;
      }
    }
  } else if (sub_command == "signal_calibration_report") {
    std::string text_file;
    std::string json_file;
    stream >> std::quoted(text_file) >> std::quoted(json_file);
    Search::NnueSignalLog::CalibrationReport(std::cout);
    if (!text_file.empty()) {
      std::ofstream output(text_file);
      if (!output)
        std::cout << "Failed to open calibration report: " << text_file << std::endl;
      else {
        Search::NnueSignalLog::CalibrationReport(output);
        std::cout << "NNUE signal calibration report written: " << text_file << std::endl;
      }
    }
    if (!json_file.empty()) {
      std::ofstream output(json_file);
      if (!output)
        std::cout << "Failed to open calibration JSON: " << json_file << std::endl;
      else {
        Search::NnueSignalLog::CalibrationReportJson(output);
        std::cout << "NNUE signal calibration JSON written: " << json_file << std::endl;
      }
    }
#if defined(ENABLE_NNUE_EVAL_HISTORY_DIAGNOSTIC)
  } else if (sub_command == "eval_history_report") {
    std::string text_file;
    std::string csv_file;
    stream >> std::quoted(text_file) >> std::quoted(csv_file);
    Search::NnueEvalHistoryLog::Report(std::cout);
    if (!text_file.empty()) {
      std::ofstream output(text_file);
      if (!output)
        std::cout << "Failed to open eval-history report: " << text_file << std::endl;
      else
        Search::NnueEvalHistoryLog::Report(output);
    }
    if (!csv_file.empty()) {
      std::ofstream output(csv_file);
      if (!output)
        std::cout << "Failed to open eval-history CSV: " << csv_file << std::endl;
      else
        Search::NnueEvalHistoryLog::ReportCsv(output);
    }
#endif
#if defined(ENABLE_NNUE_DECISION_TRACE)
  } else if (sub_command == "trace_extract_roots") {
    ExtractDecisionTraceRoots(stream);
  } else if (sub_command == "decision_trace_start") {
    std::string prefix;
    std::uint64_t chunk_records = 16384;
    stream >> std::quoted(prefix) >> chunk_records;
    const bool started = Search::NnueDecisionTrace::Start(
      prefix, static_cast<std::size_t>(chunk_records));
    std::cout << "NNUE decision trace start: "
              << (started ? "ok" : "failed") << std::endl;
  } else if (sub_command == "decision_trace_stop") {
    const bool stopped = Search::NnueDecisionTrace::Stop();
    std::cout << "NNUE decision trace stop: "
              << (stopped ? "ok" : "failed") << std::endl;
  } else if (sub_command == "decision_trace_report") {
    Search::NnueDecisionTrace::Report(std::cout);
#if defined(ENABLE_NNUE_DECISION_RISK_SHADOW)
  } else if (sub_command == "decision_risk_selftest") {
    Search::NnueDecisionTrace::RiskSelftest(std::cout);
#endif
#endif
#endif
#if defined(ENABLE_NNUE_POLICY_SHADOW)
  } else if (sub_command == "policy_shadow_reset") {
    Search::NnuePolicyShadow::Reset();
    std::cout << "NNUE policy shadow diagnostics reset." << std::endl;
  } else if (sub_command == "policy_shadow_report") {
    std::string summary_file;
    std::string raw_file;
    stream >> std::quoted(summary_file) >> std::quoted(raw_file);
    Search::NnuePolicyShadow::Report(std::cout);
    if (!summary_file.empty()) {
      std::ofstream output(summary_file);
      Search::NnuePolicyShadow::Report(output);
    }
    if (!raw_file.empty()) {
      std::ofstream output(raw_file);
      Search::NnuePolicyShadow::RawReport(output);
    }
#endif
#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
  } else if (sub_command == "make_static_eval_bin") {
    MakeStaticEvalBin(stream);
#endif
#if defined(ENABLE_QSEARCH_CORRECTION_PROBE)
  } else if (sub_command == "qsearch_correction_corpus") {
    MakeQsearchCorrectionCorpus(engine, stream);
  } else if (sub_command == "qsearch_correction_decomposition") {
    MakeQsearchCorrectionDecompositionCorpus(engine, stream);
#endif
#if defined(ENABLE_NNUE_BENCH)
  } else if (sub_command == "inspect_packed_sfen_sample") {
    InspectPackedSfenSample(stream);
  } else if (sub_command == "export_shogi_tactical_features") {
    ExportShogiTacticalFeatures(stream);
  } else if (sub_command == "export_cheap_safe_escape_features") {
    ExportCheapSafeEscapeFeatures(stream);
  } else if (sub_command == "benchmark_cheap_safe_escape_features") {
    BenchmarkCheapSafeEscapeFeatures(stream);
  } else if (sub_command == "export_calibration_corpus") {
    ExportNnueCalibrationCorpus(stream);
  } else if (sub_command == "bench_ft") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFeatureTransformerBenchmark(repeat_count);
  } else if (sub_command == "bench_multi_delta_accumulator") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestMultiDeltaAccumulatorBenchmark(repeat_count);
  } else if (sub_command == "bench_ksdg3_features") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestKsdg3FeaturesBenchmark(repeat_count);
  } else if (sub_command == "bench_long_effect_changed_mask") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestLongEffectChangedMaskBenchmark(repeat_count);
  } else if (sub_command == "long_effect_mask_variant") {
    SelectLongEffectMaskVariant(stream);
  } else if (sub_command == "ksdg3_variant") {
    SelectKsdg3BenchmarkVariant(stream);
#if defined(USE_FINNY_TABLES)
  } else if (sub_command == "bench_finny_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFinnyBenchmarkCompare(repeat_count);
  } else if (sub_command == "bench_finny_fm_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFinnyFmBenchmarkCompare(repeat_count);
#endif
  } else if (sub_command == "bench_ft_transform_stages") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFeatureTransformerStagesBenchmark(repeat_count);
  } else if (sub_command == "bench_pairweight_perspective_reuse") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestPairWeightPerspectiveReuseBenchmark(repeat_count);
#if defined(USE_AVX2)
  } else if (sub_command == "bench_ft_fm_scaling_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestFeatureTransformerFmScalingCompare(repeat_count);
#endif
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
  } else if (sub_command == "bench_router_phase_input_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestRouterPhaseInputBenchmarkCompare(repeat_count);
  } else if (sub_command == "bench_phase_l2_fixed_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestPhaseL2FixedBenchmarkCompare(repeat_count);
  } else if (sub_command == "bench_main_gate_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestMainGateBenchmarkCompare(repeat_count);
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
  } else if (sub_command == "bench_main_gate_integer_compare") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestMainGateIntegerBenchmarkCompare(repeat_count);
#endif
  } else if (sub_command == "bench_sigmoid_approx") {
    std::uint64_t repeat_count;
    if (ReadNnueBenchRepeatCount(stream, repeat_count))
      TestSigmoidApproximationBenchmark(repeat_count);
  } else if (sub_command == "validate_sigmoid_candidate") {
    ValidateProductionSigmoidCandidate();
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
    std::cout << " test nnue incremental_eval_checksum" << std::endl;
    std::cout << " test nnue fresh_eval_cost [repeats]" << std::endl;
#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
    std::cout << " test nnue side_input_selftest [repeats]" << std::endl;
    std::cout << " test nnue side_input_record <packed.bin> [index]"
              << std::endl;
    std::cout << " test nnue side_input_find <packed.bin> <score> <ply> <mask>"
              << std::endl;
#endif
#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
    std::cout << " test nnue pair_relation_selftest [repeats]" << std::endl;
    std::cout << " test nnue pair_relation_stages" << std::endl;
    std::cout << " test nnue pair_relation_quantization <sfenpack file>"
              << std::endl;
#endif
#if defined(ENABLE_NNUE_DECISION_RISK_LMR_COUNTERS)
    std::cout << " test nnue decision_risk_lmr_counters_reset" << std::endl;
    std::cout << " test nnue decision_risk_lmr_counters_report" << std::endl;
#endif
#if defined(ENABLE_NNUE_ASPIRATION_DIAGNOSTIC)
    std::cout << " test nnue aspiration_log_reset" << std::endl;
    std::cout << " test nnue aspiration_log_report [output csv]" << std::endl;
#endif
#if defined(ENABLE_NNUE_ADAPTIVE_ASPIRATION_COUNTERS)
    std::cout << " test nnue adaptive_aspiration_counters_reset" << std::endl;
    std::cout << " test nnue adaptive_aspiration_counters_report" << std::endl;
#endif
#if defined(ENABLE_NNUE_TT_REUSE_DIAGNOSTIC)
    std::cout << " test nnue tt_reuse_log_reset" << std::endl;
    std::cout << " test nnue tt_reuse_log_report [output csv]" << std::endl;
#endif
#if defined(ENABLE_ROOT_MOVE_HISTORY_DIAGNOSTIC)
    std::cout << " test nnue root_move_history_reset" << std::endl;
    std::cout << " test nnue root_move_history_report [output csv]" << std::endl;
#endif
#if defined(ENABLE_ROOT_TIME_RISK_BUDGET)
    std::cout << " test nnue root_time_risk_budget_reset" << std::endl;
    std::cout << " test nnue root_time_risk_budget_report [output csv]" << std::endl;
#endif
    std::cout << " test nnue accuracy <sfenpack file>" << std::endl;
    std::cout << " test nnue accuracy_detail <sfenpack file> <output.csv>"
              << std::endl;
    std::cout << " test nnue info [path/to/" << kFileName << "...]" << std::endl;
#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
    std::cout << " test nnue make_static_eval_bin <input> <output> <start>"
                 " <count> <csv> <csv_count> [metadata]" << std::endl;
#endif
#if defined(ENABLE_QSEARCH_CORRECTION_PROBE)
    std::cout << " test nnue qsearch_correction_corpus <input> <csv> <corpus>"
                 " <start> <count>" << std::endl;
    std::cout << " test nnue qsearch_correction_decomposition <input> <csv> <corpus>"
                 " <start> <count>" << std::endl;
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
    std::cout << " test nnue signal_log_reset" << std::endl;
    std::cout << " test nnue signal_log_report [output file]" << std::endl;
#endif
#if defined(ENABLE_NNUE_POLICY_SHADOW)
    std::cout << " test nnue policy_shadow_reset" << std::endl;
    std::cout << " test nnue policy_shadow_report [summary.csv] [raw.csv]" << std::endl;
#endif
#if defined(ENABLE_NNUE_UNCERTAINTY_SIGNAL)
    std::cout << " test nnue uncertainty_head_selftest" << std::endl;
#endif
#if defined(ENABLE_NNUE_POLICY_SHADOW)
    std::cout << " test nnue policy_probe_selftest <packed.bin> <output.csv> [count]" << std::endl;
#endif
#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL)
    std::cout << " test nnue hao_risk_head_selftest" << std::endl;
#endif
#if defined(ENABLE_NNUE_BENCH)
    std::cout << " test nnue export_calibration_corpus \"file\" [count]" << std::endl;
    std::cout << " test nnue bench_ft [repeats]" << std::endl;
    std::cout << " test nnue bench_multi_delta_accumulator [repeats]"
              << std::endl;
    std::cout << " test nnue bench_ksdg3_features [repeats]" << std::endl;
    std::cout << " test nnue bench_long_effect_changed_mask [repeats]"
              << std::endl;
    std::cout << " test nnue long_effect_mask_variant <0..2>" << std::endl;
    std::cout << " test nnue ksdg3_variant <0..3>" << std::endl;
#if defined(USE_FINNY_TABLES)
    std::cout << " test nnue bench_finny_compare [repeats]" << std::endl;
    std::cout << " test nnue bench_finny_fm_compare [repeats]" << std::endl;
#endif
    std::cout << " test nnue bench_ft_transform_stages [repeats]"
              << std::endl;
    std::cout << " test nnue bench_pairweight_perspective_reuse [repeats]"
              << std::endl;
#if defined(USE_AVX2)
    std::cout << " test nnue bench_ft_fm_scaling_compare [repeats]"
              << std::endl;
#endif
    std::cout << " test nnue bench_network [repeats]" << std::endl;
    std::cout << " test nnue bench_network_compare [repeats]" << std::endl;
    std::cout << " test nnue bench_network_stages [repeats]" << std::endl;
    std::cout << " test nnue bench_router_phase_input_compare [repeats]"
              << std::endl;
    std::cout << " test nnue bench_phase_l2_fixed_compare [repeats]"
              << std::endl;
    std::cout << " test nnue bench_main_gate_compare [repeats]"
              << std::endl;
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
    std::cout << " test nnue bench_main_gate_integer_compare [repeats]"
              << std::endl;
#endif
    std::cout << " test nnue bench_sigmoid_approx [repeats]"
              << std::endl;
    std::cout << " test nnue validate_sigmoid_candidate" << std::endl;
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
