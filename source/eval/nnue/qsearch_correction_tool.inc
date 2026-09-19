// Experiment 63 diagnostic-only batch label generator. Included inside
// YaneuraOu::Eval::NNUE's anonymous namespace after MoveAccuracyRecord.
void MakeQsearchCorrectionCorpus(IEngine& engine, std::istream& stream) {
  std::string input_name, output_csv_name, output_corpus_name;
  std::uint64_t start_record = 0, max_records = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_csv_name)
         >> std::quoted(output_corpus_name) >> start_record >> max_records;
  if (input_name.empty() || output_csv_name.empty() || output_corpus_name.empty()
      || max_records == 0) {
    std::cout << "error: qsearch_correction_corpus requires input, csv, corpus, start, count"
              << std::endl;
    return;
  }
  std::ifstream input(input_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open input: " << input_name << std::endl;
    return;
  }
  input.seekg(0, std::ios::end);
  const auto bytes = input.tellg();
  if (bytes < 0 || static_cast<std::uint64_t>(bytes) % sizeof(MoveAccuracyRecord)) {
    std::cout << "error: input size is not a multiple of 40" << std::endl;
    return;
  }
  const std::uint64_t total = static_cast<std::uint64_t>(bytes) / sizeof(MoveAccuracyRecord);
  if (start_record >= total) {
    std::cout << "error: start record out of range" << std::endl;
    return;
  }
  const std::uint64_t requested = std::min(max_records, total - start_record);
  input.seekg(static_cast<std::streamoff>(start_record * sizeof(MoveAccuracyRecord)));
  const std::string csv_tmp = output_csv_name + ".tmp";
  const std::string corpus_tmp = output_corpus_name + ".tmp";
  if (std::ifstream(output_csv_name) || std::ifstream(output_corpus_name)
      || std::ifstream(csv_tmp) || std::ifstream(corpus_tmp)) {
    std::cout << "error: output or temporary output already exists" << std::endl;
    return;
  }
  std::ofstream csv(csv_tmp, std::ios::trunc);
  std::ofstream corpus(corpus_tmp, std::ios::binary | std::ios::trunc);
  if (!csv || !corpus) {
    std::cout << "error: failed to create output" << std::endl;
    return;
  }
  csv << "row,source_record,static_eval,qsearch_eval,delta_q,abs_delta_q,side_to_move,"
         "game_ply,material_stm,bucket,in_check,legal_move_count,qsearch_nodes,"
         "qsearch_sel_depth,terminal_or_mate,router_margin,lca_abs_delta_sum,"
         "cross_abs_sum,cross_abs_max,main_gate_sum,fm_diff_activity_sum,"
         "fm_abs_activity_sum,main_reliance,fm_reliance,cross_reliance,"
         "deep_bypass_disagreement" << '\n';

  auto* worker = static_cast<Search::YaneuraOuWorker*>(
      engine.get_threads().main_thread()->diagnostic_worker());
  std::uint64_t written = 0, decode_errors = 0, illegal_positions = 0;
  std::uint64_t total_qnodes = 0;
  const auto started = std::chrono::steady_clock::now();
  auto last_report = started;
  MoveAccuracyRecord record{};
  while (written < requested && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos;
    StateInfo state;
    if (pos.set_from_packed_sfen(record.sfen, &state, false, record.game_ply).is_not_ok()) {
      ++decode_errors;
      break;
    }
    if (!pos.pos_is_ok()) {
      ++illegal_positions;
      break;
    }
    const bool in_check = bool(pos.checkers());
    const int legal_count = static_cast<int>(MoveList<LEGAL>(pos).size());
    const int material_stm = pos.state()->materialValue
      * (pos.side_to_move() == BLACK ? 1 : -1);
    const Value static_eval = ::YaneuraOu::Eval::evaluate(pos);
    const auto signal = LastNnueSignalAccess().signal;
    std::uint64_t qnodes = 0;
    int qsel = 0;
    const Value qsearch_eval = worker->diagnostic_qsearch(pos, qnodes, qsel);
    total_qnodes += qnodes;
    const int delta = static_cast<int>(qsearch_eval) - static_cast<int>(static_eval);
    const bool terminal_or_mate = legal_count == 0 || is_decisive(qsearch_eval)
      || std::abs(static_cast<int>(qsearch_eval)) >= VALUE_MAX_EVAL;
    csv << written << ',' << (start_record + written) << ',' << static_eval << ','
        << qsearch_eval << ',' << delta << ',' << std::abs(delta) << ','
        << static_cast<int>(pos.side_to_move()) << ',' << record.game_ply << ','
        << material_stm << ',' << signal.selected_bucket << ',' << (in_check ? 1 : 0)
        << ',' << legal_count << ',' << qnodes << ',' << qsel << ','
        << (terminal_or_mate ? 1 : 0) << ',' << signal.router_margin << ','
        << signal.lca_abs_delta_sum << ',' << signal.cross_abs_sum << ','
        << static_cast<int>(signal.cross_abs_max) << ',' << signal.main_gate_sum << ','
        << signal.fm_diff_activity_sum << ',' << signal.fm_abs_activity_sum << ','
        << signal.main_reliance << ',' << signal.fm_reliance << ','
        << signal.cross_reliance << ',' << signal.deep_bypass_disagreement << '\n';
    corpus.write(reinterpret_cast<const char*>(&record), sizeof(record));
    ++written;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5)) {
      last_report = now;
      const double sec = std::chrono::duration<double>(now - started).count();
      std::cout << "qsearch_correction progress=" << written << '/' << requested
                << " positions_per_sec=" << std::fixed << std::setprecision(1)
                << (written / std::max(sec, 1e-9)) << " mean_qnodes="
                << (written ? double(total_qnodes) / written : 0.0) << std::endl;
    }
  }
  csv.flush(); corpus.flush(); csv.close(); corpus.close();
  if (written != requested || decode_errors || illegal_positions) {
    std::remove(csv_tmp.c_str()); std::remove(corpus_tmp.c_str());
    std::cout << "qsearch_correction status=error written=" << written
              << " decode_errors=" << decode_errors
              << " illegal_positions=" << illegal_positions << std::endl;
    return;
  }
  if (std::rename(csv_tmp.c_str(), output_csv_name.c_str()) != 0
      || std::rename(corpus_tmp.c_str(), output_corpus_name.c_str()) != 0) {
    std::cout << "qsearch_correction status=error rename_failed" << std::endl;
    return;
  }
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "qsearch_correction status=ok written=" << written
            << " total_qnodes=" << total_qnodes << " elapsed_sec=" << sec
            << " positions_per_sec=" << (written / std::max(sec, 1e-9)) << std::endl;
}
