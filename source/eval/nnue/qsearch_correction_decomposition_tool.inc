// Experiment 64 diagnostic-only decomposition corpus generator.
void MakeQsearchCorrectionDecompositionCorpus(IEngine& engine, std::istream& stream) {
  std::string input_name, output_csv_name, output_corpus_name;
  std::string output_native_l2_name;
  std::uint64_t start_record = 0, max_records = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_csv_name)
         >> std::quoted(output_corpus_name) >> start_record >> max_records;
  if (!(stream >> std::quoted(output_native_l2_name))) {
    stream.clear();
    output_native_l2_name.clear();
  }
  if (input_name.empty() || output_csv_name.empty() || output_corpus_name.empty()
      || max_records == 0) {
    std::cout << "error: qsearch_correction_decomposition requires input, csv, corpus, start, count"
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
  const std::string native_l2_tmp = output_native_l2_name.empty()
    ? std::string{} : output_native_l2_name + ".tmp";
  if (std::ifstream(output_csv_name) || std::ifstream(output_corpus_name)
      || std::ifstream(csv_tmp) || std::ifstream(corpus_tmp)
      || (!output_native_l2_name.empty()
          && (std::ifstream(output_native_l2_name) || std::ifstream(native_l2_tmp)))) {
    std::cout << "error: output or temporary output already exists" << std::endl;
    return;
  }
  std::ofstream csv(csv_tmp, std::ios::trunc);
  std::ofstream corpus(corpus_tmp, std::ios::binary | std::ios::trunc);
  std::ofstream native_l2;
  if (!output_native_l2_name.empty())
    native_l2.open(native_l2_tmp, std::ios::binary | std::ios::trunc);
  if (!csv || !corpus || (!output_native_l2_name.empty() && !native_l2)) {
    std::cout << "error: failed to create output" << std::endl;
    return;
  }
  csv << "row,source_record,raw_static,qsearch_raw_static,corrected_static,qsearch_final,"
         "history_delta,tactical_delta,total_delta,raw_static_match,side_to_move,game_ply,"
         "material_stm,bucket,in_check,legal_move_count,qsearch_nodes,qsearch_sel_depth,"
         "qsearch_elapsed_ns,stand_pat_immediate,tactical_continuation,"
         "stand_pat_observed,terminal_or_mate,router_margin,lca_abs_delta_sum,cross_abs_sum,"
         "cross_abs_max,main_gate_sum,fm_diff_activity_sum,fm_abs_activity_sum,main_reliance,"
         "fm_reliance,cross_reliance,deep_bypass_disagreement";
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
  csv << ",shadow_valid,linear_ge100,linear_ge200,mlp32_ge100,mlp32_ge200,"
         "linear_q8_ge100,linear_q8_ge200,mlp32_q8_ge100,mlp32_q8_ge200\n";
#else
  csv << '\n';
#endif

  auto* worker = static_cast<Search::YaneuraOuWorker*>(
      engine.get_threads().main_thread()->diagnostic_worker());
  std::uint64_t written = 0, decode_errors = 0, illegal_positions = 0;
  std::uint64_t observed = 0, raw_mismatch = 0, total_qnodes = 0;
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
  std::uint64_t shadow_input_ns = 0, shadow_linear_ns = 0, shadow_mlp32_ns = 0;
#endif
  const auto started = std::chrono::steady_clock::now();
  auto last_report = started;
  MoveAccuracyRecord record{};
  while (written < requested && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos;
    StateInfo state;
    if (pos.set_from_packed_sfen(record.sfen, &state, false, record.game_ply).is_not_ok()) {
      ++decode_errors; break;
    }
    if (!pos.pos_is_ok()) {
      ++illegal_positions; break;
    }
    const bool in_check = bool(pos.checkers());
    const int legal_count = static_cast<int>(MoveList<LEGAL>(pos).size());
    const int material_stm = pos.state()->materialValue
      * (pos.side_to_move() == BLACK ? 1 : -1);
    const Value raw_static = ::YaneuraOu::Eval::evaluate(pos);
    const auto signal = LastNnueSignalAccess().signal;
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
    const auto shadow_started = std::chrono::steady_clock::now();
    const auto shadow_input = QsearchCorrectionShadow::MakeInput(
      signal.qsearch_correction_l2_input, static_cast<int>(raw_static), record.game_ply,
      material_stm, in_check ? 1 : 0, signal.selected_bucket);
    const auto linear_started = std::chrono::steady_clock::now();
    shadow_input_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<
      std::chrono::nanoseconds>(std::chrono::steady_clock::now() - shadow_started).count());
    QsearchCorrectionShadow::Prediction shadow{};
    QsearchCorrectionShadow::PredictLinear(shadow_input, shadow);
    const auto mlp_started = std::chrono::steady_clock::now();
    shadow_linear_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<
      std::chrono::nanoseconds>(mlp_started - linear_started).count());
    QsearchCorrectionShadow::PredictMlp32(shadow_input, shadow);
    shadow_mlp32_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<
      std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mlp_started).count());
#endif
    std::uint64_t qnodes = 0;
    int qsel = 0;
    Value qsearch_raw = VALUE_NONE, corrected_static = VALUE_NONE;
    bool stand_pat_observed = false;
    const auto qsearch_started = std::chrono::steady_clock::now();
    const Value qsearch_final = worker->diagnostic_qsearch(
      pos, qnodes, qsel, &qsearch_raw, &corrected_static, &stand_pat_observed);
    const auto qsearch_elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<
      std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qsearch_started).count());
    total_qnodes += qnodes;
    if (stand_pat_observed) ++observed;
    const bool raw_match = stand_pat_observed && qsearch_raw == raw_static;
    if (stand_pat_observed && !raw_match) ++raw_mismatch;
    const int history_delta = stand_pat_observed
      ? static_cast<int>(corrected_static) - static_cast<int>(raw_static) : 0;
    const int tactical_delta = stand_pat_observed
      ? static_cast<int>(qsearch_final) - static_cast<int>(corrected_static) : 0;
    const int total_delta = static_cast<int>(qsearch_final) - static_cast<int>(raw_static);
    const bool terminal_or_mate = legal_count == 0 || is_decisive(qsearch_final)
      || std::abs(static_cast<int>(qsearch_final)) >= VALUE_MAX_EVAL;
    csv << written << ',' << (start_record + written) << ',' << raw_static << ','
        << qsearch_raw << ',' << corrected_static << ',' << qsearch_final << ','
        << history_delta << ',' << tactical_delta << ',' << total_delta << ','
        << (raw_match ? 1 : 0) << ',' << static_cast<int>(pos.side_to_move()) << ','
        << record.game_ply << ',' << material_stm << ',' << signal.selected_bucket << ','
        << (in_check ? 1 : 0) << ',' << legal_count << ',' << qnodes << ',' << qsel << ','
        << qsearch_elapsed_ns << ',' << (qnodes <= 1 ? 1 : 0) << ',' << (qnodes > 1 ? 1 : 0) << ','
        << (stand_pat_observed ? 1 : 0) << ',' << (terminal_or_mate ? 1 : 0) << ','
        << signal.router_margin << ',' << signal.lca_abs_delta_sum << ','
        << signal.cross_abs_sum << ',' << static_cast<int>(signal.cross_abs_max) << ','
        << signal.main_gate_sum << ',' << signal.fm_diff_activity_sum << ','
        << signal.fm_abs_activity_sum << ',' << signal.main_reliance << ','
        << signal.fm_reliance << ',' << signal.cross_reliance << ','
        << signal.deep_bypass_disagreement;
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
    csv << ',' << (signal.valid ? 1 : 0)
        << ',' << shadow.linear_probability[0] << ',' << shadow.linear_probability[1]
        << ',' << shadow.mlp32_probability[0] << ',' << shadow.mlp32_probability[1]
        << ',' << static_cast<int>(shadow.linear_q8[0])
        << ',' << static_cast<int>(shadow.linear_q8[1])
        << ',' << static_cast<int>(shadow.mlp32_q8[0])
        << ',' << static_cast<int>(shadow.mlp32_q8[1]);
#endif
    csv << '\n';
    corpus.write(reinterpret_cast<const char*>(&record), sizeof(record));
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
    if (native_l2)
      native_l2.write(reinterpret_cast<const char*>(signal.qsearch_correction_l2_input), 128);
#endif
    ++written;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5)) {
      last_report = now;
      const double sec = std::chrono::duration<double>(now - started).count();
      std::cout << "qsearch_decomposition progress=" << written << '/' << requested
                << " positions_per_sec=" << std::fixed << std::setprecision(1)
                << written / std::max(sec, 1e-9) << " stand_pat_observed=" << observed
                << " raw_mismatch=" << raw_mismatch << std::endl;
    }
  }
  csv.flush(); corpus.flush(); csv.close(); corpus.close();
  if (native_l2) { native_l2.flush(); native_l2.close(); }
  if (written != requested || decode_errors || illegal_positions) {
    std::remove(csv_tmp.c_str()); std::remove(corpus_tmp.c_str());
    if (!native_l2_tmp.empty()) std::remove(native_l2_tmp.c_str());
    std::cout << "qsearch_decomposition status=error written=" << written
              << " decode_errors=" << decode_errors
              << " illegal_positions=" << illegal_positions << std::endl;
    return;
  }
  if (std::rename(csv_tmp.c_str(), output_csv_name.c_str()) != 0
      || std::rename(corpus_tmp.c_str(), output_corpus_name.c_str()) != 0
      || (!output_native_l2_name.empty()
          && std::rename(native_l2_tmp.c_str(), output_native_l2_name.c_str()) != 0)) {
    std::cout << "qsearch_decomposition status=error rename_failed" << std::endl;
    return;
  }
  const double sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "qsearch_decomposition status=ok written=" << written
            << " stand_pat_observed=" << observed << " raw_mismatch=" << raw_mismatch
            << " total_qnodes=" << total_qnodes << " elapsed_sec=" << sec
            << " positions_per_sec=" << written / std::max(sec, 1e-9) << std::endl;
#if defined(ENABLE_QSEARCH_CORRECTION_SHADOW)
  const auto denom = std::max<std::uint64_t>(written, 1);
  std::cout << "qsearch_shadow input_ns_per_position=" << static_cast<double>(shadow_input_ns) / denom
            << " linear_ns_per_position=" << static_cast<double>(shadow_linear_ns) / denom
            << " mlp32_ns_per_position=" << static_cast<double>(shadow_mlp32_ns) / denom
            << " total_ns_per_position="
            << static_cast<double>(shadow_input_ns + shadow_linear_ns + shadow_mlp32_ns) / denom
            << std::endl;
#endif
}
