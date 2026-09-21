#if defined(ENABLE_NNUE_MOBILITY_TACTICAL_PROTOTYPE)

struct MobilityTacticalSeries {
  std::vector<double> values;

  double mean() const {
    return values.empty() ? 0.0
        : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  }
  double stddev() const {
    if (values.empty()) return 0.0;
    const double m = mean();
    double sum = 0.0;
    for (const double value : values) sum += (value - m) * (value - m);
    return std::sqrt(sum / values.size());
  }
  double quantile(const double q) const {
    if (values.empty()) return 0.0;
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double position = q * (sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - lower;
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
  }
  double maximum() const {
    return values.empty() ? 0.0
        : *std::max_element(values.begin(), values.end());
  }
  double zero_ratio() const {
    if (values.empty()) return 0.0;
    return double(std::count(values.begin(), values.end(), 0.0)) / values.size();
  }
};

double MobilityTacticalPearson(const std::vector<double>& first,
                               const std::vector<double>& second) {
  if (first.size() != second.size() || first.size() < 2) return 0.0;
  const double first_mean =
      std::accumulate(first.begin(), first.end(), 0.0) / first.size();
  const double second_mean =
      std::accumulate(second.begin(), second.end(), 0.0) / second.size();
  double covariance = 0.0;
  double first_variance = 0.0;
  double second_variance = 0.0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const double x = first[i] - first_mean;
    const double y = second[i] - second_mean;
    covariance += x * y;
    first_variance += x * x;
    second_variance += y * y;
  }
  return first_variance == 0.0 || second_variance == 0.0 ? 0.0
      : covariance / std::sqrt(first_variance * second_variance);
}

constexpr std::array<const char*, NnueMobilityTactical::FeatureCount>
    MobilityTacticalNames = {{
        "us_rook_mobility", "them_rook_mobility",
        "us_bishop_mobility", "them_bishop_mobility",
        "us_capture_count", "them_capture_count",
        "us_max_capturable_bucket", "them_max_capturable_bucket"}};

std::array<float, NnueMobilityTactical::FeatureCount>
NormalizeMobilityTactical(const NnueMobilityTactical::RawFeatures& raw) {
  constexpr std::array<float, NnueMobilityTactical::FeatureCount> scale =
      {{20.0f, 20.0f, 16.0f, 16.0f, 8.0f, 8.0f, 6.0f, 6.0f}};
  std::array<float, NnueMobilityTactical::FeatureCount> normalized{};
  for (std::size_t i = 0; i < normalized.size(); ++i)
    normalized[i] = std::min(float(raw.value[i]) / scale[i], 1.0f);
  return normalized;
}

struct MobilityTacticalFusionWeights {
  std::array<float, 16 * 8> a_encode{};
  std::array<float, 16> a_bias{};
  std::array<float, 128 * 16> a_project{};
  std::array<float, 128> a_project_bias{};
  std::array<float, 128 * 8> b_project{};
  std::array<float, 128> b_bias{};
  std::array<float, 8> c_diagonal{};

  MobilityTacticalFusionWeights() {
    const auto fill = [](auto& values, const int salt) {
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = float((int((i * 37 + salt) % 29) - 14)) / 1024.0f;
    };
    fill(a_encode, 1); fill(a_bias, 3); fill(a_project, 5);
    fill(a_project_bias, 7); fill(b_project, 9); fill(b_bias, 11);
    fill(c_diagonal, 13);
  }
};

inline float MobilityTacticalSilu(const float value) {
  return value / (1.0f + std::exp(-value));
}

void MobilityTacticalFusionA(
    const std::array<float, 8>& input,
    const MobilityTacticalFusionWeights& weights,
    std::array<float, 128>& residual) {
  std::array<float, 16> hidden{};
  for (std::size_t row = 0; row < hidden.size(); ++row) {
    float value = weights.a_bias[row];
    for (std::size_t column = 0; column < input.size(); ++column)
      value += weights.a_encode[row * input.size() + column] * input[column];
    hidden[row] = MobilityTacticalSilu(value);
  }
  residual = weights.a_project_bias;
#if defined(USE_AVX2)
  for (std::size_t column = 0; column < hidden.size(); ++column) {
    const __m256 input = _mm256_set1_ps(hidden[column]);
    for (std::size_t row = 0; row < residual.size(); row += 8) {
      const __m256 value = _mm256_loadu_ps(residual.data() + row);
      const __m256 weight = _mm256_loadu_ps(
          weights.a_project.data() + column * residual.size() + row);
      _mm256_storeu_ps(residual.data() + row,
                       _mm256_add_ps(value, _mm256_mul_ps(input, weight)));
    }
  }
#else
  for (std::size_t column = 0; column < hidden.size(); ++column)
    for (std::size_t row = 0; row < residual.size(); ++row)
      residual[row] += weights.a_project[column * residual.size() + row]
                     * hidden[column];
#endif
}

void MobilityTacticalFusionB(
    const std::array<float, 8>& input,
    const MobilityTacticalFusionWeights& weights,
    std::array<float, 128>& residual) {
  residual = weights.b_bias;
#if defined(USE_AVX2)
  for (std::size_t column = 0; column < input.size(); ++column) {
    const __m256 value = _mm256_set1_ps(input[column]);
    for (std::size_t row = 0; row < residual.size(); row += 8) {
      const __m256 old = _mm256_loadu_ps(residual.data() + row);
      const __m256 weight = _mm256_loadu_ps(
          weights.b_project.data() + column * residual.size() + row);
      _mm256_storeu_ps(residual.data() + row,
                       _mm256_add_ps(old, _mm256_mul_ps(value, weight)));
    }
  }
#else
  for (std::size_t column = 0; column < input.size(); ++column)
    for (std::size_t row = 0; row < residual.size(); ++row)
      residual[row] += weights.b_project[column * residual.size() + row]
                     * input[column];
#endif
}

struct MobilityTacticalSubsetSpec {
  const char* name;
  std::array<std::uint8_t, 8> columns;
  std::size_t count;
};

constexpr std::array<MobilityTacticalSubsetSpec, 7>
    MobilityTacticalSubsetSpecs = {{
        {"M-R", {{0, 1, 0, 0, 0, 0, 0, 0}}, 2},
        {"M-B", {{2, 3, 0, 0, 0, 0, 0, 0}}, 2},
        {"T-C", {{4, 5, 0, 0, 0, 0, 0, 0}}, 2},
        {"T-V", {{6, 7, 0, 0, 0, 0, 0, 0}}, 2},
        {"M", {{0, 1, 2, 3, 0, 0, 0, 0}}, 4},
        {"T", {{4, 5, 6, 7, 0, 0, 0, 0}}, 4},
        {"ALL", {{0, 1, 2, 3, 4, 5, 6, 7}}, 8},
    }};

NnueMobilityTactical::RawFeatures MobilityTacticalExtractSubset(
    const Position& pos, const std::size_t subset) {
  using namespace NnueMobilityTactical;
  NnueMobilityTactical::RawFeatures result{};
  const Color us = pos.side_to_move();
  const Color them = ~us;
  if (subset == 0 || subset == 4 || subset == 6) {
    result.value[UsRookMobility] = rook_mobility(pos, us, true);
    result.value[ThemRookMobility] = rook_mobility(pos, them, true);
  }
  if (subset == 1 || subset == 4 || subset == 6) {
    result.value[UsBishopMobility] = bishop_mobility(pos, us, true);
    result.value[ThemBishopMobility] = bishop_mobility(pos, them, true);
  }
  if (subset == 2 || subset == 3 || subset == 5 || subset == 6) {
    const auto us_capture = capture_availability(pos, us);
    const auto them_capture = capture_availability(pos, them);
    result.value[UsCaptureCount] = us_capture.unique_targets;
    result.value[ThemCaptureCount] = them_capture.unique_targets;
    result.value[UsMaxCapturableBucket] = us_capture.max_bucket;
    result.value[ThemMaxCapturableBucket] = them_capture.max_bucket;
  }
  return result;
}

std::array<float, 8> MobilityTacticalNormalizeSubset(
    const NnueMobilityTactical::RawFeatures& raw, const std::size_t subset) {
  constexpr std::array<float, 8> scale =
      {{20.0f, 20.0f, 16.0f, 16.0f, 8.0f, 8.0f, 6.0f, 6.0f}};
  std::array<float, 8> normalized{};
  const auto& spec = MobilityTacticalSubsetSpecs[subset];
  for (std::size_t i = 0; i < spec.count; ++i) {
    const auto column = spec.columns[i];
    normalized[column] =
        std::min(float(raw.value[column]) / scale[column], 1.0f);
  }
  return normalized;
}

void MobilityTacticalFusionBSubset(
    const std::array<float, 8>& input,
    const MobilityTacticalFusionWeights& weights,
    const std::size_t subset,
    std::array<float, 128>& residual) {
  residual = weights.b_bias;
  const auto& spec = MobilityTacticalSubsetSpecs[subset];
#if defined(USE_AVX2)
  for (std::size_t i = 0; i < spec.count; ++i) {
    const auto column = spec.columns[i];
    const __m256 value = _mm256_set1_ps(input[column]);
    for (std::size_t row = 0; row < residual.size(); row += 8) {
      const __m256 old = _mm256_loadu_ps(residual.data() + row);
      const __m256 weight = _mm256_loadu_ps(
          weights.b_project.data() + column * residual.size() + row);
      _mm256_storeu_ps(residual.data() + row,
                       _mm256_add_ps(old, _mm256_mul_ps(value, weight)));
    }
  }
#else
  for (std::size_t i = 0; i < spec.count; ++i) {
    const auto column = spec.columns[i];
    for (std::size_t row = 0; row < residual.size(); ++row)
      residual[row] += weights.b_project[column * residual.size() + row]
                     * input[column];
  }
#endif
}

void MobilityTacticalFusionC(
    const std::array<float, 8>& input,
    const MobilityTacticalFusionWeights& weights,
    std::array<float, 128>& residual) {
  residual.fill(0.0f);
  for (std::size_t i = 0; i < input.size(); ++i)
    residual[i] = weights.c_diagonal[i] * input[i];
}

bool MobilityTacticalSet(Position& pos, StateInfo& state,
                         const std::string& sfen) {
  pos.set(sfen, &state);
  return true;
}

void TestMobilityTacticalSelftest() {
  using namespace NnueMobilityTactical;
  std::uint64_t passed = 0;
  std::uint64_t failed = 0;
  const auto expect = [&](const char* name, const bool condition,
                          const long long actual, const long long wanted) {
    std::cout << "mobility_tactical_selftest " << name << '='
              << (condition ? "PASS" : "FAIL") << " actual=" << actual
              << " expected=" << wanted << std::endl;
    condition ? ++passed : ++failed;
  };

  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "4k4/9/9/4P4/3PRP3/4P4/9/9/4K4 b - 1");
    expect("blocked_rook", rook_mobility(pos, BLACK, true) == 0,
           rook_mobility(pos, BLACK, true), 0);
  }
  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "4k4/9/9/9/4R4/9/9/9/4K4 b - 1");
    expect("open_rook", rook_mobility(pos, BLACK, true) == 15,
           rook_mobility(pos, BLACK, true), 15);
  }
  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "8k/9/9/3P1P3/4B4/3P1P3/9/9/K8 b - 1");
    expect("blocked_bishop", bishop_mobility(pos, BLACK, true) == 0,
           bishop_mobility(pos, BLACK, true), 0);
  }
  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "8k/9/9/9/4B4/9/9/9/K8 b - 1");
    // The own king on 9i blocks and is excluded from one diagonal endpoint.
    expect("open_bishop", bishop_mobility(pos, BLACK, true) == 15,
           bishop_mobility(pos, BLACK, true), 15);
  }
  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "k5B2/9/4r4/9/4R4/9/9/9/K8 b - 1");
    const auto capture = capture_availability(pos, BLACK, true);
    expect("unique_target", capture.unique_targets == 1,
           capture.unique_targets, 1);
    expect("multiple_attackers_one_target", capture.relations == 2,
           capture.relations, 2);
    expect("capturable_rook_bucket", capture.max_bucket == 5,
           capture.max_bucket, 5);
  }
  {
    Position pos; StateInfo state;
    MobilityTacticalSet(pos, state,
        "k8/9/4+r4/9/4R4/9/9/9/K8 b - 1");
    const auto capture = capture_availability(pos, BLACK);
    expect("capturable_dragon_bucket", capture.max_bucket == 6,
           capture.max_bucket, 6);
  }
  {
    Position black; Position white; StateInfo black_state; StateInfo white_state;
    const std::string board = "k5B2/9/4r4/9/4R4/9/9/9/K8";
    MobilityTacticalSet(black, black_state, board + " b - 1");
    MobilityTacticalSet(white, white_state, board + " w - 1");
    const auto b = extract(black).value;
    const auto w = extract(white).value;
    bool swapped = true;
    for (std::size_t i = 0; i < FeatureCount; i += 2)
      swapped = swapped && b[i] == w[i + 1] && b[i + 1] == w[i];
    expect("side_to_move_perspective", swapped, swapped, 1);
  }
  {
    Position dragon; Position horse; StateInfo dragon_state; StateInfo horse_state;
    MobilityTacticalSet(dragon, dragon_state,
        "4k4/9/9/9/4+R4/9/9/9/4K4 b - 1");
    MobilityTacticalSet(horse, horse_state,
        "8k/9/9/9/4+B4/9/9/9/K8 b - 1");
    expect("dragon_long_component",
           rook_mobility(dragon, BLACK, true) == 15
               && rook_mobility(dragon, BLACK, false) == 0,
           rook_mobility(dragon, BLACK, true), 15);
    expect("horse_long_component",
           bishop_mobility(horse, BLACK, true) == 15
               && bishop_mobility(horse, BLACK, false) == 0,
           bishop_mobility(horse, BLACK, true), 15);
  }
  std::cout << "mobility_tactical_selftest summary passed=" << passed
            << " failed=" << failed << std::endl;
}

void TestMobilityTacticalCorpus(std::istream& stream) {
  using namespace NnueMobilityTactical;
  std::string input_name;
  std::string distribution_name;
  std::string correlation_name;
  std::uint64_t record_limit = 100000;
  std::uint64_t cost_positions = 10000;
  std::uint64_t cost_repeats = 32;
  stream >> std::quoted(input_name) >> record_limit
         >> std::quoted(distribution_name) >> std::quoted(correlation_name)
         >> cost_positions >> cost_repeats;
  if (input_name.empty() || distribution_name.empty()
      || correlation_name.empty()) {
    std::cout << "usage: test nnue mobility_tactical_corpus <packed.bin>"
              << " <records> <distribution.csv> <correlation.csv>"
              << " [cost_positions] [cost_repeats]" << std::endl;
    return;
  }

  std::ifstream input(input_name, std::ios::binary);
  std::ofstream distribution(distribution_name);
  std::ofstream correlations(correlation_name);
  if (!input || !distribution || !correlations) {
    std::cout << "mobility_tactical_corpus file open failure" << std::endl;
    return;
  }

  std::array<MobilityTacticalSeries, FeatureCount> series;
  MobilityTacticalSeries us_rook_plain, them_rook_plain;
  MobilityTacticalSeries us_bishop_plain, them_bishop_plain;
  MobilityTacticalSeries us_capture_relations, them_capture_relations;
  std::array<long double, 9> cost_ns{};
  std::uint64_t cost_samples = 0;
  volatile std::uint64_t checksum = 0;
  std::uint64_t decoded = 0;
  std::uint64_t decode_failed = 0;
  MoveAccuracyRecord record{};
  while (decoded < record_limit
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos; StateInfo state;
    if (pos.set_from_packed_sfen(
            record.sfen, &state, false, record.game_ply).is_not_ok()) {
      ++decode_failed;
      continue;
    }
    const auto features = extract(pos);
    for (std::size_t i = 0; i < FeatureCount; ++i)
      series[i].values.push_back(features.value[i]);
    const Color us = pos.side_to_move();
    const Color them = ~us;
    us_rook_plain.values.push_back(rook_mobility(pos, us, false));
    them_rook_plain.values.push_back(rook_mobility(pos, them, false));
    us_bishop_plain.values.push_back(bishop_mobility(pos, us, false));
    them_bishop_plain.values.push_back(bishop_mobility(pos, them, false));
    const auto us_rel = capture_availability(pos, us, true);
    const auto them_rel = capture_availability(pos, them, true);
    us_capture_relations.values.push_back(us_rel.relations);
    them_capture_relations.values.push_back(them_rel.relations);

    if (cost_samples < cost_positions && cost_repeats != 0) {
      const auto measure = [&](const int slot, const auto& callback) {
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint64_t repeat = 0; repeat < cost_repeats; ++repeat)
          checksum += callback();
        const auto end = std::chrono::steady_clock::now();
        cost_ns[slot] += std::chrono::duration<long double, std::nano>(
            end - begin).count();
      };
      measure(0, [&] {
        return std::uint64_t(rook_mobility(pos, us, true))
             + rook_mobility(pos, them, true);
      });
      measure(1, [&] {
        return std::uint64_t(bishop_mobility(pos, us, true))
             + bishop_mobility(pos, them, true);
      });
      measure(2, [&] {
        const auto a = capture_availability(pos, us);
        const auto b = capture_availability(pos, them);
        return std::uint64_t(a.unique_targets + b.unique_targets);
      });
      measure(3, [&] {
        const auto a = capture_availability(pos, us);
        const auto b = capture_availability(pos, them);
        return std::uint64_t(a.max_bucket + b.max_bucket);
      });
      measure(4, [&] {
        const auto all = extract(pos);
        return std::accumulate(all.value.begin(), all.value.end(),
                               std::uint64_t(0));
      });
      measure(5, [&] {
        const auto a = capture_availability(pos, us, true);
        const auto b = capture_availability(pos, them, true);
        return std::uint64_t(a.relations + b.relations);
      });
      measure(6, [&] {
        return std::uint64_t(rook_mobility(pos, us, false))
             + rook_mobility(pos, them, false);
      });
      measure(7, [&] {
        return std::uint64_t(bishop_mobility(pos, us, false))
             + bishop_mobility(pos, them, false);
      });
      measure(8, [&] {
        const auto both = capture_pair(pos);
        return std::uint64_t(both[us].unique_targets
                           + both[them].unique_targets
                           + both[us].max_bucket + both[them].max_bucket);
      });
      ++cost_samples;
    }
    ++decoded;
  }

  distribution << "feature,count,mean,std,median,p90,p95,p99,max,zero_ratio\n";
  const auto write_series = [&](const char* name,
                                const MobilityTacticalSeries& values) {
    distribution << name << ',' << values.values.size() << ','
                 << values.mean() << ',' << values.stddev() << ','
                 << values.quantile(0.50) << ',' << values.quantile(0.90)
                 << ',' << values.quantile(0.95) << ','
                 << values.quantile(0.99) << ',' << values.maximum() << ','
                 << values.zero_ratio() << '\n';
  };
  for (std::size_t i = 0; i < FeatureCount; ++i)
    write_series(MobilityTacticalNames[i], series[i]);
  write_series("us_rook_mobility_rook_only", us_rook_plain);
  write_series("them_rook_mobility_rook_only", them_rook_plain);
  write_series("us_bishop_mobility_bishop_only", us_bishop_plain);
  write_series("them_bishop_mobility_bishop_only", them_bishop_plain);
  write_series("us_capture_relation_count", us_capture_relations);
  write_series("them_capture_relation_count", them_capture_relations);

  correlations << "first,second,pearson\n";
  const auto write_correlation = [&](const char* first_name,
                                     const MobilityTacticalSeries& first,
                                     const char* second_name,
                                     const MobilityTacticalSeries& second) {
    correlations << first_name << ',' << second_name << ','
                 << MobilityTacticalPearson(first.values, second.values)
                 << '\n';
  };
  write_correlation(MobilityTacticalNames[UsRookMobility],
                    series[UsRookMobility],
                    MobilityTacticalNames[ThemRookMobility],
                    series[ThemRookMobility]);
  write_correlation(MobilityTacticalNames[UsBishopMobility],
                    series[UsBishopMobility],
                    MobilityTacticalNames[ThemBishopMobility],
                    series[ThemBishopMobility]);
  write_correlation(MobilityTacticalNames[UsCaptureCount],
                    series[UsCaptureCount],
                    MobilityTacticalNames[ThemCaptureCount],
                    series[ThemCaptureCount]);
  write_correlation(MobilityTacticalNames[UsMaxCapturableBucket],
                    series[UsMaxCapturableBucket],
                    MobilityTacticalNames[ThemMaxCapturableBucket],
                    series[ThemMaxCapturableBucket]);
  write_correlation(MobilityTacticalNames[UsRookMobility],
                    series[UsRookMobility],
                    MobilityTacticalNames[UsBishopMobility],
                    series[UsBishopMobility]);
  write_correlation(MobilityTacticalNames[ThemRookMobility],
                    series[ThemRookMobility],
                    MobilityTacticalNames[ThemBishopMobility],
                    series[ThemBishopMobility]);
  write_correlation(MobilityTacticalNames[UsCaptureCount],
                    series[UsCaptureCount],
                    MobilityTacticalNames[UsMaxCapturableBucket],
                    series[UsMaxCapturableBucket]);
  write_correlation(MobilityTacticalNames[ThemCaptureCount],
                    series[ThemCaptureCount],
                    MobilityTacticalNames[ThemMaxCapturableBucket],
                    series[ThemMaxCapturableBucket]);

  const long double denominator =
      static_cast<long double>(cost_samples) * cost_repeats;
  std::cout << std::fixed << std::setprecision(3)
            << "mobility_tactical_corpus decoded=" << decoded
            << " decode_failed=" << decode_failed
            << " cost_positions=" << cost_samples
            << " repeats=" << cost_repeats << std::endl;
  if (denominator != 0.0L)
    std::cout << "mobility_tactical_cost_ns/eval rook_pair="
              << double(cost_ns[0] / denominator)
              << " bishop_pair=" << double(cost_ns[1] / denominator)
              << " capture_unique=" << double(cost_ns[2] / denominator)
              << " max_bucket_same_scan=" << double(cost_ns[3] / denominator)
              << " all_8_fused=" << double(cost_ns[4] / denominator)
              << " capture_relation_count=" << double(cost_ns[5] / denominator)
              << " rook_pair_plain_only=" << double(cost_ns[6] / denominator)
              << " bishop_pair_plain_only=" << double(cost_ns[7] / denominator)
              << " capture_all_piece_one_pass=" << double(cost_ns[8] / denominator)
              << " checksum=" << checksum << std::endl;
  std::cout << "mobility_tactical_outputs distribution=" << distribution_name
            << " correlations=" << correlation_name << std::endl;
}

void TestMobilityTacticalFusionCost(std::istream& stream) {
  std::string input_name;
  std::uint64_t position_limit = 10000;
  std::uint64_t repeats = 256;
  stream >> std::quoted(input_name) >> position_limit >> repeats;
  std::ifstream input(input_name, std::ios::binary);
  if (!input || position_limit == 0 || repeats == 0) {
    std::cout << "usage: test nnue mobility_tactical_fusion_cost"
              << " <packed.bin> [positions] [repeats]" << std::endl;
    return;
  }
  MobilityTacticalFusionWeights weights;
  MobilityTacticalFusionWeights neutral_weights = weights;
  neutral_weights.a_project.fill(0.0f);
  neutral_weights.a_project_bias.fill(0.0f);
  neutral_weights.b_project.fill(0.0f);
  neutral_weights.b_bias.fill(0.0f);
  neutral_weights.c_diagonal.fill(0.0f);
  std::array<long double, 7> elapsed{};
  std::array<std::array<long double, 3>, 7> subset_elapsed{};
  std::uint64_t positions = 0;
  volatile float checksum = 0.0f;
  float neutral_max_abs = 0.0f;
  MoveAccuracyRecord record{};
  while (positions < position_limit
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos; StateInfo state;
    if (pos.set_from_packed_sfen(
            record.sfen, &state, false, record.game_ply).is_not_ok())
      continue;
    const auto raw = NnueMobilityTactical::extract(pos);
    const auto normalized = NormalizeMobilityTactical(raw);
    std::array<float, 128> residual{};
    std::array<float, 128> base{};
    for (std::size_t i = 0; i < base.size(); ++i)
      base[i] = float((i * 17 + positions) % 101) / 101.0f;
    const auto measure = [&](const int slot, const auto& callback) {
      const auto begin = std::chrono::steady_clock::now();
      for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(&pos) : "memory");
#endif
        checksum += callback();
      }
      const auto end = std::chrono::steady_clock::now();
      elapsed[slot] += std::chrono::duration<long double, std::nano>(
          end - begin).count();
    };
    measure(0, [&] {
      const auto value = NnueMobilityTactical::extract(pos);
      return float(value.value[positions % value.value.size()]);
    });
    measure(1, [&] {
      const auto value = NormalizeMobilityTactical(raw);
      return value[positions % value.size()];
    });
    measure(2, [&] {
      MobilityTacticalFusionA(normalized, weights, residual);
      return residual[positions % residual.size()];
    });
    measure(3, [&] {
      MobilityTacticalFusionB(normalized, weights, residual);
      return residual[positions % residual.size()];
    });
    measure(4, [&] {
      MobilityTacticalFusionC(normalized, weights, residual);
      return residual[positions % 8];
    });
    for (std::size_t subset = 0; subset < MobilityTacticalSubsetSpecs.size();
         ++subset) {
      const auto subset_raw = MobilityTacticalExtractSubset(pos, subset);
      const auto subset_normalized =
          MobilityTacticalNormalizeSubset(subset_raw, subset);
      const auto subset_measure = [&](const std::size_t part,
                                      const auto& callback) {
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
#if defined(__GNUC__) || defined(__clang__)
          asm volatile("" : : "r"(&pos) : "memory");
#endif
          checksum += callback();
        }
        const auto end = std::chrono::steady_clock::now();
        subset_elapsed[subset][part] +=
            std::chrono::duration<long double, std::nano>(end - begin).count();
      };
      subset_measure(0, [&] {
        const auto value = MobilityTacticalExtractSubset(pos, subset);
        return float(value.value[
            MobilityTacticalSubsetSpecs[subset].columns[
                positions % MobilityTacticalSubsetSpecs[subset].count]]);
      });
      subset_measure(1, [&] {
        const auto value = MobilityTacticalNormalizeSubset(subset_raw, subset);
        return value[MobilityTacticalSubsetSpecs[subset].columns[
            positions % MobilityTacticalSubsetSpecs[subset].count]];
      });
      subset_measure(2, [&] {
        MobilityTacticalFusionBSubset(
            subset_normalized, weights, subset, residual);
        return residual[positions % residual.size()];
      });
    }
    MobilityTacticalFusionA(normalized, neutral_weights, residual);
    for (const float value : residual)
      neutral_max_abs = std::max(neutral_max_abs, std::abs(value));
    MobilityTacticalFusionB(normalized, neutral_weights, residual);
    for (const float value : residual)
      neutral_max_abs = std::max(neutral_max_abs, std::abs(value));
    MobilityTacticalFusionC(normalized, neutral_weights, residual);
    for (const float value : residual)
      neutral_max_abs = std::max(neutral_max_abs, std::abs(value));
    MobilityTacticalFusionA(normalized, weights, residual);
    measure(5, [&] {
      float sum = 0.0f;
      for (std::size_t i = 0; i < base.size(); ++i)
        sum += base[i] + residual[i];
      return sum;
    });
    // Subtract the base-only loop to isolate the residual-add cost.
    measure(6, [&] {
      float sum = 0.0f;
      for (const float value : base) sum += value;
      return sum;
    });
    ++positions;
  }
  const long double denominator = static_cast<long double>(positions) * repeats;
  const double extraction = double(elapsed[0] / denominator);
  const double normalization = double(elapsed[1] / denominator);
  const double add = double((elapsed[5] - elapsed[6]) / denominator);
  const double a_projection = double(elapsed[2] / denominator);
  const double b_projection = double(elapsed[3] / denominator);
  const double c_projection = double(elapsed[4] / denominator);
  std::cout << std::fixed << std::setprecision(3)
            << "mobility_tactical_fusion_cost positions=" << positions
            << " repeats=" << repeats << std::endl
            << "feature_extraction_ns=" << extraction
            << " normalization_ns=" << normalization
            << " residual_add_128_ns=" << add << std::endl
            << "fusion_A_projection_ns=" << a_projection
            << " total_ns=" << extraction + normalization + a_projection + add
            << std::endl
            << "fusion_B_projection_ns=" << b_projection
            << " total_ns=" << extraction + normalization + b_projection + add
            << std::endl
            << "fusion_C_projection_ns=" << c_projection
            << " total_ns=" << extraction + normalization + c_projection + add
            << " checksum=" << checksum << std::endl
            << "baseline_neutral_max_abs=" << neutral_max_abs << std::endl;
  for (std::size_t subset = 0; subset < MobilityTacticalSubsetSpecs.size();
       ++subset) {
    const double subset_extraction =
        double(subset_elapsed[subset][0] / denominator);
    const double subset_normalization =
        double(subset_elapsed[subset][1] / denominator);
    const double subset_projection =
        double(subset_elapsed[subset][2] / denominator);
    std::cout << "mobility_tactical_subset_cost subset="
              << MobilityTacticalSubsetSpecs[subset].name
              << " inputs=" << MobilityTacticalSubsetSpecs[subset].count
              << " extraction_ns=" << subset_extraction
              << " normalization_ns=" << subset_normalization
              << " projection_ns=" << subset_projection
              << " residual_add_ns=" << add
              << " total_ns=" << subset_extraction + subset_normalization
                                    + subset_projection + add
              << std::endl;
  }
}

void ExportMobilityTacticalFeatures(std::istream& stream) {
  std::string input_name;
  std::string output_name;
  std::uint64_t record_limit = 1000000;
  stream >> std::quoted(input_name) >> std::quoted(output_name) >> record_limit;
  std::ifstream input(input_name, std::ios::binary);
  std::ofstream output(output_name, std::ios::binary);
  if (!input || !output) {
    std::cout << "usage: test nnue mobility_tactical_export <packed.bin>"
              << " <features.f32> [records]" << std::endl;
    return;
  }
  MoveAccuracyRecord record{};
  std::uint64_t written = 0;
  std::uint64_t failed = 0;
  while (written < record_limit
         && input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    Position pos; StateInfo state;
    std::array<float, 8> normalized{};
    if (pos.set_from_packed_sfen(
            record.sfen, &state, false, record.game_ply).is_not_ok())
      ++failed;
    else
      normalized = NormalizeMobilityTactical(
          NnueMobilityTactical::extract(pos));
    output.write(reinterpret_cast<const char*>(normalized.data()),
                 sizeof(normalized));
    ++written;
  }
  std::cout << "mobility_tactical_export written=" << written
            << " decode_failed=" << failed << " output=" << output_name
            << std::endl;
}

#endif  // ENABLE_NNUE_MOBILITY_TACTICAL_PROTOTYPE
