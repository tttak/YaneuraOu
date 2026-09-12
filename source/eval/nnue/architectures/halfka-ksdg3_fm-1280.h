#ifndef CLASSIC_NNUE_HALFKA_KSDG3_FM_1280_H_INCLUDED
#define CLASSIC_NNUE_HALFKA_KSDG3_FM_1280_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka.h"
#include "../features/king_safety3_distinguishgolds.h"
#include "../nnue_signal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"

namespace YaneuraOu {
namespace Eval::NNUE {

// Input features used in evaluation function
// 入力特徴量: HalfKA と KingSafety3 の組み合わせ
using RawFeatures = Features::FeatureSet<
	Features::HalfKA<Features::Side::kFriend>, Features::KingSafety3_DistinguishGolds<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
// 変換後の入力特徴量の次元数
constexpr IndexType kTransformedFeatureDimensions = 1280;

// Number of networks stored in the evaluation file
constexpr int LayerStacks = 12;

// 各層の次元数
constexpr IndexType kInputDims = kTransformedFeatureDimensions;
constexpr IndexType kHidden1Dims = 31;
#if defined(USE_NNUE_ABS_SQR_REMOVED_160) && !defined(USE_NNUE_LEGACY_PHASE6)
#define NNUE_COMPACT_PHASE5
#endif
#if defined(USE_NNUE_FC1_WIDTH_64) && !defined(NNUE_COMPACT_PHASE5)
#error USE_NNUE_FC1_WIDTH_64 requires the 160-input Phase5 architecture
#endif
#if defined(USE_NNUE_CROSS_WIDTH_24) \
	&& (!defined(NNUE_COMPACT_PHASE5) || !defined(USE_NNUE_FC1_WIDTH_64))
#error USE_NNUE_CROSS_WIDTH_24 requires the Phase5, L2x160, FC1x64 architecture
#endif

#if defined(USE_NNUE_CROSS_WIDTH_24)
constexpr IndexType CROSS_OUTPUT_SIZE = 24;
#else
constexpr IndexType CROSS_OUTPUT_SIZE = 32;
#endif

#if defined(USE_NNUE_ABS_SQR_REMOVED_160)
constexpr IndexType L2_INPUT_SIZE = 160;
constexpr IndexType L2_CROSS_OFFSET = 126;
constexpr IndexType L2_REAL_SIZE = L2_CROSS_OFFSET + CROSS_OUTPUT_SIZE;
constexpr IndexType L2_LOGICAL_SIZE = L2_REAL_SIZE + 2;
constexpr IndexType L2_PADDING_SIZE = L2_INPUT_SIZE - L2_REAL_SIZE;
#if defined(NNUE_COMPACT_PHASE5)
constexpr IndexType PHASE_OUTPUT_SIZE = 5;
constexpr IndexType PHASE_CROSS_INDEX = 4;
#else
constexpr IndexType PHASE_OUTPUT_SIZE = 6;
constexpr IndexType PHASE_CROSS_INDEX = 5;
#endif
#else
constexpr IndexType L2_INPUT_SIZE = 192;
constexpr IndexType L2_REAL_SIZE = 190;
constexpr IndexType L2_LOGICAL_SIZE = 192;
constexpr IndexType L2_PADDING_SIZE = L2_INPUT_SIZE - L2_REAL_SIZE;
constexpr IndexType L2_CROSS_OFFSET = 158;
constexpr IndexType PHASE_OUTPUT_SIZE = 6;
constexpr IndexType PHASE_CROSS_INDEX = 5;
#endif
#if defined(USE_NNUE_FC1_WIDTH_64)
constexpr IndexType kHidden2Dims = 64;
#else
constexpr IndexType kHidden2Dims = 96;
#endif

// --- [追加] Router 層の型定義 ---
using Router = Layers::AffineTransformExplicit<384, 32>;

struct Network {

	// Define network structure
	// ネットワーク構造の定義

	// --- L1 Path: メインの特徴量抽出 (インデックス31はBypassとして利用) ---
	Layers::AffineTransformSparseInputExplicit<kInputDims, kHidden1Dims + 1> fc_0;

	// --- FM Path: Factorization Machines ロジック (GLU構造による相互作用抽出) ---
	// 128次元から diff(32+32) と abs(32+32) の各ゲート・値ペアを生成
	Layers::AffineTransformExplicit<128, 64> fc_diff; 
	Layers::AffineTransformExplicit<128, 64> fc_abs;

	// 活性化関数
	Layers::SqrClippedReLU<kHidden1Dims + 1> ac_sqr_0; // 二乗による非線形抽出
	Layers::ClippedReLUExplicit<kHidden1Dims + 1> ac_0;

	// --- Deep Path: L2から最終評価値へ至る深層評価パス ---
	Layers::AffineTransformExplicit<L2_INPUT_SIZE, kHidden2Dims> fc_1;
	Layers::ClippedReLUExplicit<kHidden2Dims> ac_1;
	Layers::AffineTransformExplicit<kHidden2Dims, 1> fc_2;

	// --- Interaction Layers: パス間の相関特徴 (Cross-product) ---
	Layers::AffineTransformExplicit<32, CROSS_OUTPUT_SIZE> fc_cross;
	Layers::ClippedReLUExplicit<CROSS_OUTPUT_SIZE> ac_cross;

	// --- LCA (Lightweight Cross-Attention): コンテキストの動的統合 ---
	// Query = MainPath(31), Key/Value = FM(64)
	Layers::AffineTransformExplicit<31, 32> lca_q;
	Layers::AffineTransformExplicit<64, 32> lca_k;
	Layers::AffineTransformExplicit<64, 32> lca_v;
	float lca_temp; // Attention temperature (learned)

	// --- Phase Gate: 局面の進行度・激しさに基づく信号強度の動的制御 ---
	Layers::AffineTransformExplicit<384, 32> phase_proj;

	// Bypassパスと DeepPath のブレンド係数 (バケットごとに学習)
	int32_t bucket_blend_alpha;


	using OutputType = std::int32_t;
	static constexpr IndexType kOutputDimensions = 1;

	// Hash値などは適宜実装
	static constexpr std::uint32_t GetHashValue() {
#if defined(NNUE_COMPACT_PHASE5)
	#if defined(USE_NNUE_FC1_WIDTH_64)
		// Same serialized hash derivation as the Python writer, with fc_1
		// output width 64 instead of 96.  This prevents a 96-wide network
		// from being accepted silently by the optional 64-wide build.
		#if defined(USE_NNUE_CROSS_WIDTH_24)
			// Cross24 files use logical 24-output Cross and 152-input L2,
			// padded to 32 output rows and 160 input columns on disk.
			return 0x63726A46u;
		#else
			return 0x63566A46u;
		#endif
	#else
		// Phase5 additionally distinguishes old 160-input files whose row 4 was
		// AbsSqr and row 5 was Cross. Loading one as Phase5 would silently use
		// the wrong scale even though the physical layer is padded to 32 rows.
		return 0x63566A36u;
	#endif
#elif defined(USE_NNUE_ABS_SQR_REMOVED_160)
		return 0x63536A36u;
#else
		return 0x6333718Au;
#endif
	}

	static std::string GetStructureString() {
#if defined(NNUE_COMPACT_PHASE5)
	#if defined(USE_NNUE_FC1_WIDTH_64)
		#if defined(USE_NNUE_CROSS_WIDTH_24)
			return "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr-Phase5-FC1x64-Cross24";
		#else
			return "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr-Phase5-FC1x64";
		#endif
	#else
		return "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr-Phase5";
	#endif
#elif defined(USE_NNUE_ABS_SQR_REMOVED_160)
		return "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr";
#else
		return "HalfKA-KSDG3_FM-1280";
#endif
	}

	Tools::Result ReadParameters(std::istream& stream) {
		fc_0.ReadParameters(stream);
		fc_diff.ReadParameters(stream);
		fc_abs.ReadParameters(stream);
		lca_q.ReadParameters(stream);
		lca_k.ReadParameters(stream);
		lca_v.ReadParameters(stream);
		stream.read(reinterpret_cast<char*>(&lca_temp), sizeof(float));
		std::cout << "Read LCA Temp: " << lca_temp << std::endl;
		phase_proj.ReadParameters(stream);
		fc_cross.ReadParameters(stream).is_ok();
		fc_1.ReadParameters(stream).is_ok();
		fc_2.ReadParameters(stream).is_ok();
		stream.read(reinterpret_cast<char*>(&bucket_blend_alpha), sizeof(int32_t));
		std::cout << "Read Alpha: " << bucket_blend_alpha << " / 16384" << std::endl;
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
		// tournament builds use -fno-threadsafe-statics, so construct both the
		// source float LUT and compact Main-gate LUT on the single-threaded
		// network-loading path before search workers can evaluate positions.
		MainGateCompactQ64Lut();
#endif
#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
		// Construct before worker threads start (-fno-threadsafe-statics).
		PhaseFixedC32SigmoidQ15Lut();
#endif
		return Tools::ResultCode::Ok;
	}

	bool WriteParameters(std::ostream& stream) const {
		return true;
	}

#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
	// Production approximation for the Main and FM Abs gates. The Main gate
	// consumes its exactly quantized compact Q64 representation below; FM Abs
	// continues to consume this float interpolation directly. The Phase gate
	// intentionally keeps its existing std::exp calculation.
	static constexpr int kApproxSigmoidLutMinimum = -10;
	static constexpr int kApproxSigmoidLutMaximum = 8;
	static constexpr int kApproxSigmoidLutStepsPerUnit = 32;
	static constexpr int kApproxSigmoidLutRawStep =
		8128 / kApproxSigmoidLutStepsPerUnit;
	static constexpr int kApproxSigmoidLutRawMinimum =
		kApproxSigmoidLutMinimum * 8128;
	static constexpr int kApproxSigmoidLutRawMaximum =
		kApproxSigmoidLutMaximum * 8128;
	static constexpr int kApproxSigmoidLutPoints =
		(kApproxSigmoidLutMaximum - kApproxSigmoidLutMinimum)
			* kApproxSigmoidLutStepsPerUnit + 1;
	static_assert(8128 % kApproxSigmoidLutStepsPerUnit == 0);

	static const std::array<float, kApproxSigmoidLutPoints>&
	ApproxSigmoidLut() {
		static const auto lut = []() {
			std::array<float, kApproxSigmoidLutPoints> values{};
			for (int i = 0; i < kApproxSigmoidLutPoints; ++i) {
				const float input =
					static_cast<float>(kApproxSigmoidLutMinimum)
					+ static_cast<float>(i)
						/ static_cast<float>(kApproxSigmoidLutStepsPerUnit);
				values[i] = 1.0f / (1.0f + std::exp(-input));
			}
			return values;
		}();
		return lut;
	}

	static inline float sigmoid_lut_approx(const std::int32_t raw_x) {
		const auto& lut = ApproxSigmoidLut();
		if (raw_x <= kApproxSigmoidLutRawMinimum)
			return lut.front();
		if (raw_x >= kApproxSigmoidLutRawMaximum)
			return lut.back();

		const int offset = raw_x - kApproxSigmoidLutRawMinimum;
		const int index = offset / kApproxSigmoidLutRawStep;
		const int remainder = offset - index * kApproxSigmoidLutRawStep;
		const float fraction = static_cast<float>(remainder)
			/ static_cast<float>(kApproxSigmoidLutRawStep);
		return lut[index]
			+ (lut[index + 1] - lut[index])
				* fraction;
	}
#endif

	static inline int32_t sigmoid_gate_slow(int32_t x, int32_t value) {
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
		const float sig = sigmoid_lut_approx(x);
#else
		float sig = 1.0f / (1.0f + std::exp(-static_cast<float>(x) / 8128.0f));
#endif
		return static_cast<int32_t>(value * sig);
	}

#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
	// Exact compact representation of the quantized Main-gate result produced
	// by the float LUT above.  Exhaustive validation over all 146305 integral
	// raw inputs established that every 64-value bin has at most one Q64
	// transition.  A threshold of 65 is therefore a no-transition sentinel.
	static constexpr int kMainGateCompactRawMinimum =
		kApproxSigmoidLutRawMinimum;
	static constexpr int kMainGateCompactRawMaximum =
		kApproxSigmoidLutRawMaximum;
	static constexpr int kMainGateCompactCoarseWidth = 64;
	static constexpr std::size_t kMainGateCompactRawCount =
		static_cast<std::size_t>(kMainGateCompactRawMaximum
			- kMainGateCompactRawMinimum + 1);
	static constexpr std::size_t kMainGateCompactBinCount =
		(kMainGateCompactRawCount + kMainGateCompactCoarseWidth - 1)
			/ kMainGateCompactCoarseWidth;

	struct MainGateCompactEntry {
		std::uint8_t threshold;
		std::uint8_t base;
	};
	static_assert(sizeof(MainGateCompactEntry) == 2);

	static const std::array<MainGateCompactEntry,
		kMainGateCompactBinCount>& MainGateCompactQ64Lut() {
		static const auto table = []() {
			std::array<MainGateCompactEntry,
				kMainGateCompactBinCount> result{};
			for (std::size_t bin = 0; bin < result.size(); ++bin) {
				auto& entry = result[bin];
				const std::size_t begin =
					bin * kMainGateCompactCoarseWidth;
				const std::size_t end = std::min(
					begin + kMainGateCompactCoarseWidth,
					kMainGateCompactRawCount);
				entry.base = static_cast<std::uint8_t>(sigmoid_gate_slow(
					static_cast<std::int32_t>(begin)
						+ kMainGateCompactRawMinimum,
					64));
				entry.threshold = kMainGateCompactCoarseWidth + 1;
				for (std::size_t index = begin + 1; index < end; ++index) {
					const auto value = static_cast<std::uint8_t>(
						sigmoid_gate_slow(static_cast<std::int32_t>(index)
							+ kMainGateCompactRawMinimum,
							64));
					if (value != entry.base) {
						entry.threshold = static_cast<std::uint8_t>(
							index - begin);
						break;
					}
				}
			}
			return result;
		}();
		return table;
	}

	static inline std::int32_t MainGateCompactQ64Value(
		const std::int32_t raw_x) {
		const auto& table = MainGateCompactQ64Lut();
		const std::int32_t clamped = std::clamp(raw_x,
			kMainGateCompactRawMinimum, kMainGateCompactRawMaximum);
		const auto offset = static_cast<std::uint32_t>(
			clamped - kMainGateCompactRawMinimum);
		const auto bin = static_cast<std::size_t>(
			offset / kMainGateCompactCoarseWidth);
		const auto within_bin = static_cast<std::uint8_t>(
			offset - static_cast<std::uint32_t>(
				bin * kMainGateCompactCoarseWidth));
		const auto& entry = table[bin];
		return static_cast<std::int32_t>(entry.base)
			+ static_cast<std::int32_t>(within_bin >= entry.threshold);
	}
#endif

	static inline void ComputeMainGateSigmoid(
		const std::int32_t* diff_fc_output, std::int32_t* gate_q64
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		, NnueSignalSnapshot* signal = nullptr
#endif
	) {
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		std::uint16_t gate_sum = 0;
		std::uint8_t gate_min = 63;
		std::uint8_t gate_max = 0;
		std::uint8_t saturated_low_count = 0;
		std::uint8_t saturated_high_count = 0;
#endif
		for (int j = 0; j < 32; ++j) {
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
			gate_q64[j] = MainGateCompactQ64Value(
				diff_fc_output[j] - 2438);
#else
			gate_q64[j] = sigmoid_gate_slow(diff_fc_output[j] - 2438, 64);
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
			if (signal) {
				const auto gate = static_cast<std::uint8_t>(gate_q64[j]);
				gate_sum += gate;
				gate_min = std::min(gate_min, gate);
				gate_max = std::max(gate_max, gate);
				saturated_low_count += gate <= 1;
				saturated_high_count += gate >= 63;
			}
#endif
		}
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
			signal->main_gate_sum = gate_sum;
			signal->main_gate_min = gate_min;
			signal->main_gate_max = gate_max;
			signal->main_gate_saturated_low_count = saturated_low_count;
			signal->main_gate_saturated_high_count = saturated_high_count;
		}
#endif
	}

	static inline void ComputeMainGateApply(const std::int32_t* fc_input,
		const std::int32_t* gate_q64, std::int32_t* fc_output) {
		for (int j = 0; j < 32; ++j)
			fc_output[j] = static_cast<int32_t>(
				(fc_input[j] * (64 + gate_q64[j])) / 128);
	}

	static inline void ComputeMainGateClamp(const std::int32_t* fc_input,
		std::int32_t* fc_output) {
		for (int j = 0; j < 31; ++j)
			fc_output[j] = std::clamp(fc_input[j], 0, 8128);
		fc_output[31] = fc_input[31];
	}

	static inline void ComputeMainGate(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		, NnueSignalSnapshot* signal = nullptr
#endif
	) {
		std::int32_t gate_q64[32];
		std::int32_t before_clamp[32];
		ComputeMainGateSigmoid(diff_fc_output, gate_q64
#if defined(ENABLE_NNUE_SIGNAL_LOG)
			, signal
#endif
		);
		ComputeMainGateApply(fc_input, gate_q64, before_clamp);
		ComputeMainGateClamp(before_clamp, fc_output);
	}

	static inline void ComputeAbsSquaredScalar(const std::uint8_t* input,
		std::uint8_t* output) {
		for (int j = 0; j < 32; ++j) {
			const int32_t value = input[j];
			output[j] = static_cast<std::uint8_t>((value * value) / 127);
		}
	}

	static inline void ComputeAbsSquared(const std::uint8_t* input,
		std::uint8_t* output) {
#if defined(USE_AVX2)
		// input is the clamped Abs activation in [0, 127]. For this range,
		// mulhi_u16((x * x) + 1, 516) is exactly floor((x * x) / 127).
		const __m256i input_bytes = _mm256_loadu_si256(
			reinterpret_cast<const __m256i*>(input));
		const __m256i low = _mm256_cvtepu8_epi16(
			_mm256_castsi256_si128(input_bytes));
		const __m256i high = _mm256_cvtepu8_epi16(
			_mm256_extracti128_si256(input_bytes, 1));
		const __m256i one = _mm256_set1_epi16(1);
		const __m256i division_magic = _mm256_set1_epi16(516);

		const __m256i low_squared = _mm256_mullo_epi16(low, low);
		const __m256i high_squared = _mm256_mullo_epi16(high, high);
		const __m256i low_quotient = _mm256_mulhi_epu16(
			_mm256_add_epi16(low_squared, one), division_magic);
		const __m256i high_quotient = _mm256_mulhi_epu16(
			_mm256_add_epi16(high_squared, one), division_magic);

		const __m256i packed = _mm256_packus_epi16(
			low_quotient, high_quotient);
		const __m256i ordered = _mm256_permute4x64_epi64(packed, 0xd8);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(output), ordered);
#else
		ComputeAbsSquaredScalar(input, output);
#endif
	}

	template<IndexType Dimensions>
	static inline void AssembleL2Channel(const std::uint8_t* input,
		std::uint8_t* output, const float scale) {
#if defined(USE_AVX2)
		constexpr IndexType kSimdDimensions = (Dimensions / 8) * 8;
		const __m256 scale_vector = _mm256_set1_ps(scale);
		const __m256i zero = _mm256_setzero_si256();
		const __m256i maximum = _mm256_set1_epi32(127);

		for (IndexType i = 0; i < kSimdDimensions; i += 8) {
			const __m128i input_bytes = _mm_loadl_epi64(
				reinterpret_cast<const __m128i*>(input + i));
			const __m256i input_int32 = _mm256_cvtepu8_epi32(input_bytes);
			const __m256 scaled = _mm256_mul_ps(
				_mm256_cvtepi32_ps(input_int32), scale_vector);
			const __m256i truncated = _mm256_cvttps_epi32(scaled);
			const __m256i clamped = _mm256_min_epi32(
				_mm256_max_epi32(truncated, zero), maximum);

			const __m128i packed16 = _mm_packus_epi32(
				_mm256_castsi256_si128(clamped),
				_mm256_extracti128_si256(clamped, 1));
			const __m128i packed8 = _mm_packus_epi16(
				packed16, _mm_setzero_si128());
			_mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), packed8);
		}
#else
		constexpr IndexType kSimdDimensions = 0;
#endif

		for (IndexType i = kSimdDimensions; i < Dimensions; ++i) {
			output[i] = static_cast<std::uint8_t>(
				std::clamp<int>(input[i] * scale, 0, 127));
		}
	}

	static constexpr std::int32_t kPhaseFixedRawMinimum = -65536;
	static constexpr std::int32_t kPhaseFixedRawMaximum = 65536;
	static constexpr std::int32_t kPhaseFixedC32RawStep = 32;
	static constexpr std::size_t kPhaseFixedC32LutSize =
		(kPhaseFixedRawMaximum - kPhaseFixedRawMinimum)
			/ kPhaseFixedC32RawStep + 1;

	static const std::array<std::uint16_t, kPhaseFixedC32LutSize>&
	PhaseFixedC32SigmoidQ15Lut() {
		static const auto lut = []() {
			std::array<std::uint16_t, kPhaseFixedC32LutSize> values{};
			for (std::size_t i = 0; i < values.size(); ++i) {
				const std::int32_t raw = kPhaseFixedRawMinimum
					+ static_cast<std::int32_t>(i) * kPhaseFixedC32RawStep;
				const float logit =
					(static_cast<float>(raw) / 8128.0f) * 3.0f + 1.0f;
				const float sigmoid = 1.0f / (1.0f + std::exp(-logit));
				values[i] = static_cast<std::uint16_t>(std::clamp(
					static_cast<int>(std::lround(sigmoid * 32768.0f)),
					0, 32768));
			}
			return values;
		}();
		return lut;
	}

	static inline void ComputePhaseFixedC32ScalesQ23(
		const std::int32_t* phase_output, std::int32_t* scales_q23) {
		const auto& lut = PhaseFixedC32SigmoidQ15Lut();
		constexpr std::int32_t kBaseQ15 = 18022;
		constexpr std::int32_t kGainQ15 = 14746;
#if defined(NNUE_COMPACT_PHASE5)
		constexpr std::array<std::int32_t, PHASE_OUTPUT_SIZE> kFactorQ15 = {
			42598, 49152, 32768, 22938, 49152};
#else
		constexpr std::array<std::int32_t, PHASE_OUTPUT_SIZE> kFactorQ15 = {
			42598, 49152, 32768, 22938, 28836, 49152};
#endif
		for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
			const std::int32_t raw = phase_output[i];
			std::uint16_t sigmoid_q15;
			if (raw <= kPhaseFixedRawMinimum)
				sigmoid_q15 = lut.front();
			else if (raw >= kPhaseFixedRawMaximum)
				sigmoid_q15 = lut.back();
			else {
				const auto index = static_cast<std::size_t>(
					(raw - kPhaseFixedRawMinimum + kPhaseFixedC32RawStep / 2)
						/ kPhaseFixedC32RawStep);
				sigmoid_q15 = lut[index];
			}
			const std::int32_t base_q15 = kBaseQ15
				+ (static_cast<std::int32_t>(sigmoid_q15) * kGainQ15
					+ (1 << 14)) / (1 << 15);
			scales_q23[i] = (base_q15 * kFactorQ15[i] + 64) >> 7;
		}
	}

	template<IndexType Dimensions>
	static inline void AssembleL2ChannelQ23(
		const std::uint8_t* input, std::uint8_t* output,
		const std::int32_t scale_q23) {
		// input <= 127 and scale <= 1.5, so the maximum product is
		// 127 * round(1.5 * 2^23) = 1,597,829,824 < INT32_MAX.
#if defined(USE_AVX2)
		constexpr IndexType kSimdDimensions = Dimensions / 8 * 8;
		const __m256i scale = _mm256_set1_epi32(scale_q23);
		const __m256i upper = _mm256_set1_epi32(127);
		for (IndexType i = 0; i < kSimdDimensions; i += 8) {
			const __m128i bytes = _mm_loadl_epi64(
				reinterpret_cast<const __m128i*>(input + i));
			const __m256i values = _mm256_cvtepu8_epi32(bytes);
			const __m256i product = _mm256_mullo_epi32(values, scale);
			const __m256i shifted = _mm256_srli_epi32(product, 23);
			const __m256i clamped = _mm256_min_epi32(shifted, upper);
			const __m128i packed16 = _mm_packus_epi32(
				_mm256_castsi256_si128(clamped),
				_mm256_extracti128_si256(clamped, 1));
			const __m128i packed8 = _mm_packus_epi16(
				packed16, _mm_setzero_si128());
			_mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), packed8);
		}
#else
		constexpr IndexType kSimdDimensions = 0;
#endif
		for (IndexType i = kSimdDimensions; i < Dimensions; ++i) {
			const std::int32_t scaled =
				(static_cast<std::int32_t>(input[i]) * scale_q23) >> 23;
			output[i] = static_cast<std::uint8_t>(std::min(scaled, 127));
		}
	}

	struct alignas(kCacheLineSize) Buffer {
		// 各レイヤーの中間出力を保持するバッファ
		alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
		alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
		alignas(kCacheLineSize) decltype(ac_sqr_0)::OutputBuffer ac_sqr_0_out_temp;

		// FM Path: 特徴量抽出用
		alignas(kCacheLineSize) std::int32_t diff_fc_out[64];
		alignas(kCacheLineSize) std::int32_t abs_fc_out[64];
		alignas(kCacheLineSize) std::uint8_t diff_ac_out[32];
		alignas(kCacheLineSize) std::uint8_t abs_ac_out[32];
		alignas(kCacheLineSize) std::uint8_t abs_sqr_out[32];

		// LCA (Attention) 用
		alignas(kCacheLineSize) std::int32_t lca_q_out[32];
		alignas(kCacheLineSize) std::int32_t lca_k_out[32];
		alignas(kCacheLineSize) std::int32_t lca_v_out[32];
		alignas(kCacheLineSize) std::uint8_t fm_cat_uint8[64];

		// Cross Feature (相互作用) 用
		alignas(kCacheLineSize) std::uint8_t cross_cat[32];
		alignas(kCacheLineSize) std::int32_t cross_fc_out[32];
		alignas(kCacheLineSize) std::uint8_t cross_feat[32];

		// Phase Gate (動的スケーリング) 用
		alignas(kCacheLineSize) std::uint8_t phase_input[384]; 
		alignas(kCacheLineSize) std::int32_t phase_out[32];

		// L2 Input & 深層パス用
		alignas(kCacheLineSize) std::uint8_t l2_input[L2_INPUT_SIZE];
		alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
		alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
		alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
	};

	static constexpr std::size_t kBufferSize = sizeof(Buffer);

	static inline void PropagateCrossActivation(
		const std::int32_t* input, std::uint8_t* output) {
#if defined(USE_NNUE_CROSS_WIDTH_24) && defined(USE_AVX2)
		// ClippedReLUExplicit<24> cannot use its 32-element AVX2 kernel.
		// Convert three independent groups of eight with the same operation
		// order as the generic layer: saturating int32->int16 pack, signed
		// >> kWeightScaleBits, saturating int16->int8 pack, max with zero.
		// Keeping only two XMM input registers per group limits register use.
		const __m128i zero = _mm_setzero_si128();
		for (IndexType base = 0; base < CROSS_OUTPUT_SIZE; base += 8) {
			const __m128i lo = _mm_load_si128(
				reinterpret_cast<const __m128i*>(input + base));
			const __m128i hi = _mm_load_si128(
				reinterpret_cast<const __m128i*>(input + base + 4));
			const __m128i words = _mm_srai_epi16(
				_mm_packs_epi32(lo, hi), kWeightScaleBits);
			const __m128i bytes = _mm_max_epi8(
				_mm_packs_epi16(words, zero), zero);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(output + base), bytes);
		}
#else
		Layers::ClippedReLUExplicit<CROSS_OUTPUT_SIZE> activation;
		activation.Propagate(input, output);
#endif
	}

	template<bool UsePhasePrefix = true, bool PhaseInputPrepared = false,
#if defined(USE_NNUE_PHASE_L2_FIXED_C32)
		bool UseFixedPhaseL2 = true>
#else
		bool UseFixedPhaseL2 = false>
#endif
	const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, const TransformedFeatureType* diffFeatures, const TransformedFeatureType* absFeatures, const int bucket_id, char* buffer
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		, NnueSignalSnapshot* signal = nullptr
#endif
#if (defined(USE_NNUE_PHASE_FM_LMR) || defined(USE_NNUE_LCA_LMR)) \
	&& !defined(ENABLE_NNUE_SIGNAL_LOG)
		, NnueRouterLmrSignal* lmr_signal = nullptr
#endif
	) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);

		// --- 1. Phase Gate: 局面の進行度や激しさに応じた動的スケーリング ---
		// The production caller prepares this same 384-byte input for Router and
		// overwrites only byte 127 after Router returns. Other callers retain the
		// self-contained assembly path. Both choices are compile-time, so the
		// production path has no runtime branch.
		if constexpr (!PhaseInputPrepared) {
			for (int j = 0; j < 128; ++j) {
				int32_t abs_val = static_cast<int32_t>(absFeatures[j]);
				buf.phase_input[j] = static_cast<std::uint8_t>(std::clamp((abs_val - 64) * 2, 0, 127));
				buf.phase_input[j + 128] = diffFeatures[j];
				buf.phase_input[j + 256] = transformedFeatures[j];
			}
			// 入力のabs部分の末尾を「バケットID（0～11）/11」で置き換える
			buf.phase_input[127] = static_cast<std::uint8_t>((bucket_id * 127) / 11);
		}

		// Phase Gate 推論と各パスへの係数算出 (0.1 ～ 1.0 の範囲に正規化)
		if constexpr (UsePhasePrefix)
			phase_proj.PropagatePrefix<PHASE_OUTPUT_SIZE>(buf.phase_input, buf.phase_out);
		else
			phase_proj.Propagate(buf.phase_input, buf.phase_out);
		std::int32_t phase_scales_q23[PHASE_OUTPUT_SIZE]{};
		float main_sqr_scale = 0.0f;
		float main_raw_scale = 0.0f;
		float diff_scale = 0.0f;
		float abs_raw_scale = 0.0f;
		float abs_sqr_scale = 0.0f;
		float cross_scale = 0.0f;
		if constexpr (UseFixedPhaseL2) {
			ComputePhaseFixedC32ScalesQ23(buf.phase_out, phase_scales_q23);
#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_PHASE_FM_LMR)
			constexpr float kInverseQ23 = 1.0f / static_cast<float>(1 << 23);
			main_sqr_scale = phase_scales_q23[0] * kInverseQ23;
			main_raw_scale = phase_scales_q23[1] * kInverseQ23;
			diff_scale = phase_scales_q23[2] * kInverseQ23;
			abs_raw_scale = phase_scales_q23[3] * kInverseQ23;
#if !defined(NNUE_COMPACT_PHASE5)
			abs_sqr_scale = phase_scales_q23[4] * kInverseQ23;
#endif
			cross_scale = phase_scales_q23[PHASE_CROSS_INDEX] * kInverseQ23;
#endif
		} else {
			float phase_val[PHASE_OUTPUT_SIZE];
			for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
				const float logit =
					(static_cast<float>(buf.phase_out[i]) / 8128.0f) * 3.0f + 1.0f;
				const float sig = 1.0f / (1.0f + std::exp(-logit));
				phase_val[i] = 0.1f + 0.9f * sig;
			}
			main_sqr_scale = (0.5f + 0.5f * phase_val[0]) * 1.3f;
			main_raw_scale = (0.5f + 0.5f * phase_val[1]) * 1.5f;
			diff_scale = (0.5f + 0.5f * phase_val[2]) * 1.0f;
			abs_raw_scale = (0.5f + 0.5f * phase_val[3]) * 0.7f;
#if !defined(NNUE_COMPACT_PHASE5)
			abs_sqr_scale = (0.5f + 0.5f * phase_val[4]) * 0.88f;
#endif
			cross_scale = (0.5f + 0.5f * phase_val[PHASE_CROSS_INDEX]) * 1.5f;
		}

#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
			signal->phase_scale[0] = main_sqr_scale;
			signal->phase_scale[1] = main_raw_scale;
			signal->phase_scale[2] = diff_scale;
			signal->phase_scale[3] = abs_raw_scale;
			signal->phase_scale[4] = abs_sqr_scale;
			signal->phase_scale[5] = cross_scale;
			signal->main_reliance = main_sqr_scale + main_raw_scale;
			signal->fm_reliance = diff_scale + abs_raw_scale;
			signal->cross_reliance = cross_scale;
		}
#endif
#if defined(USE_NNUE_PHASE_FM_LMR) && !defined(ENABLE_NNUE_SIGNAL_LOG)
		if (lmr_signal)
			lmr_signal->fm_reliance = diff_scale + abs_raw_scale;
#endif


		// --- 2. FM Path: Factorization Machines 的な相互作用抽出 ---
		fc_diff.Propagate(diffFeatures, buf.diff_fc_out);
		fc_abs.Propagate(absFeatures, buf.abs_fc_out);

		// Diff Path: RMSNorm を適用して信号の分散を安定化
		float sum_sq_d = 0.0f;
		for (int j = 0; j < 32; ++j) {
			float vd_f = static_cast<float>(buf.diff_fc_out[j + 32]); // val_d
			sum_sq_d += vd_f * vd_f;
		}
		#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal)
			signal->diff_rms_energy_sum = sum_sq_d;
		#endif
		float inv_rms_d = 1.0f / std::sqrt(sum_sq_d / 32.0f + 1e-8f);

#if defined(ENABLE_NNUE_SIGNAL_LOG)
		std::uint16_t fm_diff_activity_sum = 0;
		std::uint16_t fm_abs_activity_sum = 0;
		std::uint8_t fm_diff_activity_max = 0;
		std::uint8_t fm_abs_activity_max = 0;
		std::uint8_t fm_diff_saturated_count = 0;
		std::uint8_t fm_abs_saturated_count = 0;
#endif
		for (int j = 0; j < 32; ++j) {
			int32_t gd = buf.diff_fc_out[j];      // gate_d
			int32_t vd = buf.diff_fc_out[j + 32]; // val_d
			int32_t ga = buf.abs_fc_out[j];       // gate_a
			int32_t va = buf.abs_fc_out[j + 32];  // val_a

			// Diff Path: Norm適用後、量子化スケールに合わせて 0.5 基準で配置
			float vd_normed_f = static_cast<float>(vd) * inv_rms_d;
			int32_t d_scaled = static_cast<int32_t>(vd_normed_f * 25.4f) + 64;
			buf.diff_ac_out[j] = static_cast<uint8_t>(std::max(0, std::min(127, d_scaled)));

			// Abs Path: gateによるフィルタリング (GLU構造)
			int32_t a_gated = sigmoid_gate_slow(ga, va);
			float abs_gated = static_cast<float>(a_gated) / 8128.0f;
			int32_t a_scaled = static_cast<int32_t>(
				std::round(
					std::clamp(abs_gated * 0.05f + 0.6f, 0.0f, 1.0f) * 127.0f
				)
			);
			buf.abs_ac_out[j] = static_cast<uint8_t>(a_scaled);

#if defined(ENABLE_NNUE_SIGNAL_LOG)
			if (signal) {
				// Diff is offset-binary: 64 is neutral and both clamp endpoints
				// are saturation. Abs is a nonnegative [0,127] activation.
				const auto diff_q = buf.diff_ac_out[j];
				const auto diff_activity = static_cast<std::uint8_t>(
					std::abs(static_cast<int>(diff_q) - 64));
				fm_diff_activity_sum += diff_activity;
				fm_diff_activity_max = std::max(fm_diff_activity_max, diff_activity);
				fm_diff_saturated_count += diff_q <= 1 || diff_q >= 126;

				const auto abs_q = buf.abs_ac_out[j];
				fm_abs_activity_sum += abs_q;
				fm_abs_activity_max = std::max(fm_abs_activity_max, abs_q);
				fm_abs_saturated_count += abs_q >= 126;
			}
#endif
		}
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
			signal->fm_diff_activity_sum = fm_diff_activity_sum;
			signal->fm_diff_activity_max = fm_diff_activity_max;
			signal->fm_diff_saturated_count = fm_diff_saturated_count;
			signal->fm_abs_activity_sum = fm_abs_activity_sum;
			signal->fm_abs_activity_max = fm_abs_activity_max;
			signal->fm_abs_saturated_count = fm_abs_saturated_count;
		}
#endif
		// AbsSqr is not consumed by the compact 160-input architecture.  Keep
		// its Phase channel in the file/model for minimal format disruption, but
		// do not spend inference time materializing the removed L2 channel.
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
		ComputeAbsSquared(buf.abs_ac_out, buf.abs_sqr_out);
#endif


		// --- 3. Main Path: 基本骨格パスと FM による動的フィルタリング ---
		fc_0.Propagate(transformedFeatures, buf.fc_0_out);
		// FM 側の信号（gate_d）で Main パスの情報の通りやすさを制御
		ComputeMainGate(buf.fc_0_out, buf.diff_fc_out, buf.fc_0_out
#if defined(ENABLE_NNUE_SIGNAL_LOG)
			, signal
#endif
		);

		ac_sqr_0.Propagate(buf.fc_0_out, buf.ac_sqr_0_out_temp); 
		ac_0.Propagate(buf.fc_0_out, buf.ac_0_out);


		// --- 4. LCA (Lightweight Cross-Attention): Main と FM のコンテキスト統合 ---
		for (int j = 0; j < 32; ++j) {
			buf.fm_cat_uint8[j] = buf.diff_ac_out[j];
			buf.fm_cat_uint8[j + 32] = buf.abs_ac_out[j];
		}

		// Query (Mainパス) と Key/Value (FMパス) の相互アテンション
		lca_q.Propagate(buf.ac_0_out, buf.lca_q_out);
		lca_k.Propagate(buf.fm_cat_uint8, buf.lca_k_out);
		lca_v.Propagate(buf.fm_cat_uint8, buf.lca_v_out);

		// 内積による Attention Score 算出
		float dot_product = 0.0f;
		for (int j = 0; j < 32; ++j) {
			dot_product += (static_cast<float>(buf.lca_q_out[j]) / 8128.0f) * (static_cast<float>(buf.lca_k_out[j]) / 8128.0f);
		}

		// スケーリング因子 1/sqrt(d_k) = 1/sqrt(32) ≒ 0.1767
		float att_logit = (dot_product * 0.17677f) / lca_temp;
		float att_score = 1.0f / (1.0f + std::exp(-att_logit));

		// アテンションスコアに基づき、FM Diff Path の情報を動的に書き換え (Attention Blend)
#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_LCA_LMR)
		int lca_abs_delta_sum = 0;
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		int lca_abs_delta_max = 0;
#endif
		for (int j = 0; j < 32; ++j) {
#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_LCA_LMR)
			const int diff_before_lca = buf.diff_ac_out[j];
#endif
			float current_diff = static_cast<float>(buf.diff_ac_out[j]) / 127.0f;
			float v_val = static_cast<float>(buf.lca_v_out[j]) / 8128.0f;
			float v_clamped = std::max(0.0f, std::min(1.0f, v_val * 0.4f + 0.5f));
			float final_diff_f = current_diff * (1.0f - att_score) + v_clamped * att_score;
			buf.diff_ac_out[j] = static_cast<uint8_t>(final_diff_f * 127.0f);
#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_LCA_LMR)
			const int lca_abs_delta = std::abs(static_cast<int>(buf.diff_ac_out[j])
			                                   - diff_before_lca);
			lca_abs_delta_sum += lca_abs_delta;
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
			lca_abs_delta_max = std::max(lca_abs_delta_max, lca_abs_delta);
#endif
		}
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
			signal->lca_mean_abs_delta = static_cast<float>(lca_abs_delta_sum) / 32.0f;
			signal->lca_max_abs_delta = lca_abs_delta_max;
			signal->lca_abs_delta_sum = lca_abs_delta_sum;
		}
#endif
#if defined(USE_NNUE_LCA_LMR) && !defined(ENABLE_NNUE_SIGNAL_LOG)
		if (lmr_signal)
			lmr_signal->lca_abs_delta_sum = lca_abs_delta_sum;
#endif


		// --- 5. Cross Feature: 異種パス間の積による相関特徴の生成 ---
		for (int j = 0; j < 16; ++j) {
			buf.cross_cat[j]      = (uint8_t)((buf.ac_sqr_0_out_temp[j] * buf.diff_ac_out[j]) / 127);
			buf.cross_cat[j + 16] = (uint8_t)((buf.ac_0_out[j] * buf.abs_ac_out[j]) / 127);
		}

		fc_cross.PropagatePrefix<CROSS_OUTPUT_SIZE>(buf.cross_cat, buf.cross_fc_out);
		PropagateCrossActivation(buf.cross_fc_out, buf.cross_feat);
#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
#if defined(USE_AVX2)
			// One 32-byte load replaces a diagnostic 32-element reduction loop.
			// cross_feat is uint8_t in [0,127], hence abs(value) == value and
			// the exact sum is at most 32*127=4064 (fits uint16_t).
			__m256i values = _mm256_load_si256(
				reinterpret_cast<const __m256i*>(buf.cross_feat));
#if defined(USE_NNUE_CROSS_WIDTH_24)
			// Ignore the unused high eight bytes of the padded output buffer.
			values = _mm256_and_si256(values, _mm256_set_epi64x(0, -1, -1, -1));
#endif
			const __m256i sums = _mm256_sad_epu8(values, _mm256_setzero_si256());
			const std::uint64_t sum =
				static_cast<std::uint64_t>(_mm256_extract_epi64(sums, 0))
				+ static_cast<std::uint64_t>(_mm256_extract_epi64(sums, 1))
				+ static_cast<std::uint64_t>(_mm256_extract_epi64(sums, 2))
				+ static_cast<std::uint64_t>(_mm256_extract_epi64(sums, 3));
			__m128i maximum = _mm_max_epu8(
				_mm256_castsi256_si128(values), _mm256_extracti128_si256(values, 1));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 8));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 4));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 2));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 1));
			signal->cross_abs_sum = static_cast<std::uint16_t>(sum);
			signal->cross_abs_max = static_cast<std::uint8_t>(
				_mm_cvtsi128_si32(maximum) & 0xff);
#else
			std::uint16_t sum = 0;
			std::uint8_t maximum = 0;
			for (IndexType j = 0; j < CROSS_OUTPUT_SIZE; ++j) {
				sum = static_cast<std::uint16_t>(sum + buf.cross_feat[j]);
				maximum = std::max(maximum, buf.cross_feat[j]);
			}
			signal->cross_abs_sum = sum;
			signal->cross_abs_max = maximum;
#endif
		}
#endif

		// Production Cross-LMR retains only the maximum needed by search.  The
		// diagnostic build above also records the exact sum for analysis.
#if defined(USE_NNUE_CROSS_LMR) && !defined(ENABLE_NNUE_SIGNAL_LOG)
		if (lmr_signal) {
#if defined(USE_AVX2)
			__m256i values = _mm256_load_si256(
				reinterpret_cast<const __m256i*>(buf.cross_feat));
#if defined(USE_NNUE_CROSS_WIDTH_24)
			values = _mm256_and_si256(values, _mm256_set_epi64x(0, -1, -1, -1));
#endif
			__m128i maximum = _mm_max_epu8(
				_mm256_castsi256_si128(values), _mm256_extracti128_si256(values, 1));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 8));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 4));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 2));
			maximum = _mm_max_epu8(maximum, _mm_srli_si128(maximum, 1));
			lmr_signal->cross_abs_max = static_cast<std::uint8_t>(
				_mm_cvtsi128_si32(maximum) & 0xff);
#else
			std::uint8_t maximum = 0;
			for (IndexType j = 0; j < CROSS_OUTPUT_SIZE; ++j)
				maximum = std::max(maximum, buf.cross_feat[j]);
			lmr_signal->cross_abs_max = maximum;
#endif
		}
#endif

		// --- 6. L2 Input Assembly: 深層評価パスへの入力構築 ---
		// 各チャネルを Phase Gate で得たスケールで調整しつつ統合
		// 192: MainSqr31, MainRaw31, Diff32, AbsRaw32, AbsSqr32, Cross32, Pad2.
		// 160: MainSqr31, MainRaw31, Diff32, AbsRaw32, Cross32, Pad2.
		if constexpr (UseFixedPhaseL2) {
			AssembleL2ChannelQ23<31>(buf.ac_sqr_0_out_temp, &buf.l2_input[0], phase_scales_q23[0]);
			AssembleL2ChannelQ23<31>(buf.ac_0_out, &buf.l2_input[31], phase_scales_q23[1]);
			AssembleL2ChannelQ23<32>(buf.diff_ac_out, &buf.l2_input[62], phase_scales_q23[2]);
			AssembleL2ChannelQ23<32>(buf.abs_ac_out, &buf.l2_input[94], phase_scales_q23[3]);
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
			AssembleL2ChannelQ23<32>(buf.abs_sqr_out, &buf.l2_input[126], phase_scales_q23[4]);
#endif
			AssembleL2ChannelQ23<CROSS_OUTPUT_SIZE>(buf.cross_feat, &buf.l2_input[L2_CROSS_OFFSET], phase_scales_q23[PHASE_CROSS_INDEX]);
		} else {
			AssembleL2Channel<31>(buf.ac_sqr_0_out_temp, &buf.l2_input[0], main_sqr_scale);
			AssembleL2Channel<31>(buf.ac_0_out, &buf.l2_input[31], main_raw_scale);
			AssembleL2Channel<32>(buf.diff_ac_out, &buf.l2_input[62], diff_scale);
			AssembleL2Channel<32>(buf.abs_ac_out, &buf.l2_input[94], abs_raw_scale);
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
			AssembleL2Channel<32>(buf.abs_sqr_out, &buf.l2_input[126], abs_sqr_scale);
#endif
			AssembleL2Channel<CROSS_OUTPUT_SIZE>(buf.cross_feat, &buf.l2_input[L2_CROSS_OFFSET], cross_scale);
		}
		std::memset(buf.l2_input + L2_REAL_SIZE, 0, L2_PADDING_SIZE);


		// --- 7. Deep Path 推論 ---
		fc_1.Propagate(buf.l2_input, buf.fc_1_out);
		ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
		fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);


		// --- 8. Final Blending ---
		const int32_t alpha = bucket_blend_alpha;
		const int32_t inv_alpha = 16384 - alpha;

		// Main パスの 31 番目の要素を Bypass Path（直接出力）として利用
		int32_t fwdOut_main = (int(buf.fc_0_out[31]) * (600 * 16)) / (127 * 64);

#if defined(ENABLE_NNUE_SIGNAL_LOG)
		if (signal) {
			signal->deep_output = buf.fc_2_out[0];
			signal->bypass_output = fwdOut_main;
			const std::int64_t delta = static_cast<std::int64_t>(buf.fc_2_out[0]) - fwdOut_main;
			signal->signed_deep_bypass = static_cast<std::int32_t>(
				std::clamp<std::int64_t>(delta, std::numeric_limits<std::int32_t>::min(),
				                         std::numeric_limits<std::int32_t>::max()));
			signal->deep_bypass_disagreement = static_cast<std::int32_t>(
				std::min<std::int64_t>(std::llabs(delta), std::numeric_limits<std::int32_t>::max()));
		}
#endif

		// Deep Path (L3) と Bypass Path をバケットごとの alpha で加重平均
		int64_t combined = (static_cast<int64_t>(buf.fc_2_out[0]) * bucket_blend_alpha) + 
						   (static_cast<int64_t>(fwdOut_main) * (16384 - alpha));

		buf.fc_2_out[0] = static_cast<int32_t>(combined / 16384);
		return buf.fc_2_out;
	}

#if defined(ENABLE_NNUE_BENCH)
	// Benchmark-only stage entry points. These intentionally duplicate the normal
	// Propagate() arithmetic so isolated stage timing does not add branches or
	// instrumentation to the production evaluation path.
	enum class BenchmarkSigmoidImplementation {
		StdExp,
		FloatLutMinus8To8Step16,
		FloatLutMinus10To8Step16,
		FloatLutMinus8To8Step32,
		FloatLutMinus10To8Step32,
	};

	template<int Minimum, int Maximum, int StepsPerUnit>
	static const auto& BenchmarkConfiguredFloatSigmoidLut() {
		static_assert(8128 % StepsPerUnit == 0);
		constexpr int kPoints =
			(Maximum - Minimum) * StepsPerUnit + 1;
		static const auto lut = []() {
			std::array<float, kPoints> values{};
			for (int i = 0; i < kPoints; ++i) {
				const float x = static_cast<float>(Minimum)
					+ static_cast<float>(i) /
						static_cast<float>(StepsPerUnit);
				values[i] = 1.0f / (1.0f + std::exp(-x));
			}
			return values;
		}();
		return lut;
	}

	template<int Minimum, int Maximum, int StepsPerUnit>
	static float BenchmarkConfiguredSigmoidFloatLutLinear(
		const std::int32_t raw_x) {
		constexpr int kRawStep = 8128 / StepsPerUnit;
		constexpr int kRawMinimum = Minimum * 8128;
		constexpr int kRawMaximum = Maximum * 8128;
		const auto& lut = BenchmarkConfiguredFloatSigmoidLut<
			Minimum, Maximum, StepsPerUnit>();
		if (raw_x <= kRawMinimum)
			return lut.front();
		if (raw_x >= kRawMaximum)
			return lut.back();

		const int offset = raw_x - kRawMinimum;
		const int index = offset / kRawStep;
		const int remainder = offset - index * kRawStep;
		const float fraction =
			static_cast<float>(remainder) / static_cast<float>(kRawStep);
		return lut[index] + (lut[index + 1] - lut[index]) * fraction;
	}

	template<BenchmarkSigmoidImplementation Implementation>
	static float BenchmarkSigmoidValue(const std::int32_t raw_x) {
		if constexpr (Implementation == BenchmarkSigmoidImplementation::StdExp)
			return 1.0f /
				(1.0f + std::exp(-static_cast<float>(raw_x) / 8128.0f));
		else if constexpr (Implementation ==
			BenchmarkSigmoidImplementation::FloatLutMinus8To8Step16)
			return BenchmarkConfiguredSigmoidFloatLutLinear<-8, 8, 16>(raw_x);
		else if constexpr (Implementation ==
			BenchmarkSigmoidImplementation::FloatLutMinus10To8Step16)
			return BenchmarkConfiguredSigmoidFloatLutLinear<-10, 8, 16>(raw_x);
		else if constexpr (Implementation ==
			BenchmarkSigmoidImplementation::FloatLutMinus8To8Step32)
			return BenchmarkConfiguredSigmoidFloatLutLinear<-8, 8, 32>(raw_x);
		else if constexpr (Implementation ==
			BenchmarkSigmoidImplementation::FloatLutMinus10To8Step32) {
#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
			return sigmoid_lut_approx(raw_x);
#else
			return BenchmarkConfiguredSigmoidFloatLutLinear<-10, 8, 32>(raw_x);
#endif
		}
	}

	template<BenchmarkSigmoidImplementation Implementation>
	static std::int32_t BenchmarkSigmoidGate(const std::int32_t raw_x,
		const std::int32_t value) {
		return static_cast<std::int32_t>(
			static_cast<float>(value) *
			BenchmarkSigmoidValue<Implementation>(raw_x));
	}

	template<BenchmarkSigmoidImplementation Implementation>
	void BenchmarkMainGateSigmoidCandidate(
		const std::int32_t* diff_fc_output, std::int32_t* gate_q64) const {
		for (int j = 0; j < 32; ++j)
			gate_q64[j] = BenchmarkSigmoidGate<Implementation>(
				diff_fc_output[j] - 2438, 64);
	}

	template<BenchmarkSigmoidImplementation Implementation>
	void BenchmarkAbsSigmoidGateCandidate(
		const std::int32_t* abs_fc_output,
		std::int32_t* abs_gated_output) const {
		for (int j = 0; j < 32; ++j)
			abs_gated_output[j] = BenchmarkSigmoidGate<Implementation>(
				abs_fc_output[j], abs_fc_output[j + 32]);
	}

	template<BenchmarkSigmoidImplementation Implementation>
	void BenchmarkMainGateCandidate(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output) const {
		std::int32_t gate_q64[32];
		std::int32_t before_clamp[32];
		BenchmarkMainGateSigmoidCandidate<Implementation>(
			diff_fc_output, gate_q64);
		ComputeMainGateApply(fc_input, gate_q64, before_clamp);
		ComputeMainGateClamp(before_clamp, fc_output);
	}


	struct BenchmarkPhaseScales {
		float main_sqr;
		float main_raw;
		float diff;
		float abs_raw;
		float abs_sqr;
		float cross;
	};

	void BenchmarkPhaseInputAssembly(
		const TransformedFeatureType* transformed_features,
		const TransformedFeatureType* diff_features,
		const TransformedFeatureType* abs_features, const int bucket_id,
		std::uint8_t* phase_input) const {
		for (int j = 0; j < 128; ++j) {
			const int32_t abs_value = static_cast<int32_t>(abs_features[j]);
			phase_input[j] = static_cast<std::uint8_t>(
				std::clamp((abs_value - 64) * 2, 0, 127));
			phase_input[j + 128] = diff_features[j];
			phase_input[j + 256] = transformed_features[j];
		}
		phase_input[127] =
			static_cast<std::uint8_t>((bucket_id * 127) / 11);
	}

	void BenchmarkPhaseProjection(const std::uint8_t* phase_input,
		std::int32_t* phase_output) const {
		phase_proj.PropagatePrefix<PHASE_OUTPUT_SIZE>(phase_input, phase_output);
	}

	void BenchmarkPhaseSigmoid(const std::int32_t* phase_output,
		float* phase_value) const {
		for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
			const float logit =
				(static_cast<float>(phase_output[i]) / 8128.0f) * 3.0f + 1.0f;
			const float sigmoid = 1.0f / (1.0f + std::exp(-logit));
			phase_value[i] = 0.1f + 0.9f * sigmoid;
		}
	}

	BenchmarkPhaseScales BenchmarkPhaseChannelScales(
		const float* phase_value) const {
#if defined(NNUE_COMPACT_PHASE5)
		return {
			(0.5f + 0.5f * phase_value[0]) * 1.3f,
			(0.5f + 0.5f * phase_value[1]) * 1.5f,
			(0.5f + 0.5f * phase_value[2]) * 1.0f,
			(0.5f + 0.5f * phase_value[3]) * 0.7f,
			0.0f,
			(0.5f + 0.5f * phase_value[4]) * 1.5f};
#else
		return {
			(0.5f + 0.5f * phase_value[0]) * 1.3f,
			(0.5f + 0.5f * phase_value[1]) * 1.5f,
			(0.5f + 0.5f * phase_value[2]) * 1.0f,
			(0.5f + 0.5f * phase_value[3]) * 0.7f,
			(0.5f + 0.5f * phase_value[4]) * 0.88f,
			(0.5f + 0.5f * phase_value[5]) * 1.5f};
#endif
	}

	BenchmarkPhaseScales BenchmarkPhaseScalesFromOutput(
		const std::int32_t* phase_output) const {
		float phase_value[PHASE_OUTPUT_SIZE];
		for (IndexType i = 0; i < PHASE_OUTPUT_SIZE; ++i) {
			const float logit =
				(static_cast<float>(phase_output[i]) / 8128.0f) * 3.0f + 1.0f;
			const float sigmoid = 1.0f / (1.0f + std::exp(-logit));
			phase_value[i] = 0.1f + 0.9f * sigmoid;
		}

		return BenchmarkPhaseChannelScales(phase_value);
	}

	BenchmarkPhaseScales BenchmarkPhase(
		const TransformedFeatureType* transformed_features,
		const TransformedFeatureType* diff_features,
		const TransformedFeatureType* abs_features, const int bucket_id,
		std::uint8_t* phase_input, std::int32_t* phase_output) const {
		for (int j = 0; j < 128; ++j) {
			const int32_t abs_value = static_cast<int32_t>(abs_features[j]);
			phase_input[j] = static_cast<std::uint8_t>(
				std::clamp((abs_value - 64) * 2, 0, 127));
			phase_input[j + 128] = diff_features[j];
			phase_input[j + 256] = transformed_features[j];
		}
		phase_input[127] = static_cast<std::uint8_t>((bucket_id * 127) / 11);
		phase_proj.PropagatePrefix<PHASE_OUTPUT_SIZE>(phase_input, phase_output);

		return BenchmarkPhaseScalesFromOutput(phase_output);
	}

	void BenchmarkFmAffine(const TransformedFeatureType* diff_features,
		const TransformedFeatureType* abs_features, std::int32_t* diff_output,
		std::int32_t* abs_output) const {
		fc_diff.Propagate(diff_features, diff_output);
		fc_abs.Propagate(abs_features, abs_output);
	}

	void BenchmarkFcDiff(const TransformedFeatureType* input,
		std::int32_t* output) const {
		fc_diff.Propagate(input, output);
	}

	void BenchmarkFcAbs(const TransformedFeatureType* input,
		std::int32_t* output) const {
		fc_abs.Propagate(input, output);
	}

	void BenchmarkFmActivation(const std::int32_t* diff_fc_output,
		const std::int32_t* abs_fc_output, std::uint8_t* diff_output,
		std::uint8_t* abs_output, std::uint8_t* abs_sqr_output) const {
		float sum_sq_diff = 0.0f;
		for (int j = 0; j < 32; ++j) {
			const float value = static_cast<float>(diff_fc_output[j + 32]);
			sum_sq_diff += value * value;
		}
		const float inv_rms =
			1.0f / std::sqrt(sum_sq_diff / 32.0f + 1e-8f);

		for (int j = 0; j < 32; ++j) {
			const int32_t diff_value = diff_fc_output[j + 32];
			const float normalized = static_cast<float>(diff_value) * inv_rms;
			const int32_t diff_scaled =
				static_cast<int32_t>(normalized * 25.4f) + 64;
			diff_output[j] = static_cast<std::uint8_t>(
				std::max(0, std::min(127, diff_scaled)));

			const int32_t abs_gated =
				sigmoid_gate_slow(abs_fc_output[j], abs_fc_output[j + 32]);
			const float abs_value = static_cast<float>(abs_gated) / 8128.0f;
			const int32_t abs_scaled = static_cast<int32_t>(std::round(
				std::clamp(abs_value * 0.05f + 0.6f, 0.0f, 1.0f) * 127.0f));
			abs_output[j] = static_cast<std::uint8_t>(abs_scaled);

		}
		ComputeAbsSquared(abs_output, abs_sqr_output);
	}

	void BenchmarkDiffRmsNorm(const std::int32_t* diff_fc_output,
		float* sum_sq, float* inv_rms) const {
		*sum_sq = 0.0f;
		for (int j = 0; j < 32; ++j) {
			const float value = static_cast<float>(diff_fc_output[j + 32]);
			*sum_sq += value * value;
		}
		*inv_rms = 1.0f / std::sqrt(*sum_sq / 32.0f + 1e-8f);
	}

	void BenchmarkDiffQuantize(const std::int32_t* diff_fc_output,
		const float inv_rms, std::uint8_t* diff_output) const {
		for (int j = 0; j < 32; ++j) {
			const int32_t diff_value = diff_fc_output[j + 32];
			const float normalized = static_cast<float>(diff_value) * inv_rms;
			const int32_t diff_scaled =
				static_cast<int32_t>(normalized * 25.4f) + 64;
			diff_output[j] = static_cast<std::uint8_t>(
				std::max(0, std::min(127, diff_scaled)));
		}
	}

	void BenchmarkAbsSigmoidGate(const std::int32_t* abs_fc_output,
		std::int32_t* abs_gated_output) const {
		for (int j = 0; j < 32; ++j)
			abs_gated_output[j] =
				sigmoid_gate_slow(abs_fc_output[j], abs_fc_output[j + 32]);
	}

	void BenchmarkAbsGateQuantize(const std::int32_t* abs_gated_input,
		std::uint8_t* abs_output) const {
		for (int j = 0; j < 32; ++j) {
			const float abs_value =
				static_cast<float>(abs_gated_input[j]) / 8128.0f;
			const int32_t abs_scaled = static_cast<int32_t>(std::round(
				std::clamp(abs_value * 0.05f + 0.6f, 0.0f, 1.0f) * 127.0f));
			abs_output[j] = static_cast<std::uint8_t>(abs_scaled);
		}
	}

	void BenchmarkAbsSquared(const std::uint8_t* abs_input,
		std::uint8_t* abs_sqr_output) const {
		ComputeAbsSquared(abs_input, abs_sqr_output);
	}

#if defined(USE_AVX2)
	void BenchmarkAbsSquaredScalar(const std::uint8_t* abs_input,
		std::uint8_t* abs_sqr_output) const {
		ComputeAbsSquaredScalar(abs_input, abs_sqr_output);
	}

	void BenchmarkAbsSquaredAvx2(const std::uint8_t* abs_input,
		std::uint8_t* abs_sqr_output) const {
		ComputeAbsSquared(abs_input, abs_sqr_output);
	}
#endif

	void BenchmarkMainFc0(const TransformedFeatureType* transformed_features,
		std::int32_t* fc_output) const {
		fc_0.Propagate(transformed_features, fc_output);
	}

#if defined(USE_AVX2) && !defined(USE_AVX512)
	static constexpr IndexType kBenchmarkFc0InputBlocks =
		decltype(fc_0)::kBenchmarkInputBlocks;

	IndexType BenchmarkMainFc0FindNnz(
		const TransformedFeatureType* transformed_features,
		std::uint16_t* nnz) const {
		return fc_0.BenchmarkFindNnz(transformed_features, nnz);
	}

	void BenchmarkMainFc0AccumulatePreparedNnz(
		const TransformedFeatureType* transformed_features,
		const std::uint16_t* nnz, const IndexType count,
		std::int32_t* fc_output) const {
		fc_0.BenchmarkAccumulatePreparedNnz(
			transformed_features, nnz, count, fc_output);
	}

	void BenchmarkMainFc0AccumulatePreparedNnzTwoBank(
		const TransformedFeatureType* transformed_features,
		const std::uint16_t* nnz, const IndexType count,
		std::int32_t* fc_output) const {
		fc_0.BenchmarkAccumulatePreparedNnzTwoBank(
			transformed_features, nnz, count, fc_output);
	}

	void BenchmarkMainFc0StreamingSparse(
		const TransformedFeatureType* transformed_features,
		std::int32_t* fc_output) const {
		fc_0.BenchmarkPropagateStreamingSparse(
			transformed_features, fc_output);
	}

	void BenchmarkMainFc0Dense(
		const TransformedFeatureType* transformed_features,
		std::int32_t* fc_output) const {
		fc_0.BenchmarkPropagateDense(transformed_features, fc_output);
	}
#endif

	void BenchmarkMainGate(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output) const {
		BenchmarkMainGateCombined(fc_input, diff_fc_output, fc_output);
	}

	void BenchmarkMainGateCombined(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output) const {
		for (int j = 0; j < 32; ++j) {
			const int32_t gate_q64 =
				sigmoid_gate_slow(diff_fc_output[j] - 2438, 64);
			fc_output[j] = static_cast<int32_t>(
				(fc_input[j] * (64 + gate_q64)) / 128);
			if (j < 31)
				fc_output[j] = std::clamp(fc_output[j], 0, 8128);
		}
	}

	void BenchmarkMainGateReconstructed(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output) const {
		ComputeMainGate(fc_input, diff_fc_output, fc_output);
	}

	void BenchmarkMainGateSigmoid(const std::int32_t* diff_fc_output,
		std::int32_t* gate_q64) const {
		ComputeMainGateSigmoid(diff_fc_output, gate_q64);
	}

#if defined(USE_NNUE_APPROX_SIGMOID_LUT)
	void BenchmarkMainGateFloatLutSigmoid(
		const std::int32_t* diff_fc_output, std::int32_t* gate_q64) const {
		for (int j = 0; j < 32; ++j)
			gate_q64[j] = sigmoid_gate_slow(
				diff_fc_output[j] - 2438, 64);
	}

	void BenchmarkMainGateFloatLut(const std::int32_t* fc_input,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output) const {
		std::int32_t gate_q64[32];
		std::int32_t before_clamp[32];
		BenchmarkMainGateFloatLutSigmoid(diff_fc_output, gate_q64);
		ComputeMainGateApply(fc_input, gate_q64, before_clamp);
		ComputeMainGateClamp(before_clamp, fc_output);
	}
#endif

	void BenchmarkMainGateApply(const std::int32_t* fc_input,
		const std::int32_t* gate_q64, std::int32_t* fc_output) const {
		ComputeMainGateApply(fc_input, gate_q64, fc_output);
	}

	void BenchmarkMainGateClamp(const std::int32_t* fc_input,
		std::int32_t* fc_output) const {
		ComputeMainGateClamp(fc_input, fc_output);
	}

	void BenchmarkMainSqrClippedRelu(const std::int32_t* fc_output,
		std::uint8_t* sqr_output) const {
		ac_sqr_0.Propagate(fc_output, sqr_output);
	}

	void BenchmarkMainClippedRelu(const std::int32_t* fc_output,
		std::uint8_t* raw_output) const {
		ac_0.Propagate(fc_output, raw_output);
	}

	void BenchmarkMain(const TransformedFeatureType* transformed_features,
		const std::int32_t* diff_fc_output, std::int32_t* fc_output,
		std::uint8_t* sqr_output, std::uint8_t* raw_output) const {
		BenchmarkMainFc0(transformed_features, fc_output);
		BenchmarkMainGate(fc_output, diff_fc_output, fc_output);
		BenchmarkMainSqrClippedRelu(fc_output, sqr_output);
		BenchmarkMainClippedRelu(fc_output, raw_output);
	}

	void BenchmarkLca(const std::uint8_t* main_raw,
		const std::uint8_t* diff_input, const std::uint8_t* abs_input,
		std::uint8_t* diff_output, std::uint8_t* fm_input,
		std::int32_t* query_output, std::int32_t* key_output,
		std::int32_t* value_output) const {
		for (int j = 0; j < 32; ++j) {
			fm_input[j] = diff_input[j];
			fm_input[j + 32] = abs_input[j];
		}
		lca_q.Propagate(main_raw, query_output);
		lca_k.Propagate(fm_input, key_output);
		lca_v.Propagate(fm_input, value_output);

		float dot_product = 0.0f;
		for (int j = 0; j < 32; ++j)
			dot_product += (static_cast<float>(query_output[j]) / 8128.0f)
				* (static_cast<float>(key_output[j]) / 8128.0f);
		const float attention_logit = (dot_product * 0.17677f) / lca_temp;
		const float attention_score =
			1.0f / (1.0f + std::exp(-attention_logit));

		for (int j = 0; j < 32; ++j) {
			const float current_diff = static_cast<float>(diff_input[j]) / 127.0f;
			const float value = static_cast<float>(value_output[j]) / 8128.0f;
			const float clamped_value =
				std::max(0.0f, std::min(1.0f, value * 0.4f + 0.5f));
			const float final_diff = current_diff * (1.0f - attention_score)
				+ clamped_value * attention_score;
			diff_output[j] = static_cast<std::uint8_t>(final_diff * 127.0f);
		}
	}

	void BenchmarkLcaAssembleFmInput(const std::uint8_t* diff_input,
		const std::uint8_t* abs_input, std::uint8_t* fm_input) const {
		for (int j = 0; j < 32; ++j) {
			fm_input[j] = diff_input[j];
			fm_input[j + 32] = abs_input[j];
		}
	}

	void BenchmarkLcaQuery(const std::uint8_t* main_raw,
		std::int32_t* query_output) const {
		lca_q.Propagate(main_raw, query_output);
	}

	void BenchmarkLcaKey(const std::uint8_t* fm_input,
		std::int32_t* key_output) const {
		lca_k.Propagate(fm_input, key_output);
	}

	void BenchmarkLcaValue(const std::uint8_t* fm_input,
		std::int32_t* value_output) const {
		lca_v.Propagate(fm_input, value_output);
	}

	void BenchmarkLcaDotAndLogit(const std::int32_t* query_output,
		const std::int32_t* key_output, float* dot_product,
		float* attention_logit) const {
		*dot_product = 0.0f;
		for (int j = 0; j < 32; ++j)
			*dot_product +=
				(static_cast<float>(query_output[j]) / 8128.0f)
				* (static_cast<float>(key_output[j]) / 8128.0f);
		*attention_logit = (*dot_product * 0.17677f) / lca_temp;
	}

	void BenchmarkLcaAttentionScore(const float attention_logit,
		float* attention_score) const {
		*attention_score =
			1.0f / (1.0f + std::exp(-attention_logit));
	}

	void BenchmarkLcaValueClampAndCorrection(
		const std::int32_t* value_output, const float attention_score,
		float* value_clamped, float* value_correction) const {
		for (int j = 0; j < 32; ++j) {
			const float value =
				static_cast<float>(value_output[j]) / 8128.0f;
			value_clamped[j] =
				std::max(0.0f, std::min(1.0f, value * 0.4f + 0.5f));
			value_correction[j] = value_clamped[j] * attention_score;
		}
	}

	void BenchmarkLcaFinalAddAndQuantize(
		const std::uint8_t* diff_input, const float attention_score,
		const float* value_correction, float* output_before_narrow,
		std::uint8_t* diff_output) const {
		for (int j = 0; j < 32; ++j) {
			const float current_diff =
				static_cast<float>(diff_input[j]) / 127.0f;
			const float final_diff =
				current_diff * (1.0f - attention_score)
				+ value_correction[j];
			output_before_narrow[j] = final_diff * 127.0f;
			diff_output[j] =
				static_cast<std::uint8_t>(output_before_narrow[j]);
		}
	}

	void BenchmarkCross(const std::uint8_t* main_sqr,
		const std::uint8_t* main_raw, const std::uint8_t* diff_input,
		const std::uint8_t* abs_input, std::uint8_t* cross_input,
		std::int32_t* cross_fc_output, std::uint8_t* cross_output) const {
		for (int j = 0; j < 16; ++j) {
			cross_input[j] = static_cast<std::uint8_t>(
				(main_sqr[j] * diff_input[j]) / 127);
			cross_input[j + 16] = static_cast<std::uint8_t>(
				(main_raw[j] * abs_input[j]) / 127);
		}
		fc_cross.PropagatePrefix<CROSS_OUTPUT_SIZE>(cross_input, cross_fc_output);
		PropagateCrossActivation(cross_fc_output, cross_output);
	}

	void BenchmarkL2Assembly(const std::uint8_t* main_sqr,
		const std::uint8_t* main_raw, const std::uint8_t* diff_input,
		const std::uint8_t* abs_input, const std::uint8_t* abs_sqr,
		const std::uint8_t* cross_input, const BenchmarkPhaseScales& scales,
		std::uint8_t* output) const {
		AssembleL2Channel<31>(main_sqr, output, scales.main_sqr);
		AssembleL2Channel<31>(main_raw, output + 31, scales.main_raw);
		AssembleL2Channel<32>(diff_input, output + 62, scales.diff);
		AssembleL2Channel<32>(abs_input, output + 94, scales.abs_raw);
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
		AssembleL2Channel<32>(abs_sqr, output + 126, scales.abs_sqr);
#else
		(void)abs_sqr;
#endif
		AssembleL2Channel<CROSS_OUTPUT_SIZE>(cross_input, output + L2_CROSS_OFFSET, scales.cross);
		std::memset(output + L2_REAL_SIZE, 0, L2_PADDING_SIZE);
	}

	template<IndexType Dimensions>
	static void BenchmarkAssembleL2ChannelQ23(
		const std::uint8_t* input, std::uint8_t* output,
		const std::int32_t scale_q23) {
		AssembleL2ChannelQ23<Dimensions>(input, output, scale_q23);
	}

	void BenchmarkL2AssemblyQ23(const std::uint8_t* main_sqr,
		const std::uint8_t* main_raw, const std::uint8_t* diff_input,
		const std::uint8_t* abs_input, const std::uint8_t* abs_sqr,
		const std::uint8_t* cross_input, const std::int32_t* scales_q23,
		std::uint8_t* output) const {
		BenchmarkAssembleL2ChannelQ23<31>(main_sqr, output, scales_q23[0]);
		BenchmarkAssembleL2ChannelQ23<31>(main_raw, output + 31, scales_q23[1]);
		BenchmarkAssembleL2ChannelQ23<32>(diff_input, output + 62, scales_q23[2]);
		BenchmarkAssembleL2ChannelQ23<32>(abs_input, output + 94, scales_q23[3]);
#if !defined(USE_NNUE_ABS_SQR_REMOVED_160)
		BenchmarkAssembleL2ChannelQ23<32>(abs_sqr, output + 126, scales_q23[4]);
#else
		(void)abs_sqr;
#endif
		BenchmarkAssembleL2ChannelQ23<CROSS_OUTPUT_SIZE>(cross_input, output + L2_CROSS_OFFSET,
			scales_q23[PHASE_CROSS_INDEX]);
		std::memset(output + L2_REAL_SIZE, 0, L2_PADDING_SIZE);
	}

	void BenchmarkFc1Activation(const std::uint8_t* input,
		std::int32_t* fc_output, std::uint8_t* activation_output) const {
		fc_1.Propagate(input, fc_output);
		ac_1.Propagate(fc_output, activation_output);
	}

	void BenchmarkFc1(const std::uint8_t* input,
		std::int32_t* output) const {
		fc_1.Propagate(input, output);
	}

#if defined(USE_AVX2) && !defined(USE_AVX512)
	void BenchmarkFc1OutputTiled(const std::uint8_t* input,
		std::int32_t* output) const {
#if defined(USE_NNUE_ABS_SQR_REMOVED_160)
		// The historical output-tiling experiment is specialized for 192 inputs.
		// Keep benchmark builds source-compatible; compact builds use the normal
		// 160-input kernel here rather than instantiating that retired candidate.
		fc_1.Propagate(input, output);
#else
		fc_1.BenchmarkPropagateOutputTiled64And32(input, output);
#endif
	}
#endif

	void BenchmarkAc1(const std::int32_t* input,
		std::uint8_t* output) const {
		ac_1.Propagate(input, output);
	}

	void BenchmarkFc2(const std::uint8_t* input,
		std::int32_t* output) const {
		fc_2.Propagate(input, output);
	}

	std::int32_t BenchmarkBlend(const std::int32_t bypass_input,
		const std::int32_t deep_output) const {
		const int32_t bypass_output =
			(bypass_input * (600 * 16)) / (127 * 64);
		const int64_t combined =
			static_cast<int64_t>(deep_output) * bucket_blend_alpha
			+ static_cast<int64_t>(bypass_output)
				* (16384 - bucket_blend_alpha);
		return static_cast<std::int32_t>(combined / 16384);
	}
#endif
};

}  // namespace Eval::NNUE
}  // namespace YaneuraOu

#endif // CLASSIC_NNUE_HALFKA_KSDG3_FM_1280_H_INCLUDED
