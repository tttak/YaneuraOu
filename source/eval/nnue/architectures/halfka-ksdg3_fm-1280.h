#ifndef CLASSIC_NNUE_HALFKA_KSDG3_FM_1280_H_INCLUDED
#define CLASSIC_NNUE_HALFKA_KSDG3_FM_1280_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka.h"
#include "../features/king_safety3_distinguishgolds.h"

#include <cstring>

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
constexpr IndexType L2_INPUT_SIZE = 192;
constexpr IndexType kHidden2Dims = 96;

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
	Layers::AffineTransformExplicit<32, 32> fc_cross;
	Layers::ClippedReLUExplicit<32> ac_cross;

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
		return 0x6333718Au;
	}

	static std::string GetStructureString() {
		return "HalfKA-KSDG3_FM-1280";
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
		return Tools::ResultCode::Ok;
	}

	bool WriteParameters(std::ostream& stream) const {
		return true;
	}

	static inline int32_t sigmoid_gate_slow(int32_t x, int32_t value) {
		float sig = 1.0f / (1.0f + std::exp(-static_cast<float>(x) / 8128.0f));
		return static_cast<int32_t>(value * sig);
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

#if defined(ENABLE_NNUE_BENCH)
	template<bool UsePhasePrefix = true>
#endif
	const OutputType* Propagate(const TransformedFeatureType* transformedFeatures, const TransformedFeatureType* diffFeatures, const TransformedFeatureType* absFeatures, const int bucket_id, char* buffer) const {
		auto& buf = *reinterpret_cast<Buffer*>(buffer);

		// --- 1. Phase Gate: 局面の進行度や激しさに応じた動的スケーリング ---
		// abs, diff, main の各特徴量を統合し、現在の局面状態（Voltage）を判定します。
		for (int j = 0; j < 128; ++j) {
			int32_t abs_val = static_cast<int32_t>(absFeatures[j]);
			buf.phase_input[j] = static_cast<std::uint8_t>(std::clamp((abs_val - 64) * 2, 0, 127));
			buf.phase_input[j + 128] = diffFeatures[j];
			buf.phase_input[j + 256] = transformedFeatures[j]; 
		}
		// 入力のabs部分の末尾を「バケットID（0～11）/11」で置き換える
		buf.phase_input[127] = static_cast<std::uint8_t>((bucket_id * 127) / 11);

		// Phase Gate 推論と各パスへの係数算出 (0.1 ～ 1.0 の範囲に正規化)
#if defined(ENABLE_NNUE_BENCH)
		if constexpr (UsePhasePrefix)
			phase_proj.PropagatePrefix<6>(buf.phase_input, buf.phase_out);
		else
			phase_proj.Propagate(buf.phase_input, buf.phase_out);
#else
		phase_proj.PropagatePrefix<6>(buf.phase_input, buf.phase_out);
#endif
		float phase_val[6];
		for (int i = 0; i < 6; ++i) {
			float logit = (static_cast<float>(buf.phase_out[i]) / 8128.0f) * 3.0f + 1.0f;
			float sig = 1.0f / (1.0f + std::exp(-logit));
			phase_val[i] = 0.1f + 0.9f * sig;
		}

		// L2入力時の各チャネルごとの最終スケールを決定
		float main_sqr_scale = (0.5f + 0.5f * phase_val[0]) * 1.3f;
		float main_raw_scale = (0.5f + 0.5f * phase_val[1]) * 1.5f;
		float diff_scale     = (0.5f + 0.5f * phase_val[2]) * 1.0f;
		float abs_raw_scale  = (0.5f + 0.5f * phase_val[3]) * 0.7f;
		float abs_sqr_scale  = (0.5f + 0.5f * phase_val[4]) * 0.88f;
		float cross_scale    = (0.5f + 0.5f * phase_val[5]) * 1.5f;


		// --- 2. FM Path: Factorization Machines 的な相互作用抽出 ---
		fc_diff.Propagate(diffFeatures, buf.diff_fc_out);
		fc_abs.Propagate(absFeatures, buf.abs_fc_out);

		// Diff Path: RMSNorm を適用して信号の分散を安定化
		float sum_sq_d = 0.0f;
		for (int j = 0; j < 32; ++j) {
			float vd_f = static_cast<float>(buf.diff_fc_out[j + 32]); // val_d
			sum_sq_d += vd_f * vd_f;
		}
		float inv_rms_d = 1.0f / std::sqrt(sum_sq_d / 32.0f + 1e-8f);

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

		}
		// Abs Sqr Path: 二乗による非線形強調
		ComputeAbsSquared(buf.abs_ac_out, buf.abs_sqr_out);


		// --- 3. Main Path: 基本骨格パスと FM による動的フィルタリング ---
		fc_0.Propagate(transformedFeatures, buf.fc_0_out);
		for (int j = 0; j < 32; ++j) {
			// FM 側の信号（gate_d）で Main パスの情報の通りやすさを制御
			int32_t sig_half = sigmoid_gate_slow(buf.diff_fc_out[j] - 2438, 64);
			buf.fc_0_out[j] = static_cast<int32_t>((buf.fc_0_out[j] * (64 + sig_half)) / 128);

	    	if (j < 31) {
				buf.fc_0_out[j] = std::clamp(buf.fc_0_out[j], 0, 8128);
			}
		}

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
		for (int j = 0; j < 32; ++j) {
			float current_diff = static_cast<float>(buf.diff_ac_out[j]) / 127.0f;
			float v_val = static_cast<float>(buf.lca_v_out[j]) / 8128.0f;
			float v_clamped = std::max(0.0f, std::min(1.0f, v_val * 0.4f + 0.5f));
			float final_diff_f = current_diff * (1.0f - att_score) + v_clamped * att_score;
			buf.diff_ac_out[j] = static_cast<uint8_t>(final_diff_f * 127.0f);
		}


		// --- 5. Cross Feature: 異種パス間の積による相関特徴の生成 ---
		for (int j = 0; j < 16; ++j) {
			buf.cross_cat[j]      = (uint8_t)((buf.ac_sqr_0_out_temp[j] * buf.diff_ac_out[j]) / 127);
			buf.cross_cat[j + 16] = (uint8_t)((buf.ac_0_out[j] * buf.abs_ac_out[j]) / 127);
		}

		fc_cross.Propagate(buf.cross_cat, buf.cross_fc_out);
		ac_cross.Propagate(buf.cross_fc_out, buf.cross_feat);


		// --- 6. L2 Input Assembly: 深層評価パスへの入力構築 (192次元) ---
		// 各チャネルを Phase Gate で得たスケールで調整しつつ統合
		// [0:30] MainSqr, [31:61] MainRaw, [62:93] Diff, [94:125] AbsRaw, [126:157] AbsSqr, [158:189] Cross, [190:191] Pad
		AssembleL2Channel<31>(buf.ac_sqr_0_out_temp, &buf.l2_input[0], main_sqr_scale);
		AssembleL2Channel<31>(buf.ac_0_out, &buf.l2_input[31], main_raw_scale);
		AssembleL2Channel<32>(buf.diff_ac_out, &buf.l2_input[62], diff_scale);
		AssembleL2Channel<32>(buf.abs_ac_out, &buf.l2_input[94], abs_raw_scale);
		AssembleL2Channel<32>(buf.abs_sqr_out, &buf.l2_input[126], abs_sqr_scale);
		AssembleL2Channel<32>(buf.cross_feat, &buf.l2_input[158], cross_scale);
		std::memset(buf.l2_input + 190, 0, 2);


		// --- 7. Deep Path 推論 ---
		fc_1.Propagate(buf.l2_input, buf.fc_1_out);
		ac_1.Propagate(buf.fc_1_out, buf.ac_1_out);
		fc_2.Propagate(buf.ac_1_out, buf.fc_2_out);


		// --- 8. Final Blending ---
		const int32_t alpha = bucket_blend_alpha;
		const int32_t inv_alpha = 16384 - alpha;

		// Main パスの 31 番目の要素を Bypass Path（直接出力）として利用
		int32_t fwdOut_main = (int(buf.fc_0_out[31]) * (600 * 16)) / (127 * 64);

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
	struct BenchmarkPhaseScales {
		float main_sqr;
		float main_raw;
		float diff;
		float abs_raw;
		float abs_sqr;
		float cross;
	};

	BenchmarkPhaseScales BenchmarkPhaseScalesFromOutput(
		const std::int32_t* phase_output) const {
		float phase_value[6];
		for (int i = 0; i < 6; ++i) {
			const float logit =
				(static_cast<float>(phase_output[i]) / 8128.0f) * 3.0f + 1.0f;
			const float sigmoid = 1.0f / (1.0f + std::exp(-logit));
			phase_value[i] = 0.1f + 0.9f * sigmoid;
		}

		return {
			(0.5f + 0.5f * phase_value[0]) * 1.3f,
			(0.5f + 0.5f * phase_value[1]) * 1.5f,
			(0.5f + 0.5f * phase_value[2]) * 1.0f,
			(0.5f + 0.5f * phase_value[3]) * 0.7f,
			(0.5f + 0.5f * phase_value[4]) * 0.88f,
			(0.5f + 0.5f * phase_value[5]) * 1.5f};
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
		phase_proj.PropagatePrefix<6>(phase_input, phase_output);

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
		for (int j = 0; j < 32; ++j) {
			const int32_t sigmoid_half =
				sigmoid_gate_slow(diff_fc_output[j] - 2438, 64);
			fc_output[j] = static_cast<int32_t>(
				(fc_input[j] * (64 + sigmoid_half)) / 128);
			if (j < 31)
				fc_output[j] = std::clamp(fc_output[j], 0, 8128);
		}
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
		fc_cross.Propagate(cross_input, cross_fc_output);
		ac_cross.Propagate(cross_fc_output, cross_output);
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
		AssembleL2Channel<32>(abs_sqr, output + 126, scales.abs_sqr);
		AssembleL2Channel<32>(cross_input, output + 158, scales.cross);
		std::memset(output + 190, 0, 2);
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
		fc_1.BenchmarkPropagateOutputTiled64And32(input, output);
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
