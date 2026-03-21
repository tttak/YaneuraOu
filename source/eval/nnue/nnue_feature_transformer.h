// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#if defined(SFNNwoPSQT)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <algorithm>  // std::clamp
#include <cstring>  // std::memset()

namespace YaneuraOu {
namespace Eval::NNUE {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// ベクトル命令が有効な場合、変数のタイルを、
// 各タイルがCPUのベクトルレジスタに収まるように、更新してリフレッシュする。
#define VECTOR

#if defined(USE_AVX512)
using vec_t = __m512i;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
#define vec_set_16(a) _mm512_set1_epi16(a)
#define vec_max_16(a, b) _mm512_max_epi16(a, b)
#define vec_min_16(a, b) _mm512_min_epi16(a, b)
#define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
#define vec_zero() _mm512_setzero_si512()
static constexpr IndexType kNumRegs = 8;  // only 8 are needed

#elif defined(USE_AVX2)
using vec_t = __m256i;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
#define vec_set_16(a) _mm256_set1_epi16(a)
#define vec_max_16(a, b) _mm256_max_epi16(a, b)
#define vec_min_16(a, b) _mm256_min_epi16(a, b)
#define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
#define vec_zero() _mm256_setzero_si256()
static constexpr IndexType kNumRegs = 16;

#elif defined(USE_SSE2)
using vec_t = __m128i;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_epi16(a, b)
#define vec_sub_16(a, b) _mm_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
#define vec_set_16(a) _mm_set1_epi16(a)
#define vec_max_16(a, b) _mm_max_epi16(a, b)
#define vec_min_16(a, b) _mm_min_epi16(a, b)
#define vec_slli_16(a, b) _mm_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm_packus_epi16(a, b)
#define vec_zero() _mm_setzero_si128()
static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

#elif defined(USE_MMX)
using vec_t = __m64;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero() _mm_setzero_si64()
static constexpr IndexType kNumRegs = 8;

#elif defined(USE_NEON)
using vec_t = int16x8_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) vaddq_s16(a, b)
#define vec_sub_16(a, b) vsubq_s16(a, b)
#define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
#define vec_set_16(a) vdupq_n_s16(a)
#define vec_max_16(a, b) vmaxq_s16(a, b)
#define vec_min_16(a, b) vminq_s16(a, b)
#define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
#define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
#define vec_zero() \
	vec_t { 0 }
static constexpr IndexType kNumRegs = 16;

#else
#undef VECTOR

#endif

constexpr IndexType MaxChunkSize = 16;

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

#if defined(VECTOR)
	//static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
	//static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
	// ⇨  AVX-512でこの制約守れないっぽ。
#endif

   public:
	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;
	using BiasType   = std::int16_t;
	using WeightType = std::int16_t;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions  = RawFeatures::kDimensions;
#if defined(USE_ELEMENT_WISE_MULTIPLY)
	static constexpr IndexType kOutputDimensions = kHalfDimensions;
#else
	static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;
#endif

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr std::uint32_t GetHashValue() {
#if defined(SFNNwoPSQT)
		// 学習部と整合性とるの面倒なのでSFNNwoPSQTのときはこれに固定しておく。
		return 0x5f134ab8u;
#else
		return RawFeatures::kHashValue ^ kOutputDimensions;
#endif
	}

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(kInputDimensions) + "->"
		       + std::to_string(kHalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
//#if defined(USE_ELEMENT_WISE_MULTIPLY)
#if 0
		read_leb_128<BiasType>(stream, biases_, kHalfDimensions);
		read_leb_128<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);

#if defined(VECTOR)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#else
		std::cout << "非圧縮版から読み込み" << std::endl;
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);

		// --- v_weightsの読み込み（FM項用） ---
		for (std::size_t i = 0; i < kFactorDimensions * kInputDimensions; ++i)
			v_weights_[i] = read_little_endian<WeightType>(stream);

#endif

		// --- pair_weights_ の読み込み ---
		for (std::size_t p = 0; p < kPhaseBuckets; ++p) {
			// Mul (640次元)
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i) 
				pair_weights_mul[p][i] = read_little_endian<int16_t>(stream);

			// Diff (640次元)
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i) 
				pair_weights_diff[p][i] = read_little_endian<int16_t>(stream);

			// Sum (640次元)
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i) 
				pair_weights_sum[p][i] = read_little_endian<int16_t>(stream);
		}

		// --- 全12項目の統計を表示 ---
		const char* phase_names[] = {"OPEN", "MID1", "MID2", "END "};
		
		auto print_stats = [&](const char* p_name, const char* t_name, int16_t* arr) {
			int16_t min_v = 32767, max_v = -32768;
			double sum_v = 0;
			for (int i = 0; i < kPairWeightDimensions; ++i) {
				min_v = std::min(min_v, arr[i]);
				max_v = std::max(max_v, arr[i]);
				sum_v += arr[i];
			}

			// 16384.0 で割って 0.0 ～ 1.0 の比率として表示
			std::cout << "[" << p_name << "] " << t_name << " | "
					  << std::fixed << std::setprecision(4) 
					  << "Avg: " << (sum_v / kPairWeightDimensions / 16384.0) << " | "
					  << std::setprecision(3)
					  << "Range: [" << (min_v / 16384.0) << " ~ " << (max_v / 16384.0) << "]" 
					  << std::endl;
		};

		std::cout << "------------------------------------------------------------" << std::endl;
		for (int p = 0; p < kPhaseBuckets; ++p) {
			print_stats(phase_names[p], "MUL ", pair_weights_mul[p]);
			print_stats(phase_names[p], "DIFF", pair_weights_diff[p]);
			print_stats(phase_names[p], "SUM ", pair_weights_sum[p]);
			std::cout << "------------------------------------------------------------" << std::endl;
		}

		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos) const {
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
			return true;
		}
		const auto prev = now->previous;
		if (prev && prev->accumulator.computed_accumulation) {
			update_accumulator(pos);
			return true;
		}
		return false;
	}

	// 中立点へのマッピング
	OutputType ToOutputRange(int64_t shifted_value) const {
		return static_cast<OutputType>(std::clamp(shifted_value + 63, 0LL, 127LL));
	}

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, OutputType* diff_output, OutputType* abs_output, bool refresh, const int bucket_id) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;
		const auto& factors      = pos.state()->accumulator.factors; // FM項用

		// --- 1. Phase-based Weight Blending: 局面（序終盤）に応じた重みの線形補間 ---
		// 進行度(bucket_id)に基づき、4つのフェーズ（OPEN/MID1/MID2/END）の重みをブレンドします。

		// 進行度 [0, 11] を [0.0, 3.0] にスケールし、隣接する2つのフェーズ係数を算出
		float pf = static_cast<float>(bucket_id) / 11.0f;
		float p3 = pf * 3.0f;

		float fw[4] = {
			std::max(0.0f, 1.0f - p3),
			std::max(0.0f, 1.0f - std::abs(p3 - 1.0f)),
			std::max(0.0f, 1.0f - std::abs(p3 - 2.0f)),
			std::max(0.0f, p3 - 2.0f)
		};

		// 固定小数点(Q14)に変換して整数演算で高速にブレンド
		int32_t iw[4];
		iw[0] = static_cast<int32_t>(fw[0] * 16384.0f);
		iw[1] = static_cast<int32_t>(fw[1] * 16384.0f);
		iw[2] = static_cast<int32_t>(fw[2] * 16384.0f);
		iw[3] = 16384 - (iw[0] + iw[1] + iw[2]);

		// ブレンド済み重みバッファ
		alignas(kCacheLineSize) int16_t curr_w_mul[kPairWeightDimensions];
		alignas(kCacheLineSize) int16_t curr_w_diff[kPairWeightDimensions];
		alignas(kCacheLineSize) int16_t curr_w_sum[kPairWeightDimensions];

		for (IndexType j = 0; j < kPairWeightDimensions; ++j) {
			curr_w_mul[j]  = (pair_weights_mul[0][j] * iw[0] + pair_weights_mul[1][j] * iw[1] + 
							  pair_weights_mul[2][j] * iw[2] + pair_weights_mul[3][j] * iw[3]) >> 14;
			curr_w_diff[j] = (pair_weights_diff[0][j] * iw[0] + pair_weights_diff[1][j] * iw[1] + 
							  pair_weights_diff[2][j] * iw[2] + pair_weights_diff[3][j] * iw[3]) >> 14;
			curr_w_sum[j]  = (pair_weights_sum[0][j] * iw[0] + pair_weights_sum[1][j] * iw[1] + 
							  pair_weights_sum[2][j] * iw[2] + pair_weights_sum[3][j] * iw[3]) >> 14;
		}

		// --- 2. Main Transformation: Accumulatorからの特徴抽出 ---
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p)
		{
			const IndexType offset = (kHalfDimensions / 2) * p;
			const Color side = perspectives[p];

#if defined(VECTOR)
			// SIMDによる高速変換: a, bの2つのハーフ入力を3つの方法(Mul, Diff, Sum)で結合
			constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / 32;
			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[side][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[side][0][kHalfDimensions / 2]));

			// (a*b*w_mul) + (a-b)^2*w_diff + (a+b)*64*w_sum をSIMDで並列計算
			auto blend_vec_3way = [&](vec_t a, vec_t b, IndexType j) {
				const __m256i V_Zero = _mm256_setzero_si256();
				const __m256i V_127  = _mm256_set1_epi16(127);
				const __m256i V_64   = _mm256_set1_epi32(64);

				a = _mm256_max_epi16(_mm256_min_epi16(a, V_127), V_Zero);
				b = _mm256_max_epi16(_mm256_min_epi16(b, V_127), V_Zero);

				auto compute8 = [&](__m128i a16, __m128i b16, IndexType idx) {
					__m256i a32 = _mm256_cvtepi16_epi32(a16);
					__m256i b32 = _mm256_cvtepi16_epi32(b16);

					__m256i w_mul  = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&curr_w_mul[idx]));
					__m256i w_diff = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&curr_w_diff[idx]));
					__m256i w_sum  = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)&curr_w_sum[idx]));

					__m256i term_mul  = _mm256_mullo_epi32(_mm256_mullo_epi32(a32, b32), w_mul);
					__m256i diff      = _mm256_sub_epi32(a32, b32);
					__m256i term_diff = _mm256_mullo_epi32(_mm256_mullo_epi32(diff, diff), w_diff);
					__m256i term_sum  = _mm256_mullo_epi32(_mm256_mullo_epi32(_mm256_add_epi32(a32, b32), w_sum), V_64);

					return _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(term_mul, term_diff), term_sum), 21);
				};

				__m256i res_lo = compute8(_mm256_extracti128_si256(a, 0), _mm256_extracti128_si256(b, 0), j);
				__m256i res_hi = compute8(_mm256_extracti128_si256(a, 1), _mm256_extracti128_si256(b, 1), j + 8);
				__m256i packed16 = _mm256_packs_epi32(res_lo, res_hi);
				return _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
			};

			for (IndexType j = 0; j < NumOutputChunks; j++) {
				// 256bitレジスタ2つ分(32要素)をまとめて処理
				vec_t blended0 = blend_vec_3way(in0[j * 2 + 0], in1[j * 2 + 0], j * 32);
				vec_t blended1 = blend_vec_3way(in0[j * 2 + 1], in1[j * 2 + 1], j * 32 + 16);
				// 結果を 8bit にパッキングして出力バッファへ
				__m256i packed8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(blended0, blended1), _MM_SHUFFLE(3, 1, 2, 0));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[offset + j * 32]), packed8);
			}

#else // #if defined(VECTOR)

			// 非SIMD用
			for (IndexType j = 0; j < kHalfDimensions / 2; ++j) {
				// 1. 入力を 0-127 にクランプ
				int32_t a = std::clamp<int32_t>(accumulation[side][0][j], 0, 127);
				int32_t b = std::clamp<int32_t>(accumulation[side][0][j + kHalfDimensions / 2], 0, 127);

				// 合成済みバッファ(curr_w_...)を参照
				int32_t w_mul  = curr_w_mul[j];
				int32_t w_diff = curr_w_diff[j];
				int32_t w_sum  = curr_w_sum[j];

				// 2. ブレンド計算
				int32_t blended_total = (a * b * w_mul) + 
										((a - b) * (a - b) * w_diff) + 
										((a + b) * 64 * w_sum);

				// 3. スケーリング (>> 21)
				// 127^2 * 16384 / 2^21 ≒ 126.0
				output[offset + j] = static_cast<OutputType>(blended_total >> 21);
			}

#endif // #if defined(VECTOR)

		} // for (IndexType p = 0; p < 2; ++p)


		// --- 3. FM Path calculation: 2次相互作用（Factorization Machines）の抽出 ---
		// Accumulator更新時に計算済みの Σv と Σv^2 を用いて、
		// 内積和 ΣΣ<vi, vj>xi xj = ( (Σvx)^2 - Σ(vx)^2 ) / 2  を計算します。
		const Color us = pos.side_to_move();
		const Color them = ~us;
		
		for (IndexType j = 0; j < 32; ++j)
		{
			// 2次相互作用項の公式: (Σv)^2 - Σ(v^2)
			auto get_inter = [](int64_t v, int64_t v2) { return (v * v - v2) / 2; };

			// 各特徴量（HalfKA, KSDG）の1次和と2次項を個別に抽出
			// ih = HalfKA 2次項
			// ik = KSDG3  2次項
			// sh = HalfKA 1次項
			// sk = KSDG3  1次項

			int64_t ih_u = get_inter(factors[us].halfka.sum_v[j], factors[us].halfka.sum_v2[j]);
			int64_t ik_u = get_inter(factors[us].ksdg.sum_v[j], factors[us].ksdg.sum_v2[j]);
			int64_t sh_u = factors[us].halfka.sum_v[j];
			int64_t sk_u = factors[us].ksdg.sum_v[j];

			int64_t ih_t = get_inter(factors[them].halfka.sum_v[j], factors[them].halfka.sum_v2[j]);
			int64_t ik_t = get_inter(factors[them].ksdg.sum_v[j], factors[them].ksdg.sum_v2[j]);
			int64_t sh_t = factors[them].halfka.sum_v[j];
			int64_t sk_t = factors[them].ksdg.sum_v[j];

			// 差分(Diff)と合計(Abs)の両方のパスを生成し、後段のネットワークに渡す
			// スケーリングとビットシフトにより 0-127 の範囲へ収める

			// --- Diff Path (128次元) ---
			diff_output[ 0 + j] = ToOutputRange(((ih_u - ih_t) * FMScale1) >> 37); // ih(2次)差分
			diff_output[32 + j] = ToOutputRange(((ik_u - ik_t) * FMScale2) >> 37); // ik(2次)差分
			diff_output[64 + j] = ToOutputRange(((sh_u - sh_t) * FMScale3) >> 22); // sh(1次)差分
			diff_output[96 + j] = ToOutputRange(((sk_u - sk_t) * FMScale4) >> 22); // sk(1次)差分

			// --- Abs Path (128次元) ---
			abs_output [ 0 + j] = ToOutputRange(((ih_u + ih_t) * FMScale5) >> 37); // ih(2次)和
			abs_output [32 + j] = ToOutputRange(((ik_u + ik_t) * FMScale6) >> 37); // ik(2次)和
			abs_output [64 + j] = ToOutputRange(((sh_u + sh_t) * FMScale7) >> 22); // sh(1次)和
			abs_output [96 + j] = ToOutputRange(((sk_u + sk_t) * FMScale8) >> 22); // sk(1次)和
		}
	}

   private:
	static void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_set_epi32 packs in the order [15 11 7 3 14 10 6 2 13 9 5 1 12 8 4 0]
		uint64_t tmp0 = v[4], tmp1 = v[5];
		v[4] = v[6], v[5] = v[7];
		v[6] = tmp0, v[7] = tmp1;
		tmp0 = v[8], tmp1 = v[9];
		v[8] = v[12], v[9] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // _mm256_set_epi32 packs in the order [7 3 6 2 5 1 4 0]
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	static void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = v[8], v[5] = v[9];
		v[8] = tmp0, v[9] = tmp1;
		tmp0 = v[6], tmp1 = v[7];
		v[6] = v[12], v[7] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases_[0]));
		for (IndexType i = 0; i < kHalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
			order_fn(&b[i]);

		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			uint64_t* w =
				reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights_[j * kHalfDimensions]));
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights_[j * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (Color perspective : {BLACK, WHITE}) {

				// --- 1. 初期化処理 ---
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
					std::memset(&accumulator.factors[perspective], 0, sizeof(accumulator.factors[perspective]));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}

				// --- 2. 駒インデックスによる加算処理 ---
				for (const auto index : active_indices[perspective]) {

					// A. NNUEメインパスの加算
#if defined(VECTOR)
					const IndexType offset = kHalfDimensions * index;
					auto accumulation      = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
					auto column            = reinterpret_cast<const vec_t*>(&weights_[offset]);
#if defined(USE_AVX512)
					constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#else
					constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
#endif
					for (IndexType j = 0; j < kNumChunks; ++j) {
						accumulation[j] = vec_add_16(accumulation[j], column[j]);
					}

#else
					const IndexType offset = kHalfDimensions * index;
					for (IndexType j = 0; j < kHalfDimensions; ++j) {
						accumulator.accumulation[perspective][i][j] += weights_[offset + j];
					}
#endif

					// B. FM項の加算 (i=0 の時のみ)
					if (i == 0) {
						const IndexType v_offset = index * kFactorDimensions;
						auto& group = (index < SPLIT_IDX) ? 
									  accumulator.factors[perspective].ksdg: 
									  accumulator.factors[perspective].halfka;
						for (IndexType j = 0; j < kFactorDimensions; ++j) {
							const std::int64_t v = v_weights_[v_offset + j];
							group.sum_v[j]  += v;
							group.sum_v2[j] += (v * v);
						}
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
#if defined(USE_AVX512)
				constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#else
				constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
#endif
				auto accumulation              = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
				if (reset[perspective]) {
					if (i == 0) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
						std::memset(&accumulator.factors[perspective], 0, sizeof(accumulator.factors[perspective]));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));

					if (i == 0) {
						std::memcpy(&accumulator.factors[perspective], &prev_accumulator.factors[perspective], sizeof(accumulator.factors[perspective]));
					}

					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif

						// FM項の減算
						if (i == 0) {
							const IndexType v_offset = index * kFactorDimensions;
							auto& group = (index < SPLIT_IDX) ? 
										  accumulator.factors[perspective].ksdg: 
										  accumulator.factors[perspective].halfka;
							for (IndexType j = 0; j < kFactorDimensions; ++j) {
								const std::int64_t v = v_weights_[v_offset + j];
								group.sum_v[j]  -= v;
								group.sum_v2[j] -= (v * v);
							}
						}
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
						}
#endif

						// FM項の加算
						if (i == 0) {
							const IndexType v_offset = index * kFactorDimensions;
							auto& group = (index < SPLIT_IDX) ? 
										  accumulator.factors[perspective].ksdg: 
										  accumulator.factors[perspective].halfka;
							for (IndexType j = 0; j < kFactorDimensions; ++j) {
								const std::int64_t v = v_weights_[v_offset + j];
								group.sum_v[j]  += v;
								group.sum_v2[j] += (v * v);
							}
						}
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// parameter type
	// パラメータの型

	// Make the learning class a friend
	// 学習用クラスをfriendにする
	friend class Trainer<FeatureTransformer>;

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kHalfDimensions * kInputDimensions];

	// FM用
	static constexpr IndexType kFactorDimensions = 32;
	alignas(kCacheLineSize) WeightType v_weights_[kFactorDimensions * kInputDimensions];
	static constexpr IndexType SPLIT_IDX = 12672; // KSDG3とHalfKAの境界

	// pair_weights
	static constexpr IndexType kPairWeightDimensions = 640;
	static constexpr IndexType kPhaseBuckets = 4;

	alignas(kCacheLineSize) int16_t pair_weights_mul [kPhaseBuckets][kPairWeightDimensions];
	alignas(kCacheLineSize) int16_t pair_weights_diff[kPhaseBuckets][kPairWeightDimensions];
	alignas(kCacheLineSize) int16_t pair_weights_sum [kPhaseBuckets][kPairWeightDimensions];

};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
