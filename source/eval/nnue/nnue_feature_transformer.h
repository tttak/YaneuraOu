// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#if defined(SFNNwoPSQT) && !defined(USE_ELEMENT_WISE_MULTIPLY)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <algorithm>  // std::clamp
#include <array>
#include <cstdint>
#include <cstring>  // std::memset()
#include <memory>

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

#if defined(ENABLE_NNUE_BENCH) && defined(USE_FINNY_TABLES)
	struct BenchmarkFinnyStatistics {
		std::uint64_t hits = 0;
		std::uint64_t misses = 0;
		std::uint64_t removed_features = 0;
		std::uint64_t added_features = 0;
	};
#endif

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
		// 先頭の4 phaseデータ [4, 3, 640] はPyTorch復元用なので読み飛ばす。
		for (std::size_t p = 0; p < kPhaseBuckets; ++p) {
			for (std::size_t t = 0; t < kPairWeightTypes; ++t) {
				for (std::size_t i = 0; i < kPairWeightDimensions; ++i)
					(void)read_little_endian<int16_t>(stream);
			}
		}

		// 続くC++推論用データ [12, 3, 640] を読み込む。
		for (std::size_t b = 0; b < kPairWeightBuckets; ++b) {
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i)
				pair_weights_mul[b][i] = read_little_endian<int16_t>(stream);
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i)
				pair_weights_diff[b][i] = read_little_endian<int16_t>(stream);
			for (std::size_t i = 0; i < kPairWeightDimensions; ++i)
				pair_weights_sum[b][i] = read_little_endian<int16_t>(stream);
		}

		// --- 12 bucket x 3項目の統計を表示 ---
		const char* bucket_names[] = {
			"B00", "B01", "B02", "B03", "B04", "B05",
			"B06", "B07", "B08", "B09", "B10", "B11"
		};
		
		auto print_stats = [&](const char* p_name, const char* t_name, int16_t* arr) {
			int16_t min_v = 32767, max_v = -32768;
			double sum_v = 0;
			for (IndexType i = 0; i < kPairWeightDimensions; ++i) {
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
		for (IndexType b = 0; b < kPairWeightBuckets; ++b) {
			print_stats(bucket_names[b], "MUL ", pair_weights_mul[b]);
			print_stats(bucket_names[b], "DIFF", pair_weights_diff[b]);
			print_stats(bucket_names[b], "SUM ", pair_weights_sum[b]);
			std::cout << "------------------------------------------------------------" << std::endl;
		}

		if (!stream.fail()) {
#if defined(USE_FINNY_TABLES)
			++finny_generation_;
#endif
			return Tools::ResultCode::Ok;
		}
		return Tools::ResultCode::FileReadError;
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

#if defined(USE_AVX2)
   private:
	// Scale four signed FM linear terms and pack them to four output bytes.
	// HalfKA has at most 40 active factors and KSDG3 at most 24. Since every
	// factor weight is int16, |sum_v| <= 40 * 32768 = 1,310,720 and
	// |sum_v(us)-sum_v(them)| <= 2,621,440. Both operands therefore fit in
	// signed 32 bits, and even the maximum supported FMScale (1,000,000,000)
	// produces a signed 64-bit product. _mm256_mul_epi32 is consequently exact.
	static inline std::uint32_t ScaleAndPackFmLinear4(
		const __m256i values, const int scale) {
		const __m256i zero = _mm256_setzero_si256();
		const __m256i scale32 = _mm256_set1_epi32(scale);
		const __m256i product = _mm256_mul_epi32(values, scale32);

		// Match the arithmetic signed >> 22 emitted for the scalar expression.
		const __m256i negative_product = _mm256_cmpgt_epi64(zero, product);
		const __m256i shifted = _mm256_or_si256(
			_mm256_srli_epi64(product, 22),
			_mm256_slli_epi64(negative_product, 64 - 22));
		const __m256i centered =
			_mm256_add_epi64(shifted, _mm256_set1_epi64x(63));

		const __m256i negative_centered =
			_mm256_cmpgt_epi64(zero, centered);
		const __m256i clipped_low =
			_mm256_andnot_si256(negative_centered, centered);
		const __m256i upper = _mm256_set1_epi64x(127);
		const __m256i above_upper =
			_mm256_cmpgt_epi64(clipped_low, upper);
		const __m256i clipped =
			_mm256_blendv_epi8(clipped_low, upper, above_upper);

		// Each qword is in [0,127]. Select its low dword, then saturating-pack
		// four dwords to the low four bytes without changing their order.
		const __m256i low_dword_indices =
			_mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
		const __m128i dwords = _mm256_castsi256_si128(
			_mm256_permutevar8x32_epi32(clipped, low_dword_indices));
		const __m128i words = _mm_packus_epi32(dwords, dwords);
		const __m128i bytes = _mm_packus_epi16(words, words);
		return static_cast<std::uint32_t>(_mm_cvtsi128_si32(bytes));
	}

	static inline void StoreFmLinear4(OutputType* output,
	                                  const __m256i values,
	                                  const int scale) {
		const std::uint32_t packed = ScaleAndPackFmLinear4(values, scale);
		std::memcpy(output, &packed, sizeof(packed));
	}

   public:
#endif

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, OutputType* diff_output, OutputType* abs_output, bool refresh, const int bucket_id) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;
		const auto& factors      = pos.state()->accumulator.factors; // FM項用

		// serialize.pyでPyTorchと同じ順序でsoftmax済みの12 bucket weightを生成している。
		// PairWeightテーブルの参照は必ず0～11に制限する。
		const int pair_bucket = std::clamp(bucket_id, 0, static_cast<int>(kPairWeightBuckets) - 1);
		const int16_t* curr_w_mul  = pair_weights_mul[pair_bucket];
		const int16_t* curr_w_diff = pair_weights_diff[pair_bucket];
		const int16_t* curr_w_sum  = pair_weights_sum[pair_bucket];

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

					__m256i w_mul  = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&curr_w_mul[idx])));
					__m256i w_diff = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&curr_w_diff[idx])));
					__m256i w_sum  = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&curr_w_sum[idx])));

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

		auto write_fm_quadratic_outputs = [&](IndexType j, int64_t ih_u,
			int64_t ik_u, int64_t ih_t, int64_t ik_t) {
			// 差分(Diff)と合計(Abs)の両方のパスを生成し、後段のネットワークに渡す
			// スケーリングとビットシフトにより 0-127 の範囲へ収める

			// ih/ik are true int64 interaction values. Keep their existing scalar
			// multiply, shift, clamp, and narrowing sequence unchanged.
			diff_output[ 0 + j] = ToOutputRange(((ih_u - ih_t) * FMScale1) >> 37); // ih(2次)差分
			diff_output[32 + j] = ToOutputRange(((ik_u - ik_t) * FMScale2) >> 37); // ik(2次)差分
			abs_output [ 0 + j] = ToOutputRange((ih_u * FMScale5) >> 37); // ih(2次)和
			abs_output [32 + j] = ToOutputRange((ik_u * FMScale6) >> 37); // ik(2次)和
		};

#if defined(USE_AVX2)
		static_assert(kFactorDimensions == 32,
			"AVX2 FM interaction expects 32 factor dimensions");
		const __m256i zero64 = _mm256_setzero_si256();
		const __m256i one64 = _mm256_set1_epi64x(1);
		auto get_inter4 = [&](const int64_t* sum_v, const int64_t* sum_v2,
			IndexType offset) {
			const __m256i v = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(sum_v + offset));
			const __m256i v2 = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(sum_v2 + offset));
			// HalfKA has at most 40 active int16 factors and KSDG3 at most 24,
			// so every sum_v fits in signed 32 bits. _mm256_mul_epi32 then
			// produces four exact signed 64-bit squares from the low dword of
			// each 64-bit lane.
			const __m256i numerator =
				_mm256_sub_epi64(_mm256_mul_epi32(v, v), v2);

			// C++ signed division by two truncates toward zero. Emulate it for
			// all signed inputs instead of relying only on the mathematically
			// even FM numerator.
			const __m256i negative = _mm256_cmpgt_epi64(zero64, numerator);
			const __m256i arithmetic_half = _mm256_or_si256(
				_mm256_srli_epi64(numerator, 1),
				_mm256_slli_epi64(negative, 63));
			const __m256i negative_odd_correction = _mm256_and_si256(
				_mm256_and_si256(negative, numerator), one64);
			return _mm256_add_epi64(arithmetic_half, negative_odd_correction);
		};

		for (IndexType j = 0; j < kFactorDimensions; j += 4) {
			alignas(32) int64_t interactions[4][4];
			_mm256_store_si256(reinterpret_cast<__m256i*>(interactions[0]),
				get_inter4(factors[us].halfka.sum_v,
					factors[us].halfka.sum_v2, j));
			_mm256_store_si256(reinterpret_cast<__m256i*>(interactions[1]),
				get_inter4(factors[us].ksdg.sum_v,
					factors[us].ksdg.sum_v2, j));
			_mm256_store_si256(reinterpret_cast<__m256i*>(interactions[2]),
				get_inter4(factors[them].halfka.sum_v,
					factors[them].halfka.sum_v2, j));
			_mm256_store_si256(reinterpret_cast<__m256i*>(interactions[3]),
				get_inter4(factors[them].ksdg.sum_v,
					factors[them].ksdg.sum_v2, j));

			for (IndexType lane = 0; lane < 4; ++lane) {
				const IndexType index = j + lane;
				write_fm_quadratic_outputs(
					index,
					interactions[0][lane], interactions[1][lane],
					interactions[2][lane], interactions[3][lane]);
			}

			const __m256i sh_u = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[us].halfka.sum_v + j));
			const __m256i sk_u = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[us].ksdg.sum_v + j));
			const __m256i sh_t = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[them].halfka.sum_v + j));
			const __m256i sk_t = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[them].ksdg.sum_v + j));
			StoreFmLinear4(diff_output + 64 + j,
				_mm256_sub_epi64(sh_u, sh_t), FMScale3);
			StoreFmLinear4(diff_output + 96 + j,
				_mm256_sub_epi64(sk_u, sk_t), FMScale4);
			StoreFmLinear4(abs_output + 64 + j, sh_u, FMScale7);
			StoreFmLinear4(abs_output + 96 + j, sk_u, FMScale8);
		}
#else
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			// 2次相互作用項の公式: (Σv)^2 - Σ(v^2)
			auto get_inter = [](int64_t v, int64_t v2) { return (v * v - v2) / 2; };
			const int64_t ih_u = get_inter(
				factors[us].halfka.sum_v[j], factors[us].halfka.sum_v2[j]);
			const int64_t ik_u = get_inter(
				factors[us].ksdg.sum_v[j], factors[us].ksdg.sum_v2[j]);
			const int64_t ih_t = get_inter(
				factors[them].halfka.sum_v[j], factors[them].halfka.sum_v2[j]);
			const int64_t ik_t = get_inter(
				factors[them].ksdg.sum_v[j], factors[them].ksdg.sum_v2[j]);
			write_fm_quadratic_outputs(j, ih_u, ik_u, ih_t, ik_t);
			const int64_t sh_u = factors[us].halfka.sum_v[j];
			const int64_t sk_u = factors[us].ksdg.sum_v[j];
			const int64_t sh_t = factors[them].halfka.sum_v[j];
			const int64_t sk_t = factors[them].ksdg.sum_v[j];
			diff_output[64 + j] = ToOutputRange(
				((sh_u - sh_t) * FMScale3) >> 22);
			diff_output[96 + j] = ToOutputRange(
				((sk_u - sk_t) * FMScale4) >> 22);
			abs_output[64 + j] = ToOutputRange((sh_u * FMScale7) >> 22);
			abs_output[96 + j] = ToOutputRange((sk_u * FMScale8) >> 22);
		}
#endif
	}

#if defined(ENABLE_NNUE_TRACE)
	// Trace-only accessors. These are compiled out of normal tournament builds.
	void TracePairWeights(const int bucket_id, std::int16_t* weight_mul,
	                      std::int16_t* weight_diff,
	                      std::int16_t* weight_sum) const {
		const int pair_bucket =
			std::clamp(bucket_id, 0, static_cast<int>(kPairWeightBuckets) - 1);
		for (IndexType j = 0; j < kPairWeightDimensions; ++j) {
			weight_mul[j] = pair_weights_mul[pair_bucket][j];
			weight_diff[j] = pair_weights_diff[pair_bucket][j];
			weight_sum[j] = pair_weights_sum[pair_bucket][j];
		}
	}

	void TraceMainPair(const Position& pos, const Color perspective,
	                   const int bucket_id, std::int32_t* a_values,
	                   std::int32_t* b_values, std::int32_t* mul_terms,
	                   std::int32_t* diff_sq_terms,
	                   std::int32_t* sum_terms,
	                   std::int32_t* mixed_numerators) const {
		const int pair_bucket =
			std::clamp(bucket_id, 0, static_cast<int>(kPairWeightBuckets) - 1);
		const auto& accumulation = pos.state()->accumulator.accumulation;
		for (IndexType j = 0; j < kPairWeightDimensions; ++j) {
			const std::int32_t a =
				std::clamp<std::int32_t>(accumulation[perspective][0][j], 0, 127);
			const std::int32_t b = std::clamp<std::int32_t>(
				accumulation[perspective][0][j + kPairWeightDimensions], 0, 127);
			const std::int32_t mul_term = a * b;
			const std::int32_t diff = a - b;
			const std::int32_t diff_sq_term = diff * diff;
			const std::int32_t sum_term = (a + b) * 64;
			const std::int32_t mixed_numerator =
				mul_term * pair_weights_mul[pair_bucket][j]
				+ diff_sq_term * pair_weights_diff[pair_bucket][j]
				+ sum_term * pair_weights_sum[pair_bucket][j];

			a_values[j] = a;
			b_values[j] = b;
			mul_terms[j] = mul_term;
			diff_sq_terms[j] = diff_sq_term;
			sum_terms[j] = sum_term;
			mixed_numerators[j] = mixed_numerator;
		}
	}
#endif  // defined(ENABLE_NNUE_TRACE)

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

#if defined(USE_AVX2)
	template <bool Add, IndexType Offset>
	static inline void update_fm_factor_chunk(
		Accumulator::FactorGroup& group, const WeightType* values) {
		const __m128i values16 =
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(values + Offset));
		const __m256i values32 = _mm256_cvtepi16_epi32(values16);
		const __m256i squares32 = _mm256_mullo_epi32(values32, values32);

		const __m128i values32_low = _mm256_castsi256_si128(values32);
		const __m128i values32_high = _mm256_extracti128_si256(values32, 1);
		const __m128i squares32_low = _mm256_castsi256_si128(squares32);
		const __m128i squares32_high = _mm256_extracti128_si256(squares32, 1);

		const __m256i values64_low = _mm256_cvtepi32_epi64(values32_low);
		const __m256i values64_high = _mm256_cvtepi32_epi64(values32_high);
		const __m256i squares64_low = _mm256_cvtepi32_epi64(squares32_low);
		const __m256i squares64_high = _mm256_cvtepi32_epi64(squares32_high);

		auto* sum_v_low = reinterpret_cast<__m256i*>(group.sum_v + Offset);
		auto* sum_v_high = reinterpret_cast<__m256i*>(group.sum_v + Offset + 4);
		auto* sum_v2_low = reinterpret_cast<__m256i*>(group.sum_v2 + Offset);
		auto* sum_v2_high = reinterpret_cast<__m256i*>(group.sum_v2 + Offset + 4);

		const __m256i old_sum_v_low = _mm256_loadu_si256(sum_v_low);
		const __m256i old_sum_v_high = _mm256_loadu_si256(sum_v_high);
		const __m256i old_sum_v2_low = _mm256_loadu_si256(sum_v2_low);
		const __m256i old_sum_v2_high = _mm256_loadu_si256(sum_v2_high);

		if constexpr (Add) {
			_mm256_storeu_si256(sum_v_low, _mm256_add_epi64(old_sum_v_low, values64_low));
			_mm256_storeu_si256(sum_v_high, _mm256_add_epi64(old_sum_v_high, values64_high));
			_mm256_storeu_si256(sum_v2_low, _mm256_add_epi64(old_sum_v2_low, squares64_low));
			_mm256_storeu_si256(sum_v2_high, _mm256_add_epi64(old_sum_v2_high, squares64_high));
		} else {
			_mm256_storeu_si256(sum_v_low, _mm256_sub_epi64(old_sum_v_low, values64_low));
			_mm256_storeu_si256(sum_v_high, _mm256_sub_epi64(old_sum_v_high, values64_high));
			_mm256_storeu_si256(sum_v2_low, _mm256_sub_epi64(old_sum_v2_low, squares64_low));
			_mm256_storeu_si256(sum_v2_high, _mm256_sub_epi64(old_sum_v2_high, squares64_high));
		}
	}
#endif

	template <bool Add>
	static inline void update_fm_factor_group(
		Accumulator::FactorGroup& group, const WeightType* values) {
#if defined(USE_AVX2)
		static_assert(kFactorDimensions == 32,
			"AVX2 FM accumulator update expects 32 factor dimensions");
		update_fm_factor_chunk<Add, 0>(group, values);
		update_fm_factor_chunk<Add, 8>(group, values);
		update_fm_factor_chunk<Add, 16>(group, values);
		update_fm_factor_chunk<Add, 24>(group, values);
#else
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			const std::int64_t v = values[j];
			if constexpr (Add) {
				group.sum_v[j] += v;
				group.sum_v2[j] += v * v;
			} else {
				group.sum_v[j] -= v;
				group.sum_v2[j] -= v * v;
			}
		}
#endif
	}

	void refresh_fm_factors_from_scratch(
		Accumulator::FactorPart& factors,
		const Features::IndexList& active_indices) const {
		std::memset(&factors, 0, sizeof(factors));
		for (const auto index : active_indices) {
			const IndexType v_offset = index * kFactorDimensions;
			auto& group = index < SPLIT_IDX ? factors.ksdg : factors.halfka;
			update_fm_factor_group<true>(group, &v_weights_[v_offset]);
		}
	}

	// The correctness oracle: rebuild both Main and FM without using either
	// the previous StateInfo or a Finny entry.
	void refresh_accumulator_from_scratch(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[COLOR_NB];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (const Color perspective : {BLACK, WHITE}) {
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_,
					            kHalfDimensions * sizeof(BiasType));
					std::memset(&accumulator.factors[perspective], 0,
					            sizeof(accumulator.factors[perspective]));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0,
					            kHalfDimensions * sizeof(BiasType));
				}

				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
					auto accumulation = reinterpret_cast<vec_t*>(
						&accumulator.accumulation[perspective][i][0]);
					auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
					constexpr IndexType kNumChunks =
						kHalfDimensions * sizeof(BiasType) / sizeof(vec_t);
					for (IndexType j = 0; j < kNumChunks; ++j)
						accumulation[j] = vec_add_16(accumulation[j], column[j]);
#else
					for (IndexType j = 0; j < kHalfDimensions; ++j)
						accumulator.accumulation[perspective][i][j] +=
							weights_[offset + j];
#endif
					if (i == 0) {
						const IndexType v_offset = index * kFactorDimensions;
						auto& group = index < SPLIT_IDX
							? accumulator.factors[perspective].ksdg
							: accumulator.factors[perspective].halfka;
						update_fm_factor_group<true>(group, &v_weights_[v_offset]);
					}
				}
			}
		}
		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}

#if defined(USE_FINNY_TABLES)
	static constexpr bool kUseFinnyTables = kHalfDimensions <= 4096;

	struct alignas(kCacheLineSize) FinnyEntry {
		BiasType accumulation[kHalfDimensions];
		Features::IndexList active_indices;
		bool initialized = false;
	};

	struct FinnyCache {
		using TriggerEntries =
			std::array<std::array<FinnyEntry, SQ_NB>, COLOR_NB>;

		const FeatureTransformer* owner = nullptr;
		std::uint64_t generation = 0;
		std::array<TriggerEntries, kRefreshTriggers.size()> entries;

		void reset(const FeatureTransformer* new_owner,
		           const std::uint64_t new_generation) {
			owner = new_owner;
			generation = new_generation;
			for (auto& trigger_entries : entries)
				for (auto& perspective_entries : trigger_entries)
					for (auto& entry : perspective_entries)
						entry.initialized = false;
		}
	};

	static_assert(alignof(FinnyEntry) >= kCacheLineSize,
		"Finny entries must be cache-line aligned");

	static Square finny_bucket_square(
		const Position& pos, const Features::TriggerEvent trigger,
		const Color perspective) {
		switch (trigger) {
		case Features::TriggerEvent::kFriendKingMoved:
			return pos.square<KING>(perspective);
		case Features::TriggerEvent::kEnemyKingMoved:
			return pos.square<KING>(~perspective);
		case Features::TriggerEvent::kAnyKingMoved:
			return pos.square<KING>(perspective);
		default:
			return SQ_ZERO;
		}
	}

	static void copy_index_list(
		Features::IndexList& destination,
		const Features::IndexList& source) {
		destination.resize(source.size());
		for (std::size_t i = 0; i < source.size(); ++i)
			destination[i] = source[i];
	}

	static void make_index_diff(
		const Features::IndexList& old_active,
		const Features::IndexList& new_active,
		Features::IndexList& removed,
		Features::IndexList& added) {
		if (old_active.size() == new_active.size()) {
			for (std::size_t i = 0; i < old_active.size(); ++i) {
				if (old_active[i] != new_active[i]) {
					removed.push_back(old_active[i]);
					added.push_back(new_active[i]);
				}
			}
			return;
		}

		bool old_matched[RawFeatures::kMaxActiveDimensions] = {};
		bool new_matched[RawFeatures::kMaxActiveDimensions] = {};
		for (std::size_t old_index = 0; old_index < old_active.size(); ++old_index) {
			for (std::size_t new_index = 0; new_index < new_active.size(); ++new_index) {
				if (!new_matched[new_index]
					&& old_active[old_index] == new_active[new_index]) {
					old_matched[old_index] = true;
					new_matched[new_index] = true;
					break;
				}
			}
		}
		for (std::size_t i = 0; i < old_active.size(); ++i)
			if (!old_matched[i])
				removed.push_back(old_active[i]);
		for (std::size_t i = 0; i < new_active.size(); ++i)
			if (!new_matched[i])
				added.push_back(new_active[i]);
	}

	FinnyCache& finny_cache() const {
		static thread_local std::unique_ptr<FinnyCache> cache;
		if (!cache)
			cache = std::make_unique<FinnyCache>();
		if (cache->owner != this || cache->generation != finny_generation_)
			cache->reset(this, finny_generation_);
		return *cache;
	}

	template <typename ApplyChanges>
	void update_main_accumulator_to_two(
		const BiasType* source, BiasType* first_destination,
		BiasType* second_destination, ApplyChanges apply_changes) const {
#if defined(VECTOR)
		constexpr IndexType kNumChunks =
			kHalfDimensions * sizeof(BiasType) / sizeof(vec_t);
		for (IndexType chunk = 0; chunk < kNumChunks; ++chunk) {
			vec_t value = source
				? vec_load(reinterpret_cast<const vec_t*>(source) + chunk)
				: vec_zero();
			apply_changes(value, chunk);
			vec_store(reinterpret_cast<vec_t*>(first_destination) + chunk, value);
			vec_store(reinterpret_cast<vec_t*>(second_destination) + chunk, value);
		}
#else
		if (source && source != first_destination)
			std::memcpy(first_destination, source,
			            kHalfDimensions * sizeof(BiasType));
		else if (!source)
			std::memset(first_destination, 0,
			            kHalfDimensions * sizeof(BiasType));
		apply_changes(first_destination, 0);
		std::memcpy(second_destination, first_destination,
		            kHalfDimensions * sizeof(BiasType));
#endif
	}

	void refresh_main_accumulator_using_finny_entry(
		BiasType* current, FinnyEntry& entry,
		const Features::IndexList& active_indices,
		const IndexType trigger_index
#if defined(ENABLE_NNUE_BENCH)
		, BenchmarkFinnyStatistics* const statistics = nullptr
#endif
		) const {
		if (!entry.initialized) {
#if defined(ENABLE_NNUE_BENCH)
			if (statistics) {
				++statistics->misses;
				statistics->added_features += active_indices.size();
			}
#endif
			const BiasType* source = trigger_index == 0 ? biases_ : nullptr;
			update_main_accumulator_to_two(
				source, entry.accumulation, current,
				[&](auto& value, const IndexType chunk) {
#if defined(VECTOR)
					for (const auto index : active_indices) {
						const auto* column = reinterpret_cast<const vec_t*>(
							&weights_[kHalfDimensions * index]);
						value = vec_add_16(value, vec_load(column + chunk));
					}
#else
					for (const auto index : active_indices)
						for (IndexType j = 0; j < kHalfDimensions; ++j)
							value[j] += weights_[kHalfDimensions * index + j];
#endif
				});
			copy_index_list(entry.active_indices, active_indices);
			entry.initialized = true;
			return;
		}

		Features::IndexList removed_indices;
		Features::IndexList added_indices;
		make_index_diff(entry.active_indices, active_indices,
		                removed_indices, added_indices);
#if defined(ENABLE_NNUE_BENCH)
		if (statistics) {
			++statistics->hits;
			statistics->removed_features += removed_indices.size();
			statistics->added_features += added_indices.size();
		}
#endif
		update_main_accumulator_to_two(
			entry.accumulation, entry.accumulation, current,
			[&](auto& value, const IndexType chunk) {
#if defined(VECTOR)
				for (const auto index : removed_indices) {
					const auto* column = reinterpret_cast<const vec_t*>(
						&weights_[kHalfDimensions * index]);
					value = vec_sub_16(value, vec_load(column + chunk));
				}
				for (const auto index : added_indices) {
					const auto* column = reinterpret_cast<const vec_t*>(
						&weights_[kHalfDimensions * index]);
					value = vec_add_16(value, vec_load(column + chunk));
				}
#else
				for (const auto index : removed_indices)
					for (IndexType j = 0; j < kHalfDimensions; ++j)
						value[j] -= weights_[kHalfDimensions * index + j];
				for (const auto index : added_indices)
					for (IndexType j = 0; j < kHalfDimensions; ++j)
						value[j] += weights_[kHalfDimensions * index + j];
#endif
			});
		copy_index_list(entry.active_indices, active_indices);
	}

	void refresh_accumulator_with_finny_cache(
		const Position& pos, FinnyCache& cache
#if defined(ENABLE_NNUE_BENCH)
		, BenchmarkFinnyStatistics* const statistics = nullptr
#endif
		) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[COLOR_NB];
			const auto trigger = kRefreshTriggers[i];
			RawFeatures::AppendActiveIndices(pos, trigger, active_indices);
			for (const Color perspective : {BLACK, WHITE}) {
				const Square bucket = finny_bucket_square(pos, trigger, perspective);
				auto& entry = cache.entries[i][perspective][bucket];
				refresh_main_accumulator_using_finny_entry(
					accumulator.accumulation[perspective][i], entry,
					active_indices[perspective], i
#if defined(ENABLE_NNUE_BENCH)
					, statistics
#endif
					);
				if (i == 0)
					refresh_fm_factors_from_scratch(
						accumulator.factors[perspective], active_indices[perspective]);
			}
		}
		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}

	void refresh_accumulator_with_finny_cache(const Position& pos) const {
		auto& cache = finny_cache();
		refresh_accumulator_with_finny_cache(pos, cache
#if defined(ENABLE_NNUE_BENCH)
			, nullptr
#endif
			);
	}
#endif

	// Calculate cumulative value without using StateInfo difference calculation.
	// With Finny enabled, Main uses the per-thread cache while FM is always
	// rebuilt from the current active feature list.
	void refresh_accumulator(const Position& pos) const {
#if defined(USE_FINNY_TABLES)
		if constexpr (kUseFinnyTables) {
			refresh_accumulator_with_finny_cache(pos);
			return;
		}
#endif
		refresh_accumulator_from_scratch(pos);
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto& prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
			for (Color perspective : {BLACK, WHITE}) {
				bool fused_main_update = false;
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
					const bool direct_fused_main_update =
						removed_indices[perspective].size() == 1
						&& added_indices[perspective].size() == 1;

					if (!direct_fused_main_update) {
						std::memcpy(accumulator.accumulation[perspective][i],
						            prev_accumulator.accumulation[perspective][i],
						            kHalfDimensions * sizeof(BiasType));
					}

					if (i == 0) {
						std::memcpy(&accumulator.factors[perspective], &prev_accumulator.factors[perspective], sizeof(accumulator.factors[perspective]));
					}

					// Build the common one-remove/one-add Main FT update directly
					// from the previous accumulator, avoiding the preceding Main copy.
					// FM updates remain in the existing removed/added loops below.
					if (direct_fused_main_update) {
						const IndexType removed_offset =
							kHalfDimensions * removed_indices[perspective][0];
						const IndexType added_offset =
							kHalfDimensions * added_indices[perspective][0];
#if defined(VECTOR)
						auto previous_accumulation = reinterpret_cast<const vec_t*>(
							&prev_accumulator.accumulation[perspective][i][0]);
						auto removed_column =
							reinterpret_cast<const vec_t*>(&weights_[removed_offset]);
						auto added_column =
							reinterpret_cast<const vec_t*>(&weights_[added_offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							const vec_t after_remove =
								vec_sub_16(previous_accumulation[j], removed_column[j]);
							accumulation[j] = vec_add_16(after_remove, added_column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							BiasType after_remove = static_cast<BiasType>(
								prev_accumulator.accumulation[perspective][i][j]
								- weights_[removed_offset + j]);
							accumulator.accumulation[perspective][i][j] =
								static_cast<BiasType>(after_remove
									+ weights_[added_offset + j]);
						}
#endif
						fused_main_update = true;
					}

					for (const auto index : removed_indices[perspective]) {
						if (!fused_main_update) {
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
						}

						// FM項の減算
						if (i == 0) {
							const IndexType v_offset = index * kFactorDimensions;
							auto& group = (index < SPLIT_IDX) ? 
										  accumulator.factors[perspective].ksdg: 
										  accumulator.factors[perspective].halfka;
							update_fm_factor_group<false>(group, &v_weights_[v_offset]);
						}
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
						if (!fused_main_update) {
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
						}

						// FM項の加算
						if (i == 0) {
							const IndexType v_offset = index * kFactorDimensions;
							auto& group = (index < SPLIT_IDX) ? 
										  accumulator.factors[perspective].ksdg: 
										  accumulator.factors[perspective].halfka;
							update_fm_factor_group<true>(group, &v_weights_[v_offset]);
						}
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

#if defined(ENABLE_TEST_CMD)
	public:
	void TestRefreshAccumulatorFromScratch(const Position& pos) const {
		refresh_accumulator_from_scratch(pos);
	}
#if defined(USE_FINNY_TABLES)
	void TestRefreshAccumulatorWithFinny(const Position& pos) const {
		refresh_accumulator_with_finny_cache(pos);
	}

	void TestResetFinnyCache() const {
		finny_cache().reset(this, finny_generation_);
	}
#endif
	private:
#endif

#if defined(ENABLE_NNUE_BENCH)
	public:
	// Benchmark-only entry points. These are not compiled into normal builds.
	void BenchmarkRefreshAccumulator(const Position& pos) const {
		refresh_accumulator(pos);
	}

	void BenchmarkRefreshAccumulatorFromScratch(const Position& pos) const {
		refresh_accumulator_from_scratch(pos);
	}

#if defined(USE_FINNY_TABLES)
	void BenchmarkRefreshAccumulatorWithFinny(
		const Position& pos, BenchmarkFinnyStatistics* const statistics) const {
		auto& cache = finny_cache();
		refresh_accumulator_with_finny_cache(pos, cache, statistics);
	}

	void BenchmarkResetFinnyCache() const {
		finny_cache().reset(this, finny_generation_);
	}

	static constexpr std::size_t BenchmarkFinnyEntrySize() {
		return sizeof(FinnyEntry);
	}

	static constexpr std::size_t BenchmarkFinnyCacheSize() {
		return sizeof(FinnyCache);
	}

	static constexpr std::size_t BenchmarkFinnyEntryCount() {
		return kRefreshTriggers.size()
			 * static_cast<std::size_t>(COLOR_NB)
			 * static_cast<std::size_t>(SQ_NB);
	}
#endif

	struct alignas(32) BenchmarkFmInteractions {
		std::int64_t values[4][32];
	};

	// Recompute only the Main FT + PairWeight portion of Transform().
	// This intentionally duplicates the production expression so the benchmark
	// can isolate it without inserting callbacks or timers into Transform().
	void BenchmarkTransformMain(const Position& pos, OutputType* output,
	                            const int bucket_id) const {
		const auto& accumulation = pos.state()->accumulator.accumulation;
		const int pair_bucket =
			std::clamp(bucket_id, 0, static_cast<int>(kPairWeightBuckets) - 1);
		const int16_t* curr_w_mul  = pair_weights_mul[pair_bucket];
		const int16_t* curr_w_diff = pair_weights_diff[pair_bucket];
		const int16_t* curr_w_sum  = pair_weights_sum[pair_bucket];
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};

		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;
			const Color side = perspectives[p];
#if defined(VECTOR)
			constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / 32;
			const vec_t* in0 = reinterpret_cast<const vec_t*>(
				&(accumulation[side][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(
				&(accumulation[side][0][kHalfDimensions / 2]));

			auto blend_vec_3way = [&](vec_t a, vec_t b, IndexType j) {
				const __m256i V_Zero = _mm256_setzero_si256();
				const __m256i V_127  = _mm256_set1_epi16(127);
				const __m256i V_64   = _mm256_set1_epi32(64);
				a = _mm256_max_epi16(_mm256_min_epi16(a, V_127), V_Zero);
				b = _mm256_max_epi16(_mm256_min_epi16(b, V_127), V_Zero);

				auto compute8 = [&](__m128i a16, __m128i b16, IndexType idx) {
					const __m256i a32 = _mm256_cvtepi16_epi32(a16);
					const __m256i b32 = _mm256_cvtepi16_epi32(b16);
					const __m256i w_mul = _mm256_cvtepi16_epi32(
						_mm_loadu_si128(reinterpret_cast<const __m128i*>(
							&curr_w_mul[idx])));
					const __m256i w_diff = _mm256_cvtepi16_epi32(
						_mm_loadu_si128(reinterpret_cast<const __m128i*>(
							&curr_w_diff[idx])));
					const __m256i w_sum = _mm256_cvtepi16_epi32(
						_mm_loadu_si128(reinterpret_cast<const __m128i*>(
							&curr_w_sum[idx])));
					const __m256i term_mul = _mm256_mullo_epi32(
						_mm256_mullo_epi32(a32, b32), w_mul);
					const __m256i diff = _mm256_sub_epi32(a32, b32);
					const __m256i term_diff = _mm256_mullo_epi32(
						_mm256_mullo_epi32(diff, diff), w_diff);
					const __m256i term_sum = _mm256_mullo_epi32(
						_mm256_mullo_epi32(_mm256_add_epi32(a32, b32), w_sum),
						V_64);
					return _mm256_srai_epi32(_mm256_add_epi32(
						_mm256_add_epi32(term_mul, term_diff), term_sum), 21);
				};

				const __m256i res_lo = compute8(
					_mm256_extracti128_si256(a, 0),
					_mm256_extracti128_si256(b, 0), j);
				const __m256i res_hi = compute8(
					_mm256_extracti128_si256(a, 1),
					_mm256_extracti128_si256(b, 1), j + 8);
				return _mm256_permute4x64_epi64(
					_mm256_packs_epi32(res_lo, res_hi),
					_MM_SHUFFLE(3, 1, 2, 0));
			};

			for (IndexType j = 0; j < NumOutputChunks; ++j) {
				const vec_t blended0 = blend_vec_3way(
					in0[j * 2], in1[j * 2], j * 32);
				const vec_t blended1 = blend_vec_3way(
					in0[j * 2 + 1], in1[j * 2 + 1], j * 32 + 16);
				const __m256i packed8 = _mm256_permute4x64_epi64(
					_mm256_packus_epi16(blended0, blended1),
					_MM_SHUFFLE(3, 1, 2, 0));
				_mm256_storeu_si256(
					reinterpret_cast<__m256i*>(&output[offset + j * 32]), packed8);
			}
#else
			for (IndexType j = 0; j < kHalfDimensions / 2; ++j) {
				const int32_t a = std::clamp<int32_t>(
					accumulation[side][0][j], 0, 127);
				const int32_t b = std::clamp<int32_t>(
					accumulation[side][0][j + kHalfDimensions / 2], 0, 127);
				const int32_t diff = a - b;
				const int32_t blended_total =
					a * b * curr_w_mul[j]
					+ diff * diff * curr_w_diff[j]
					+ (a + b) * 64 * curr_w_sum[j];
				output[offset + j] =
					static_cast<OutputType>(blended_total >> 21);
			}
#endif
		}
	}

	// Recompute the four 32-element FM interaction vectors. The stored layout
	// is us.HalfKA, us.KSDG3, them.HalfKA, them.KSDG3.
	void BenchmarkTransformFmInteractionsOnly(
		const Position& pos, BenchmarkFmInteractions& interactions) const {
		const auto& factors = pos.state()->accumulator.factors;
		const Color us = pos.side_to_move();
		const Color them = ~us;
#if defined(USE_AVX2)
		const __m256i zero64 = _mm256_setzero_si256();
		const __m256i one64 = _mm256_set1_epi64x(1);
		auto get_inter4 = [&](const int64_t* sum_v, const int64_t* sum_v2,
		                      IndexType offset) {
			const __m256i v = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(sum_v + offset));
			const __m256i v2 = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(sum_v2 + offset));
			const __m256i numerator =
				_mm256_sub_epi64(_mm256_mul_epi32(v, v), v2);
			const __m256i negative = _mm256_cmpgt_epi64(zero64, numerator);
			const __m256i arithmetic_half = _mm256_or_si256(
				_mm256_srli_epi64(numerator, 1),
				_mm256_slli_epi64(negative, 63));
			const __m256i correction = _mm256_and_si256(
				_mm256_and_si256(negative, numerator), one64);
			return _mm256_add_epi64(arithmetic_half, correction);
		};

		for (IndexType j = 0; j < kFactorDimensions; j += 4) {
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(&interactions.values[0][j]),
				get_inter4(factors[us].halfka.sum_v,
				           factors[us].halfka.sum_v2, j));
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(&interactions.values[1][j]),
				get_inter4(factors[us].ksdg.sum_v,
				           factors[us].ksdg.sum_v2, j));
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(&interactions.values[2][j]),
				get_inter4(factors[them].halfka.sum_v,
				           factors[them].halfka.sum_v2, j));
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(&interactions.values[3][j]),
				get_inter4(factors[them].ksdg.sum_v,
				           factors[them].ksdg.sum_v2, j));
		}
#else
		auto get_inter = [](int64_t v, int64_t v2) {
			return (v * v - v2) / 2;
		};
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			interactions.values[0][j] = get_inter(
				factors[us].halfka.sum_v[j], factors[us].halfka.sum_v2[j]);
			interactions.values[1][j] = get_inter(
				factors[us].ksdg.sum_v[j], factors[us].ksdg.sum_v2[j]);
			interactions.values[2][j] = get_inter(
				factors[them].halfka.sum_v[j], factors[them].halfka.sum_v2[j]);
			interactions.values[3][j] = get_inter(
				factors[them].ksdg.sum_v[j], factors[them].ksdg.sum_v2[j]);
		}
#endif
	}

	void BenchmarkTransformFmOutputsScalarOnly(
		const Position& pos, const BenchmarkFmInteractions& interactions,
		OutputType* diff_output, OutputType* abs_output) const {
		const auto& factors = pos.state()->accumulator.factors;
		const Color us = pos.side_to_move();
		const Color them = ~us;
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			const int64_t ih_u = interactions.values[0][j];
			const int64_t ik_u = interactions.values[1][j];
			const int64_t ih_t = interactions.values[2][j];
			const int64_t ik_t = interactions.values[3][j];
			const int64_t sh_u = factors[us].halfka.sum_v[j];
			const int64_t sk_u = factors[us].ksdg.sum_v[j];
			const int64_t sh_t = factors[them].halfka.sum_v[j];
			const int64_t sk_t = factors[them].ksdg.sum_v[j];

			diff_output[0 + j] = ToOutputRange(((ih_u - ih_t) * FMScale1) >> 37);
			diff_output[32 + j] = ToOutputRange(((ik_u - ik_t) * FMScale2) >> 37);
			diff_output[64 + j] = ToOutputRange(((sh_u - sh_t) * FMScale3) >> 22);
			diff_output[96 + j] = ToOutputRange(((sk_u - sk_t) * FMScale4) >> 22);
			abs_output[0 + j] = ToOutputRange((ih_u * FMScale5) >> 37);
			abs_output[32 + j] = ToOutputRange((ik_u * FMScale6) >> 37);
			abs_output[64 + j] = ToOutputRange((sh_u * FMScale7) >> 22);
			abs_output[96 + j] = ToOutputRange((sk_u * FMScale8) >> 22);
		}
	}

	void BenchmarkTransformFmOutputsHybridOnly(
		const Position& pos, const BenchmarkFmInteractions& interactions,
		OutputType* diff_output, OutputType* abs_output) const {
		const auto& factors = pos.state()->accumulator.factors;
		const Color us = pos.side_to_move();
		const Color them = ~us;

		// Preserve the production scalar ih/ik path exactly.
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			const int64_t ih_u = interactions.values[0][j];
			const int64_t ik_u = interactions.values[1][j];
			const int64_t ih_t = interactions.values[2][j];
			const int64_t ik_t = interactions.values[3][j];
			diff_output[0 + j] = ToOutputRange(
				((ih_u - ih_t) * FMScale1) >> 37);
			diff_output[32 + j] = ToOutputRange(
				((ik_u - ik_t) * FMScale2) >> 37);
			abs_output[0 + j] = ToOutputRange((ih_u * FMScale5) >> 37);
			abs_output[32 + j] = ToOutputRange((ik_u * FMScale6) >> 37);
		}

#if defined(USE_AVX2)
		for (IndexType j = 0; j < kFactorDimensions; j += 4) {
			const __m256i sh_u = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[us].halfka.sum_v + j));
			const __m256i sk_u = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[us].ksdg.sum_v + j));
			const __m256i sh_t = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[them].halfka.sum_v + j));
			const __m256i sk_t = _mm256_loadu_si256(
				reinterpret_cast<const __m256i*>(
					factors[them].ksdg.sum_v + j));
			StoreFmLinear4(diff_output + 64 + j,
				_mm256_sub_epi64(sh_u, sh_t), FMScale3);
			StoreFmLinear4(diff_output + 96 + j,
				_mm256_sub_epi64(sk_u, sk_t), FMScale4);
			StoreFmLinear4(abs_output + 64 + j, sh_u, FMScale7);
			StoreFmLinear4(abs_output + 96 + j, sk_u, FMScale8);
		}
#else
		for (IndexType j = 0; j < kFactorDimensions; ++j) {
			const int64_t sh_u = factors[us].halfka.sum_v[j];
			const int64_t sk_u = factors[us].ksdg.sum_v[j];
			const int64_t sh_t = factors[them].halfka.sum_v[j];
			const int64_t sk_t = factors[them].ksdg.sum_v[j];
			diff_output[64 + j] = ToOutputRange(
				((sh_u - sh_t) * FMScale3) >> 22);
			diff_output[96 + j] = ToOutputRange(
				((sk_u - sk_t) * FMScale4) >> 22);
			abs_output[64 + j] = ToOutputRange((sh_u * FMScale7) >> 22);
			abs_output[96 + j] = ToOutputRange((sk_u * FMScale8) >> 22);
		}
#endif
	}

	void BenchmarkTransformFmOutputsOnly(
		const Position& pos, const BenchmarkFmInteractions& interactions,
		OutputType* diff_output, OutputType* abs_output) const {
		BenchmarkTransformFmOutputsHybridOnly(
			pos, interactions, diff_output, abs_output);
	}

	void BenchmarkTransformReconstructed(
		const Position& pos, OutputType* output, OutputType* diff_output,
		OutputType* abs_output, const int bucket_id) const {
		BenchmarkFmInteractions interactions;
		BenchmarkTransformMain(pos, output, bucket_id);
		BenchmarkTransformFmInteractionsOnly(pos, interactions);
		BenchmarkTransformFmOutputsOnly(
			pos, interactions, diff_output, abs_output);
	}

	void BenchmarkTransformReconstructedScalarFm(
		const Position& pos, OutputType* output, OutputType* diff_output,
		OutputType* abs_output, const int bucket_id) const {
		BenchmarkFmInteractions interactions;
		BenchmarkTransformMain(pos, output, bucket_id);
		BenchmarkTransformFmInteractionsOnly(pos, interactions);
		BenchmarkTransformFmOutputsScalarOnly(
			pos, interactions, diff_output, abs_output);
	}

	static bool BenchmarkIsHalfKaIndex(const IndexType index) {
		return index >= SPLIT_IDX;
	}
	private:
#endif

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
	static constexpr IndexType kPairWeightTypes = 3;        // Mul / Diff / Sum
	static constexpr IndexType kPhaseBuckets = 4;           // nn.binのPyTorch復元用ブロック
	static constexpr IndexType kPairWeightBuckets = 12;     // C++推論用ブロック

	alignas(kCacheLineSize) int16_t pair_weights_mul [kPairWeightBuckets][kPairWeightDimensions];
	alignas(kCacheLineSize) int16_t pair_weights_diff[kPairWeightBuckets][kPairWeightDimensions];
	alignas(kCacheLineSize) int16_t pair_weights_sum [kPairWeightBuckets][kPairWeightDimensions];

#if defined(USE_FINNY_TABLES)
	std::uint64_t finny_generation_ = 0;
#endif

};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
