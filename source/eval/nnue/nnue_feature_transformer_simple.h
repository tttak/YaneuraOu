// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_SIMPLE_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_SIMPLE_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#if defined(SFNNwoPSQT)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"
#include "layers/simd.h"
#if defined(NNUE_SIMPLE_PP3WIDE_ANY)
#include "features/pp3wide_shogi.h"
#endif
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
#include "features/local_pair64_shogi.h"
#endif
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
#include "kp_progress_shadow.h"
#endif

#include <algorithm>  // std::clamp
#include <atomic>
#include <cstring>  // std::memset()

#if defined(USE_FINNY_TABLES)
#include <array>
#include <cstdint>
#include <memory>
#endif

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

/*
 例) SFNN1536のときのkNumChunksの計算

┌─────────┬───────────────┬─────────────────┬────────────┐
│  SIMD            │ sizeof(vec_t)                │ / sizeof(int16)                  │ kNumChunks             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX-512          │ 64                           │ 32                               │ 1536/32=48             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX2             │ 32                           │ 16                               │ 1536/16=96             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ SSE2             │ 16                           │ 8                                │ 1536/8=192             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ NEON             │ 16                           │ 8                                │ 1536/8=192             │
└─────────┴───────────────┴─────────────────┴────────────┘
*/

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
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
#if defined(KP_PROGRESS_SHADOW_TAIL_COMPACT)
	// Experiment-only compact tail.  Rows after the first are unaligned, so
	// weight loads use the unaligned intrinsic.  This is the literal
	// "one scalar after each row" layout used as Stage-2 layout B.
	static constexpr IndexType kProgressTailElements = 1;
#else
	// Keep every ordinary FT row cache-line aligned.  A single int16 tail
	// would make row N+1 unaligned and break the aligned AVX2 loads below.
	// Reserve one full cache line and use its first element for progress.
	static constexpr IndexType kProgressTailElements =
		kCacheLineSize / sizeof(WeightType);
#endif
	static constexpr IndexType kWeightRowStride =
		kHalfDimensions + kProgressTailElements;
#else
	static constexpr IndexType kWeightRowStride = kHalfDimensions;
#endif
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
#if defined(NNUE_HALFKAHM2_SIMPLE)
		return (0x7f234cb8u ^ 1536u)
#if defined(NNUE_SIMPLE_PP3WIDE)
		       ^ 0x50335731u
#elif defined(NNUE_SIMPLE_PP3WIDE64)
		       ^ 0x50335764u
#elif defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
#if defined(NNUE_SIMPLE_LOCALPAIR32_R5_D1)
		       ^ 0x47334431u
#elif defined(NNUE_SIMPLE_LOCALPAIR32_R5)
		       ^ 0x47533332u
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R5)
		       ^ 0x47535235u
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R2)
		       ^ 0x4B534732u
#else
		       ^ 0x4C503634u
#endif
#endif
		       ;
#elif defined(SFNNwoPSQT)
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
#if defined(USE_ELEMENT_WISE_MULTIPLY)
		// HalfKA_HM2 simple uses the ordinary, directly inspectable int16 FT
		// layout by default.  Retain automatic LEB128 detection so experimental
		// files produced before the non-compressed default remain loadable.
		const auto ft_start = stream.tellg();
		char magic[Leb128MagicStringSize];
		stream.read(magic, Leb128MagicStringSize);
		const bool compressed = stream.good()
			&& std::memcmp(magic, Leb128MagicString, Leb128MagicStringSize) == 0;
		stream.clear();
		stream.seekg(ft_start);

		if (compressed) {
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
			return Tools::ResultCode::FileMismatch;
#else
			read_leb_128<BiasType>(stream, biases_, kHalfDimensions);
			read_leb_128<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);
#endif
		} else {
			read_little_endian<BiasType>(stream, biases_, kHalfDimensions);
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
			NnueKpProgressShadow::load_from_environment(nullptr);
			for (IndexType row=0;row<kInputDimensions;++row) {
				read_little_endian<WeightType>(stream,
					&weights_[row*kWeightRowStride], kHalfDimensions);
				std::fill_n(&weights_[row*kWeightRowStride+kHalfDimensions],
					kProgressTailElements, WeightType{});
				weights_[row*kWeightRowStride+kHalfDimensions]=
					static_cast<WeightType>(NnueKpProgressShadow::runtime_weight(row));
			}
#else
			read_little_endian<WeightType>(
				stream, weights_, kHalfDimensions * kInputDimensions);
#endif
		}

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#if defined(USE_FINNY_TABLES)
		if (!stream.fail())
			++finny_generation_;
#endif

#if defined(NNUE_SIMPLE_PP3WIDE)
		read_pp3wide_weights(stream);
#elif defined(NNUE_SIMPLE_PP3WIDE64)
		read_pp3wide64_parameters(stream);
#elif defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		read_pp3wide64_parameters(stream);
#endif
#else
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t row = 0; row < kInputDimensions; ++row) {
			for (std::size_t i = 0; i < kHalfDimensions; ++i)
				weights_[row*kWeightRowStride+i] = read_little_endian<WeightType>(stream);
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
			weights_[row*kWeightRowStride+kHalfDimensions]=
				static_cast<WeightType>(NnueKpProgressShadow::runtime_weight(row));
#endif
		}
#if defined(USE_FINNY_TABLES)
		if (!stream.fail())
			++finny_generation_;
#endif
#endif
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		for (IndexType row=0;row<kInputDimensions;++row)
			stream.write(reinterpret_cast<const char*>(&weights_[row*kWeightRowStride]),
				kHalfDimensions*sizeof(WeightType));
#if defined(NNUE_SIMPLE_PP3WIDE)
		write_pp3wide_weights(stream);
#elif defined(NNUE_SIMPLE_PP3WIDE64)
		write_pp3wide64_parameters(stream);
#elif defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		write_pp3wide64_parameters(stream);
#endif
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos) const {
	#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
		return update_accumulator_stack(pos);
	#else
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
			++StackCounters().computed_hits;
	#endif
			return true;
		}
		const auto prev = now->previous;
		if (prev && prev->accumulator.computed_accumulation) {
			update_accumulator(pos);
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
			++StackCounters().one_ply;
	#endif
			return true;
		}
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
		++StackCounters().legacy_parent_uncomputed;
		++StackCounters().scratch_refresh;
	#endif
		return false;
	#endif
	}

	// Keep an exact accumulator chain when an external score cache bypasses
	// Transform().
	void EnsureAccumulator(const Position& pos) const {
		if (!UpdateAccumulatorIfPossible(pos))
			refresh_accumulator(pos);
	}


	void ForceRefreshAccumulator(const Position& pos) const {
		refresh_accumulator(pos);
	}

	void EnsureAccumulator(const Position& pos, bool refresh) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
	}

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, bool refresh) const {
		EnsureAccumulator(pos, refresh);
		const auto& accumulation = pos.state()->accumulator.accumulation;

#if defined(USE_ELEMENT_WISE_MULTIPLY)

#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT)
			// Packed output is sizeof(vec_t) bytes for each SIMD register
#if defined(USE_AVX512)
			constexpr IndexType OutputChunkSize = 64;
#else
			constexpr IndexType OutputChunkSize = kSimdWidth;
#endif
		static_assert((kHalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / OutputChunkSize;

		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127 * 2);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);

			constexpr int shift =
#if defined(USE_SSE2)
				7;
#else
				6;
#endif

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				vec_t raw0a = in0[j * 2 + 0], raw0b = in0[j * 2 + 1];
				vec_t raw1a = in1[j * 2 + 0], raw1b = in1[j * 2 + 1];
				const vec_t sum0a = vec_slli_16(vec_max_16(vec_min_16(raw0a, One), Zero), shift);
				const vec_t sum0b = vec_slli_16(vec_max_16(vec_min_16(raw0b, One), Zero), shift);
				const vec_t sum1a = vec_min_16(raw1a, One);
				const vec_t sum1b = vec_min_16(raw1b, One);

				const vec_t pa = vec_mulhi_16(sum0a, sum1a);
				const vec_t pb = vec_mulhi_16(sum0b, sum1b);

				out[j] = vec_packus_16(pa, pb);
			}

		}

#else
		constexpr int shift =
#if defined(VECTOR) && !defined(USE_SSE2)
			6;
#else
			7;
#endif

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			for (IndexType j = 0; j < kHalfDimensions / 2; ++j)
			{
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
				sum0 = std::clamp<BiasType>(sum0, 0, 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, 127 * 2);
				const int product = (int(sum0) << shift) * int(sum1);
				const int value = product >> 16;
				output[offset + j] = static_cast<OutputType>(std::clamp(value, 0, 255));
			}

		}
#endif

#else

		// 以下は旧NNUEのコード。
		// ループ本体がx86とNEONで異なる（2入力→1出力 vs 1入力→1出力）ため、
		// kNumChunksの意味自体がアーキテクチャごとに違うため、共通化しにくい。触らないことにする。

#if defined(USE_AVX512)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth * 2);
		static_assert(kHalfDimensions % (kSimdWidth * 2) == 0);
		const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
		const __m512i kZero    = _mm512_setzero_si512();

#elif defined(USE_AVX2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		constexpr int       kControl   = 0b11011000;
		const __m256i       kZero      = _mm256_setzero_si256();

#elif defined(USE_SSE2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#if defined(USE_SSE41)
		const __m128i kZero = _mm_setzero_si128();
#else  // SSE41非対応だがSSE2は使える環境
		const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		const __m64         k0x80s     = _mm_set1_pi8(-128);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		const int8x8_t      kZero      = {0};
#endif
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = kHalfDimensions * p;
#if defined(USE_AVX512)
			auto out = reinterpret_cast<__m512i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m512i sum0 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m512i sum1 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm512_add_epi16(
					    sum0,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm512_add_epi16(
					    sum1,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}
				_mm512_store_si512(&out[j], _mm512_permutexvar_epi64(
								 kControl, _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
			}

#elif defined(USE_AVX2)
			auto out = reinterpret_cast<__m256i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
					__m256i sum0 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
					__m256i sum1 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
					for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
						sum0 = _mm256_add_epi16(
							sum0,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 0]));
						sum1 = _mm256_add_epi16(
							sum1,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 1]));
					}
					_mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
									 _mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), kZero), kControl));
			}

#elif defined(USE_SSE2)
			auto out = reinterpret_cast<__m128i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m128i sum0 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m128i sum1 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm_add_epi16(sum0,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm_add_epi16(sum1,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}

				const __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
				_mm_store_si128(&out[j],
#if defined(USE_SSE41)
				                _mm_max_epi8(packedbytes, kZero)
#else  // SSE41非対応だがSSE2は使える環境
				                _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif
				);
			}

#elif defined(USE_MMX)
			// USE_MMX を config.h では現状、有効化することがないので dead code
			auto out = reinterpret_cast<__m64*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m64       sum0 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m64       sum1 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				const __m64 packedbytes = _mm_packs_pi16(sum0, sum1);
				out[j]                  = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
			}

#elif defined(USE_NEON)
			const auto out = reinterpret_cast<int8x8_t*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				int16x8_t sum = reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][0])[j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum = vaddq_s16(sum, reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][i])[j]);
				}
				out[j] = vmax_s8(vqmovn_s16(sum), kZero);
			}
#else
			for (IndexType j = 0; j < kHalfDimensions; ++j) {
				BiasType sum = accumulation[perspectives[p]][0][j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum += accumulation[perspectives[p]][i][j];
				}
				output[offset + j] = static_cast<OutputType>(std::clamp<int>(sum, 0, 127));
			}
#endif
		}
#if defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		_mm_empty();
#endif
#endif
	}

	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
	struct StackMaterializationCounters {
		std::atomic<std::uint64_t> computed_hits{0};
		std::atomic<std::uint64_t> one_ply{0};
		std::atomic<std::uint64_t> multi_ply{0};
		std::atomic<std::uint64_t> scratch_refresh{0};
		std::atomic<std::uint64_t> finny_refresh{0};
		std::atomic<std::uint64_t> finny_entry_hit{0};
		std::atomic<std::uint64_t> finny_entry_miss{0};
		std::atomic<std::uint64_t> refresh_in_chain{0};
		std::atomic<std::uint64_t> dirty_chain_total{0};
		std::atomic<std::uint64_t> dirty_chain_multi_count{0};
		std::atomic<std::uint64_t> max_dirty_chain{0};
		std::atomic<std::uint64_t> legacy_parent_uncomputed{0};
		std::atomic<std::uint64_t> evalhash_multi_ply{0};
	};

	static StackMaterializationCounters& StackCounters() {
		static StackMaterializationCounters counters;
		return counters;
	}

	static void ResetStackCounters() {
		auto& c = StackCounters();
		c.computed_hits = 0; c.one_ply = 0; c.multi_ply = 0;
		c.scratch_refresh = 0; c.finny_refresh = 0; c.refresh_in_chain = 0;
		c.finny_entry_hit = 0; c.finny_entry_miss = 0;
		c.dirty_chain_total = 0; c.dirty_chain_multi_count = 0;
		c.max_dirty_chain = 0;
		c.legacy_parent_uncomputed = 0;
		c.evalhash_multi_ply = 0;
	}
#endif

#if defined(NNUE_SIMPLE_PAIR64_ANY)
	// Convert the independent compact PP accumulator to two half-width EWM
	// vectors (side-to-move first), then apply the shared int8 width->16
	// projection.  The result is in the exact int32 raw unit used by fc_0.
	void TransformPp3Wide64(const Position& pos,
	                        std::int32_t* residual,
	                        std::uint8_t* diagnostic_transformed = nullptr) const {
		alignas(kCacheLineSize) std::uint8_t transformed[kPp64Width];
		TransformPp3Wide64Ewm(pos, transformed);
		ProjectPp3Wide64(transformed, residual);
		if (diagnostic_transformed)
			std::memcpy(diagnostic_transformed, transformed,
			            sizeof(transformed));
	}

	// Test-only entry points also document the two independent runtime costs.
	// Production calls the combined wrapper above.
	void TransformPp3Wide64Ewm(const Position& pos,
	                           std::uint8_t* transformed) const {
		const auto& accumulation =
			pos.state()->accumulator.pp3wide64_accumulation;
		const Color perspectives[2] = {
			pos.side_to_move(), ~pos.side_to_move()};
		constexpr int shift =
#if defined(USE_SSE2)
			7;
#else
			6;
#endif
		for (IndexType p = 0; p < 2; ++p) {
			const auto* source = accumulation[perspectives[p]];
			for (IndexType j = 0; j < kPp64Width / 2; ++j) {
				const int a = std::clamp<int>(source[j], 0, 127 * 2);
				const int b = std::clamp<int>(source[j + kPp64Width / 2], 0, 127 * 2);
				const int value = ((a << shift) * b) >> 16;
				transformed[p * (kPp64Width / 2) + j] = static_cast<std::uint8_t>(
					std::clamp(value, 0, 255));
			}
		}
	}

	void ProjectPp3Wide64(const std::uint8_t* transformed,
	                      std::int32_t* residual) const {
		for (IndexType o = 0; o < 16; ++o) {
			const auto* weights = &pp3wide64_projection_[o * kPp64Width];
#if defined(USE_AVX2)
			// Widen before multiplication.  maddubs would saturate each adjacent
			// int16 pair for the legal uint8/int8 extrema and would therefore
			// violate the scalar fixed-point contract.
			__m256i sum = _mm256_setzero_si256();
			for (IndexType i = 0; i < kPp64Width; i += 16) {
				const __m128i input8 = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(transformed + i));
				const __m128i weight8 = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(weights + i));
				const __m256i input16 = _mm256_cvtepu8_epi16(input8);
				const __m256i weight16 = _mm256_cvtepi8_epi16(weight8);
				sum = _mm256_add_epi32(
					sum, _mm256_madd_epi16(input16, weight16));
			}
			residual[o] = Simd::m256_hadd(sum, 0);
#else
			std::int32_t sum = 0;
			for (IndexType i = 0; i < kPp64Width; ++i)
				sum += static_cast<std::int32_t>(transformed[i])
				     * static_cast<std::int32_t>(weights[i]);
			residual[o] = sum;
#endif
		}
	}

	const std::int8_t* TestPp3Wide64Row(const IndexType index) const {
		return &pp3wide64_weights_[index * kPp64Width];
	}

	static constexpr IndexType TestPp3Wide64Width() { return kPp64Width; }

	static constexpr IndexType TestPp3Wide64Dimensions() {
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		return Features::LocalPair64Shogi::kDimensions;
#else
		return Features::Pp3WideShogi::kDimensions;
#endif
	}

	void TestApplyPp3Wide64Rows(std::int16_t* destination,
	                           const IndexType base) const {
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		Features::LocalPair64Shogi::IndexList removed, added;
		constexpr IndexType dimensions = Features::LocalPair64Shogi::kDimensions;
#else
		Features::Pp3WideShogi::IndexList removed, added;
		constexpr IndexType dimensions = Features::Pp3WideShogi::kDimensions;
#endif
		for (IndexType i = 0; i < 3; ++i)
			removed.push_back((base + i) % dimensions);
		for (IndexType i = 0; i < 4; ++i)
			added.push_back((base + 17 + i) % dimensions);
		pp64_apply_diff(destination, removed, added);
	}
#endif

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

#if defined(NNUE_SIMPLE_PAIR64_ANY)
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
	using Pp64Feature = Features::LocalPair64Shogi::IndexList;
	using Pp64Board = Features::LocalPair64Shogi::BoardState;
#else
	using Pp64Feature = Features::Pp3WideShogi::IndexList;
	using Pp64Board = Features::Pp3WideShogi::BoardState;
#endif
	static constexpr IndexType kPp64Width = kSimplePairDimensions;
	static constexpr IndexType kPp64Dimensions =
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		Features::LocalPair64Shogi::kDimensions;
#else
		Features::Pp3WideShogi::kDimensions;
#endif

	static void pp64_append_active(const Pp64Board& board, Color perspective,
	                              Square king, Pp64Feature& active) {
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		Features::LocalPair64Shogi::append_active(
			board, perspective, king, active);
#else
		Features::Pp3WideShogi::append_active(
			board, perspective, king, active);
#endif
	}

	static void pp64_make_dirty(const Pp64Board& before,
	                           const Pp64Board& after, Color perspective,
	                           Square king, Pp64Feature& removed,
	                           Pp64Feature& added) {
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		Features::LocalPair64Shogi::make_local_dirty_diff(
			before, after, perspective, king, removed, added);
#else
		Features::Pp3WideShogi::make_local_dirty_diff(
			before, after, perspective, king, removed, added);
#endif
	}

#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
	static void pp64_make_dirty_fast(const Position& pos,
	                                 Color perspective, Square king,
	                                 Pp64Feature& removed,
	                                 Pp64Feature& added) {
		Features::LocalPair64Shogi::make_local_dirty_diff_fast(
			pos, pos.state()->localpairDirty, perspective, king,
			removed, added);
	}
#endif

	static void pp64_make_diff(const Pp64Feature& before,
	                          const Pp64Feature& after, Pp64Feature& removed,
	                          Pp64Feature& added) {
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
		Features::LocalPair64Shogi::make_diff(before, after, removed, added);
#else
		Features::Pp3WideShogi::make_diff(before, after, removed, added);
#endif
	}

	void read_pp3wide64_parameters(std::istream& stream) {
		stream.read(reinterpret_cast<char*>(pp3wide64_weights_),
		            sizeof(pp3wide64_weights_));
		stream.read(reinterpret_cast<char*>(pp3wide64_projection_),
		            sizeof(pp3wide64_projection_));
	}

	void write_pp3wide64_parameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(pp3wide64_weights_),
		             sizeof(pp3wide64_weights_));
		stream.write(reinterpret_cast<const char*>(pp3wide64_projection_),
		             sizeof(pp3wide64_projection_));
	}

	static Pp64Board pp64_board_from_position(const Position& pos) {
		Pp64Board board{};
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto color = static_cast<Color>(c);
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
#if defined(NNUE_SIMPLE_LOCALPAIR64_R5)
			constexpr PieceType types[2] = {SILVER, GOLDS};
			for (int pc = 0; pc < int(std::size(types)); ++pc) {
				Bitboard bb = pos.pieces(color, types[pc]);
				while (bb)
					Features::LocalPair64Shogi::set_piece(
						board, c, pc, bb.pop());
			}
#elif defined(NNUE_SIMPLE_LOCALPAIR64_R2)
			constexpr PieceType types[3] = {KNIGHT, SILVER, GOLDS};
			for (int pc = 0; pc < 3; ++pc) {
				Bitboard bb = pos.pieces(color, types[pc]);
				while (bb)
					Features::LocalPair64Shogi::set_piece(
						board, c, pc, bb.pop());
			}
#else
			constexpr PieceType types[8] = {
				LANCE, KNIGHT, SILVER, GOLDS,
				BISHOP, HORSE, ROOK, DRAGON};
#if defined(NNUE_LOCALPAIR64_DIRTY_SCAN_REFERENCE)
			for (int pc = 0; pc < int(std::size(types)); ++pc)
				board.pieces[c][pc] = pos.pieces(color, types[pc]);
#else
			for (int pc = 0; pc < int(std::size(types)); ++pc) {
				Bitboard bb = pos.pieces(color, types[pc]);
				while (bb)
					Features::LocalPair64Shogi::set_piece(
						board, c, pc, bb.pop());
			}
#endif
#endif
#else
			board.pieces[c][0] = pos.pieces(color, PAWN);
			board.pieces[c][1] = pos.pieces(color, LANCE);
#endif
		}
		return board;
	}

	static Pp64Board pp64_board_from_state(const StateInfo& state,
	                                      bool after) {
		Pp64Board board{};
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
#if !defined(NNUE_LOCALPAIR_DIRTY_FASTPATH) \
 || defined(NNUE_LOCALPAIR_FASTPATH_KEEP_SNAPSHOTS)
		const auto* entries = after ? state.localpair64_after
		                            : state.localpair64_before;
		const auto count = after ? state.localpair64_after_count
		                         : state.localpair64_before_count;
		for (std::uint8_t i = 0; i < count; ++i) {
			const auto packed = entries[i];
			const auto sq = static_cast<Square>(packed & 0x7f);
			const int owner = (packed >> 7) & 1;
			const int pc = (packed >> 8) & 7;
#if defined(NNUE_LOCALPAIR64_DIRTY_SCAN_REFERENCE)
			board.pieces[owner][pc] |= Bitboard(sq);
#else
			board.at[static_cast<int>(sq)] = static_cast<std::uint8_t>(
				1 + owner * Features::LocalPair64Shogi::kClasses + pc);
#endif
		}
#else
		(void)state;
		(void)after;
#endif
#else
		for (int c = 0; c < COLOR_NB; ++c)
			for (int pc = 0; pc < 2; ++pc)
				board.pieces[c][pc] = after ? state.pp3wide_after[c][pc]
				                                  : state.pp3wide_before[c][pc];
#endif
		return board;
	}

	void pp64_apply_diff(std::int16_t* destination,
	                     const Pp64Feature& removed,
	                     const Pp64Feature& added) const {
#if defined(USE_AVX2)
		for (IndexType j = 0; j < kPp64Width; j += 16) {
			__m256i acc = _mm256_load_si256(
				reinterpret_cast<const __m256i*>(destination + j));
			for (const auto index : removed) {
				const auto* row = &pp3wide64_weights_[index * kPp64Width + j];
				const __m128i packed = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(row));
				acc = _mm256_sub_epi16(acc,
					_mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1));
			}
			for (const auto index : added) {
				const auto* row = &pp3wide64_weights_[index * kPp64Width + j];
				const __m128i packed = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(row));
				acc = _mm256_add_epi16(acc,
					_mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1));
			}
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(destination + j), acc);
		}
#else
		for (const auto index : removed)
			for (IndexType j = 0; j < kPp64Width; ++j)
				destination[j] -= 2 * pp3wide64_weights_[index * kPp64Width + j];
		for (const auto index : added)
			for (IndexType j = 0; j < kPp64Width; ++j)
				destination[j] += 2 * pp3wide64_weights_[index * kPp64Width + j];
#endif
	}

	void pp64_refresh_rows(std::int16_t* destination,
	                       const Pp64Feature& active) const {
		std::memset(destination, 0, kPp64Width * sizeof(std::int16_t));
		const Pp64Feature empty;
		pp64_apply_diff(destination, empty, active);
	}

	void refresh_pp3wide64(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		const auto board = pp64_board_from_position(pos);
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			Pp64Feature active;
			pp64_append_active(
				board, perspective, pos.square<KING>(perspective), active);
			pp64_refresh_rows(
				accumulator.pp3wide64_accumulation[perspective], active);
		}
	}

	void update_pp3wide64(const Position& pos) const {
		const auto& state = *pos.state();
		const auto& previous = state.previous->accumulator;
		auto& current = pos.state()->accumulator;
#if !defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 || !defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
		const auto before = pp64_board_from_state(state, false);
		const auto after = pp64_board_from_state(state, true);
		bool board_changed = false;
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
	&& !defined(NNUE_LOCALPAIR64_DIRTY_SCAN_REFERENCE)
		board_changed = before.at != after.at;
#else
		for (int c = 0; c < COLOR_NB; ++c)
			for (int pc = 0; pc <
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY)
				8
#else
				2
#endif
				; ++pc)
				board_changed |= before.pieces[c][pc] != after.pieces[c][pc];
#endif
#endif
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			auto* destination =
				current.pp3wide64_accumulation[perspective];
			const bool king_moved =
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
				state.dirtyPiece.pieceNo[0]
				== PIECE_NUMBER_KING + perspective;
#else
				state.dirtyPiece.pieceNo[0]
				== PIECE_NUMBER_KING + perspective;
#endif
			const Square king = pos.square<KING>(perspective);
			const bool mirror_boundary_crossed =
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
				(state.localpairDirty.flags & 2) != 0;
#else
				king_moved;
#endif
			if (king_moved && mirror_boundary_crossed) {
				Pp64Feature active;
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
	&& !defined(NNUE_LOCALPAIR64_DIRTY_SCAN_REFERENCE)
				// Lookup-only delayed snapshots intentionally omit bitboards.
				// King moves already require a full refresh, so enumerate from
				// the live Position instead of rebuilding bitboards every move.
				const auto refresh_board = pp64_board_from_position(pos);
				pp64_append_active(refresh_board, perspective, king, active);
#else
				pp64_append_active(after, perspective, king, active);
#endif
				pp64_refresh_rows(destination, active);
				continue;
			}
			std::memcpy(destination,
			            previous.pp3wide64_accumulation[perspective],
			            kPp64Width * sizeof(std::int16_t));
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
			if (state.localpairDirty.removed_count == 0
			    && state.localpairDirty.added_count == 0)
				continue;
#else
			if (!board_changed)
				continue;
#endif
			Pp64Feature removed, added;
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
			pp64_make_dirty_fast(pos, perspective, king, removed, added);
#else
			pp64_make_dirty(before, after, perspective, king, removed, added);
#endif
			if (removed.overflow || added.overflow) {
				Pp64Feature old_active, new_active;
				removed = Pp64Feature{};
				added = Pp64Feature{};
#if defined(NNUE_SIMPLE_LOCALPAIR64_ANY) \
 && defined(NNUE_LOCALPAIR_DIRTY_FASTPATH)
				// The fast path can emit at most 2*24 rows, below capacity.
				ASSERT(false);
#else
				pp64_append_active(before, perspective, king, old_active);
				pp64_append_active(after, perspective, king, new_active);
				pp64_make_diff(old_active, new_active, removed, added);
#endif
			}
			pp64_apply_diff(destination, removed, added);
		}
	}
#endif

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
				reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights_[j * kWeightRowStride]));
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights_[j * kWeightRowStride]);
			for (IndexType i = 0; i < kHalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

#if defined(NNUE_SIMPLE_PP3WIDE)
	using PpFeature = Features::Pp3WideShogi::IndexList;
	using PpBoard = Features::Pp3WideShogi::BoardState;

	static std::array<std::uint16_t, kHalfDimensions> pp_runtime_channel_map() {
		std::array<std::uint16_t, kHalfDimensions> map{};
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			map[i] = i;
#if defined(VECTOR) && !defined(NNUE_SMALL_SFNN_FT) && defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		auto* packed = reinterpret_cast<std::uint64_t*>(map.data());
		for (IndexType i = 0;
		     i < kHalfDimensions * sizeof(std::uint16_t) / sizeof(std::uint64_t);
		     i += di)
			inverse_order_packs(&packed[i]);
#endif
		return map;
	}

	void read_pp3wide_weights(std::istream& stream) {
		const auto map = pp_runtime_channel_map();
		std::array<std::int8_t, kHalfDimensions> logical{};
		for (IndexType row = 0; row < Features::Pp3WideShogi::kDimensions; ++row) {
			stream.read(reinterpret_cast<char*>(logical.data()), logical.size());
			auto* runtime = &pp3wide_weights_[row * kHalfDimensions];
			for (IndexType r = 0; r < kHalfDimensions; ++r)
				runtime[r] = logical[map[r]];
		}
	}

	void write_pp3wide_weights(std::ostream& stream) const {
		const auto map = pp_runtime_channel_map();
		std::array<std::int8_t, kHalfDimensions> logical{};
		for (IndexType row = 0; row < Features::Pp3WideShogi::kDimensions; ++row) {
			const auto* runtime = &pp3wide_weights_[row * kHalfDimensions];
			for (IndexType r = 0; r < kHalfDimensions; ++r)
				logical[map[r]] = runtime[r];
			stream.write(reinterpret_cast<const char*>(logical.data()), logical.size());
		}
	}

	static PpBoard pp_board_from_position(const Position& pos) {
		PpBoard board{};
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto color = static_cast<Color>(c);
			board.pieces[c][0] = pos.pieces(color, PAWN);
			board.pieces[c][1] = pos.pieces(color, LANCE);
		}
		return board;
	}

	static PpBoard pp_board_from_state(const StateInfo& state, bool after) {
		PpBoard board{};
		for (int c = 0; c < COLOR_NB; ++c)
			for (int pc = 0; pc < 2; ++pc)
				board.pieces[c][pc] = after ? state.pp3wide_after[c][pc]
				                                  : state.pp3wide_before[c][pc];
		return board;
	}

	void pp_apply_row(std::int16_t* destination, IndexType index, int sign) const {
		const auto* row = &pp3wide_weights_[index * kHalfDimensions];
	#if defined(USE_AVX2)
		const __m256i zero = _mm256_setzero_si256();
		for (IndexType j = 0; j < kHalfDimensions; j += 16) {
			const __m128i packed = _mm_loadu_si128(
				reinterpret_cast<const __m128i*>(row + j));
			__m256i delta = _mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1);
			const __m256i old = _mm256_load_si256(
				reinterpret_cast<const __m256i*>(destination + j));
			if (sign < 0)
				delta = _mm256_sub_epi16(zero, delta);
			_mm256_store_si256(reinterpret_cast<__m256i*>(destination + j),
			                   _mm256_add_epi16(old, delta));
		}
	#else
		for (IndexType j = 0; j < kHalfDimensions; ++j)
			destination[j] = static_cast<std::int16_t>(
				destination[j] + sign * 2 * static_cast<int>(row[j]));
	#endif
	}

	void pp_apply_diff(std::int16_t* destination,
	                   const PpFeature& removed,
	                   const PpFeature& added) const {
	#if defined(USE_AVX2)
		for (IndexType j = 0; j < kHalfDimensions; j += 16) {
			__m256i acc = _mm256_load_si256(
				reinterpret_cast<const __m256i*>(destination + j));
			for (const auto index : removed) {
				const auto* row = &pp3wide_weights_[index * kHalfDimensions + j];
				const __m128i packed = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(row));
				acc = _mm256_sub_epi16(acc,
					_mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1));
			}
			for (const auto index : added) {
				const auto* row = &pp3wide_weights_[index * kHalfDimensions + j];
				const __m128i packed = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(row));
				acc = _mm256_add_epi16(acc,
					_mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1));
			}
			_mm256_store_si256(reinterpret_cast<__m256i*>(destination + j), acc);
		}
	#else
		for (const auto index : removed) pp_apply_row(destination, index, -1);
		for (const auto index : added) pp_apply_row(destination, index, +1);
	#endif
	}

	void pp_refresh_rows(std::int16_t* destination,
	                     const PpFeature& active) const {
	#if defined(USE_AVX2)
		for (IndexType j = 0; j < kHalfDimensions; j += 16) {
			__m256i acc = _mm256_setzero_si256();
			for (const auto index : active) {
				const auto* row = &pp3wide_weights_[index * kHalfDimensions + j];
				const __m128i packed = _mm_loadu_si128(
					reinterpret_cast<const __m128i*>(row));
				acc = _mm256_add_epi16(acc,
					_mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1));
			}
			_mm256_store_si256(reinterpret_cast<__m256i*>(destination + j), acc);
		}
	#else
		std::memset(destination, 0, kHalfDimensions * sizeof(std::int16_t));
		for (const auto index : active) pp_apply_row(destination, index, +1);
	#endif
	}

	void refresh_pp3wide(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		const auto board = pp_board_from_position(pos);
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			PpFeature active;
			Features::Pp3WideShogi::append_active(
				board, perspective, pos.square<KING>(perspective), active);
			// PP rows use the same int16 accumulator scale as the ordinary FT
			// rows.  Fold them directly into trigger 0 so that every StateInfo
			// keeps only one 1536-wide accumulator and Transform needs no second
			// 6 KiB stream.
			auto* destination = accumulator.accumulation[perspective][0];
			const PpFeature empty;
			pp_apply_diff(destination, empty, active);
		}
	}

	void update_pp3wide(const Position& pos) const {
		const auto& state = *pos.state();
		auto& current = pos.state()->accumulator;
		const auto before = pp_board_from_state(state, false);
		const auto after = pp_board_from_state(state, true);
		bool board_changed = false;
		for (int c = 0; c < COLOR_NB; ++c)
			for (int pc = 0; pc < 2; ++pc)
				board_changed |= before.pieces[c][pc] != after.pieces[c][pc];
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			auto* destination = current.accumulation[perspective][0];
			const bool king_moved = state.dirtyPiece.pieceNo[0]
				== PIECE_NUMBER_KING + perspective;
			if (king_moved) {
				PpFeature active;
				Features::Pp3WideShogi::append_active(
					after, perspective, pos.square<KING>(perspective), active);
				const PpFeature empty;
				pp_apply_diff(destination, empty, active);
				continue;
			}
			if (!board_changed)
				continue;
			PpFeature removed, added;
			const Square king = pos.square<KING>(perspective);
			Features::Pp3WideShogi::make_local_dirty_diff(
				before, after, perspective, king, removed, added);
			if (removed.overflow || added.overflow) {
				// Preserve the already-copied parent accumulator: apply an exact
				// full-set difference instead of replacing the complete Main row.
				PpFeature old_active, new_active, full_removed, full_added;
				Features::Pp3WideShogi::append_active(
					before, perspective, king, old_active);
				Features::Pp3WideShogi::append_active(
					after, perspective, king, new_active);
				Features::Pp3WideShogi::make_diff(
					old_active, new_active, full_removed, full_added);
				pp_apply_diff(destination, full_removed, full_added);
			} else {
				pp_apply_diff(destination, removed, added);
			}
		}
	}
#endif

#if defined(VECTOR)
	// 変更された各特徴量ごとにaccumulator全体を読み書きするのを避けるため、
	// SIMDレジスタに収まるタイル単位で差分をまとめて適用する。
	static constexpr IndexType kVectorHeight = sizeof(vec_t) / sizeof(BiasType);
	static_assert(kHalfDimensions % kVectorHeight == 0, "kVectorHeight must divide kHalfDimensions");
	static constexpr IndexType kNumVectorChunks = kHalfDimensions / kVectorHeight;
	static constexpr IndexType kTileRegs = [] {
		IndexType regs = std::min(kNumRegs, kNumVectorChunks);
		while (kNumVectorChunks % regs != 0)
			--regs;
		return regs;
	}();
	static constexpr IndexType kTileHeight = kTileRegs * kVectorHeight;

	template <typename ApplyChanges>
	void update_accumulator_tiled(
		const BiasType* source, BiasType* destination,
		ApplyChanges apply_changes) const {
		for (IndexType tile_offset = 0; tile_offset < kHalfDimensions; tile_offset += kTileHeight) {
			vec_t acc[kTileRegs];

			if (source) {
				const auto* source_tile = reinterpret_cast<const vec_t*>(source + tile_offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_load(source_tile + k);
			} else {
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_zero();
			}

			apply_changes(acc, tile_offset);

			auto* destination_tile = reinterpret_cast<vec_t*>(destination + tile_offset);
			for (IndexType k = 0; k < kTileRegs; ++k)
				vec_store(destination_tile + k, acc[k]);
		}
	}

	void add_weight_to_tile(vec_t* acc, IndexType index, IndexType tile_offset) const {
		const auto* column = reinterpret_cast<const vec_t*>(
			&weights_[kWeightRowStride * index + tile_offset]);
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_add_16(acc[k], weight_vec_load(column + k));
	}

	void sub_weight_from_tile(vec_t* acc, IndexType index, IndexType tile_offset) const {
		const auto* column = reinterpret_cast<const vec_t*>(
			&weights_[kWeightRowStride * index + tile_offset]);
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_sub_16(acc[k], weight_vec_load(column + k));
	}

#if defined(NNUE_SIMPLE_PP3WIDE)
	static inline vec_t pp_weight_vec_load(const std::int8_t* source) {
#if defined(USE_AVX512)
		const __m256i packed = _mm256_loadu_si256(
			reinterpret_cast<const __m256i*>(source));
		return _mm512_slli_epi16(_mm512_cvtepi8_epi16(packed), 1);
#elif defined(USE_AVX2)
		const __m128i packed = _mm_loadu_si128(
			reinterpret_cast<const __m128i*>(source));
		return _mm256_slli_epi16(_mm256_cvtepi8_epi16(packed), 1);
#else
		alignas(kCacheLineSize) BiasType expanded[kVectorHeight];
		for (IndexType i = 0; i < kVectorHeight; ++i)
			expanded[i] = static_cast<BiasType>(2 * static_cast<int>(source[i]));
		return vec_load(reinterpret_cast<const vec_t*>(expanded));
#endif
	}

	void add_pp_weight_to_tile(vec_t* acc, IndexType index,
	                           IndexType tile_offset) const {
		const auto* row = &pp3wide_weights_[index * kHalfDimensions + tile_offset];
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_add_16(
				acc[k], pp_weight_vec_load(row + k * kVectorHeight));
	}

	void sub_pp_weight_from_tile(vec_t* acc, IndexType index,
	                            IndexType tile_offset) const {
		const auto* row = &pp3wide_weights_[index * kHalfDimensions + tile_offset];
		for (IndexType k = 0; k < kTileRegs; ++k)
			acc[k] = vec_sub_16(
				acc[k], pp_weight_vec_load(row + k * kVectorHeight));
	}
#endif

	static inline vec_t weight_vec_load(const vec_t* source) {
#if defined(KP_PROGRESS_SHADOW_TAIL_COMPACT) && defined(USE_AVX512)
		return _mm512_loadu_si512(source);
#elif defined(KP_PROGRESS_SHADOW_TAIL_COMPACT) && defined(USE_AVX2)
		return _mm256_loadu_si256(source);
#else
		return vec_load(source);
#endif
	}
#endif

	// StockfishのAccumulatorCaches(Finny Tables)と同じ発想。
	// 王位置ごとにrefresh済みaccumulatorを持ち、次回はactive featureの差分だけを適用する。
#if defined(USE_FINNY_TABLES)
	static constexpr bool kUseFinnyTables = kHalfDimensions <= 4096;

	struct alignas(kCacheLineSize) FinnyEntry {
		BiasType accumulation[kHalfDimensions];
		Features::IndexList active_indices;
		bool initialized = false;
	};

	struct FinnyCache {
		using TriggerEntries = std::array<std::array<FinnyEntry, SQ_NB>, COLOR_NB>;

		const FeatureTransformer* owner = nullptr;
		std::uint64_t generation = 0;
		std::array<TriggerEntries, kRefreshTriggers.size()> entries;

		void reset(const FeatureTransformer* new_owner, std::uint64_t new_generation) {
			owner = new_owner;
			generation = new_generation;
			for (auto& trigger_entries : entries)
				for (auto& perspective_entries : trigger_entries)
					for (auto& entry : perspective_entries)
						entry.initialized = false;
		}
	};

	static Square finny_bucket_square(
		const Position& pos, Features::TriggerEvent trigger, Color perspective) {
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

	static void make_index_diff(
		const Features::IndexList& old_active,
		const Features::IndexList& new_active,
		Features::IndexList& removed,
		Features::IndexList& added) {
		if (old_active.size() == new_active.size()) {
			for (std::size_t i = 0; i < old_active.size(); ++i) {
				if (old_active[i] == new_active[i])
					continue;
				removed.push_back(old_active[i]);
				added.push_back(new_active[i]);
			}
			return;
		}

		bool old_matched[RawFeatures::kMaxActiveDimensions] = {};
		bool new_matched[RawFeatures::kMaxActiveDimensions] = {};

		for (std::size_t oi = 0; oi < old_active.size(); ++oi) {
			for (std::size_t ni = 0; ni < new_active.size(); ++ni) {
				if (!new_matched[ni] && old_active[oi] == new_active[ni]) {
					old_matched[oi] = true;
					new_matched[ni] = true;
					break;
				}
			}
		}

		for (std::size_t oi = 0; oi < old_active.size(); ++oi)
			if (!old_matched[oi])
				removed.push_back(old_active[oi]);
		for (std::size_t ni = 0; ni < new_active.size(); ++ni)
			if (!new_matched[ni])
				added.push_back(new_active[ni]);
	}

	static void copy_index_list(
		Features::IndexList& destination,
		const Features::IndexList& source) {
		destination.resize(source.size());
		for (std::size_t i = 0; i < source.size(); ++i)
			destination[i] = source[i];
	}
#endif

	void refresh_accumulator_from_scratch(
		BiasType* current, const Features::IndexList& active_indices, IndexType trigger_index) const {
#if defined(VECTOR)
		const auto* source = trigger_index == 0 ? biases_ : nullptr;
		update_accumulator_tiled(
			source, current,
			[&](vec_t* acc, IndexType tile_offset) {
				for (const auto index : active_indices)
					add_weight_to_tile(acc, index, tile_offset);
			});
#else
		if (trigger_index == 0)
			std::memcpy(current, biases_, kHalfDimensions * sizeof(BiasType));
		else
			std::memset(current, 0, kHalfDimensions * sizeof(BiasType));

		for (const auto index : active_indices) {
			const IndexType offset = kWeightRowStride * index;
			for (IndexType j = 0; j < kHalfDimensions; ++j)
				current[j] += weights_[offset + j];
		}
#endif
	}

#if defined(USE_FINNY_TABLES)
#if defined(VECTOR)
	template <typename ApplyChanges>
	void update_accumulator_tiled_to_two(
		const BiasType* source, BiasType* destination0, BiasType* destination1,
		ApplyChanges apply_changes) const {
		for (IndexType tile_offset = 0; tile_offset < kHalfDimensions; tile_offset += kTileHeight) {
			vec_t acc[kTileRegs];

			if (source) {
				const auto* source_tile = reinterpret_cast<const vec_t*>(source + tile_offset);
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_load(source_tile + k);
			} else {
				for (IndexType k = 0; k < kTileRegs; ++k)
					acc[k] = vec_zero();
			}

			apply_changes(acc, tile_offset);

			auto* destination0_tile = reinterpret_cast<vec_t*>(destination0 + tile_offset);
			auto* destination1_tile = reinterpret_cast<vec_t*>(destination1 + tile_offset);
			for (IndexType k = 0; k < kTileRegs; ++k) {
				vec_store(destination0_tile + k, acc[k]);
				vec_store(destination1_tile + k, acc[k]);
			}
		}
	}
#endif

	void refresh_accumulator_using_finny_entry(
		BiasType* current,
		FinnyEntry& entry,
		const Features::IndexList& active_indices,
		IndexType trigger_index) const {
#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
		if (entry.initialized)
			++StackCounters().finny_entry_hit;
		else
			++StackCounters().finny_entry_miss;
#endif
		if (!entry.initialized) {
#if defined(VECTOR)
			const auto* source = trigger_index == 0 ? biases_ : nullptr;
			update_accumulator_tiled_to_two(
				source, entry.accumulation, current,
				[&](vec_t* acc, IndexType tile_offset) {
					for (const auto index : active_indices)
						add_weight_to_tile(acc, index, tile_offset);
				});
#else
			refresh_accumulator_from_scratch(entry.accumulation, active_indices, trigger_index);
			std::memcpy(current, entry.accumulation, kHalfDimensions * sizeof(BiasType));
#endif
			copy_index_list(entry.active_indices, active_indices);
			entry.initialized = true;
		} else {
			Features::IndexList removed_indices, added_indices;
			make_index_diff(entry.active_indices, active_indices, removed_indices, added_indices);

#if defined(VECTOR)
			update_accumulator_tiled_to_two(
				entry.accumulation, entry.accumulation, current,
				[&](vec_t* acc, IndexType tile_offset) {
					for (const auto index : removed_indices)
						sub_weight_from_tile(acc, index, tile_offset);
					for (const auto index : added_indices)
						add_weight_to_tile(acc, index, tile_offset);
				});
#else
			for (const auto index : removed_indices) {
				const IndexType offset = kWeightRowStride * index;
				for (IndexType j = 0; j < kHalfDimensions; ++j)
					entry.accumulation[j] -= weights_[offset + j];
			}
			for (const auto index : added_indices) {
				const IndexType offset = kWeightRowStride * index;
				for (IndexType j = 0; j < kHalfDimensions; ++j)
					entry.accumulation[j] += weights_[offset + j];
			}
#endif
			copy_index_list(entry.active_indices, active_indices);
#if !defined(VECTOR)
			std::memcpy(current, entry.accumulation, kHalfDimensions * sizeof(BiasType));
#endif
		}
	}

#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
	void update_stack_transition(const StateInfo& transition) const {
		auto& destination = const_cast<Accumulator&>(transition.accumulator);
		auto& source = const_cast<Accumulator&>(transition.previous->accumulator);
		EnsureSimpleAccumulatorStorage(source);
		EnsureSimpleAccumulatorStorage(destination);

		static_assert(kRefreshTriggers.size() == 1,
		              "Experiment 136 Phase A requires HalfKA_HM2-only");
		using HalfKa = Features::HalfKA_hm2<Features::Side::kFriend>;
		const auto& dirty = transition.dirtyPiece;
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			Features::IndexList removed, added;
			const Square king = destination.stack_king_square[perspective];
			for (int n = 0; n < dirty.dirty_num; ++n) {
				const auto oldPiece = static_cast<BonaPiece>(
				  dirty.changed_piece[n].old_piece.from[perspective]);
				const auto newPiece = static_cast<BonaPiece>(
				  dirty.changed_piece[n].new_piece.from[perspective]);
				removed.push_back(HalfKa::MakeIndex(king, oldPiece));
				added.push_back(HalfKa::MakeIndex(king, newPiece));
			}

#if defined(VECTOR)
			update_accumulator_tiled(
			  source.accumulation[perspective][0],
			  destination.accumulation[perspective][0],
			  [&](vec_t* acc, IndexType tileOffset) {
				  for (const auto index : removed)
					  sub_weight_from_tile(acc, index, tileOffset);
				  for (const auto index : added)
					  add_weight_to_tile(acc, index, tileOffset);
			  });
#else
			std::memcpy(destination.accumulation[perspective][0],
			            source.accumulation[perspective][0],
			            kHalfDimensions * sizeof(BiasType));
			for (const auto index : removed)
				for (IndexType lane = 0; lane < kHalfDimensions; ++lane)
					destination.accumulation[perspective][0][lane]
					  -= weights_[kWeightRowStride * index + lane];
			for (const auto index : added)
				for (IndexType lane = 0; lane < kHalfDimensions; ++lane)
					destination.accumulation[perspective][0][lane]
					  += weights_[kWeightRowStride * index + lane];
#endif
		}
		destination.computed_accumulation = true;
		destination.computed_score = false;
		if (destination.stack_computed) *destination.stack_computed = true;
	}

	bool update_accumulator_stack(const Position& pos) const {
		auto* current = pos.state();
		if (current->accumulator.computed_accumulation) {
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
			auto& counters = StackCounters();
			++counters.computed_hits;
	#endif
			return true;
		}

		std::array<StateInfo*, MAX_PLY + 1> chain{};
		std::size_t length = 0;
		auto* cursor = current;
		while (cursor && !cursor->accumulator.computed_accumulation
		       && length < chain.size()) {
			chain[length++] = cursor;
			cursor = cursor->previous;
		}
		if (!cursor || length == chain.size()) {
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
			auto& counters = StackCounters();
			++counters.scratch_refresh;
	#endif
			return false;
		}

		// A king move changes the HalfKA_HM2 row coordinate.  Phase A keeps the
		// safe existing contract: refresh the requested leaf rather than trying
		// to reconstruct an intermediate board.
		for (std::size_t i = 0; i < length; ++i) {
			const auto& dirty = chain[i]->dirtyPiece;
			if (dirty.dirty_num && dirty.pieceNo[0] >= PIECE_NUMBER_KING) {
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
				auto& counters = StackCounters();
				++counters.refresh_in_chain;
				++counters.scratch_refresh;
	#endif
				refresh_accumulator(pos);
				return true;
			}
		}

		for (std::size_t i = length; i-- > 0;)
			update_stack_transition(*chain[i]);

	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
		auto& counters = StackCounters();
		if (length == 1)
			++counters.one_ply;
		else {
			++counters.multi_ply;
			bool evalhash_caused = false;
	#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
			for (std::size_t i = 0; i < length; ++i)
				evalhash_caused |= chain[i]->accumulator.stack_score_valid;
	#endif
			if (evalhash_caused)
				++counters.evalhash_multi_ply;
			counters.dirty_chain_total.fetch_add(length,
			  std::memory_order_relaxed);
			++counters.dirty_chain_multi_count;
			auto observed = counters.max_dirty_chain.load(std::memory_order_relaxed);
			while (observed < length
			       && !counters.max_dirty_chain.compare_exchange_weak(
				         observed, length, std::memory_order_relaxed)) {}
		}
	#endif
		return true;
	}
#endif

	// Phase B: Finny owns only the cache.  The caller explicitly supplies the
	// accumulator destination so the same cache algorithm can target either a
	// legacy StateInfo payload or a Worker-stack entry.
	void refresh_accumulator_with_finny_cache(
		const Position& pos, Accumulator& accumulator) const {
#if defined(NNUE_SIMPLE_ACCUMULATOR_DIAGNOSTICS)
		++StackCounters().finny_refresh;
#endif
		static thread_local std::unique_ptr<FinnyCache> cache;
		if (!cache)
			cache = std::make_unique<FinnyCache>();
		if (cache->owner != this || cache->generation != finny_generation_)
			cache->reset(this, finny_generation_);

#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
		EnsureSimpleAccumulatorStorage(accumulator);
		accumulator.stack_king_square[BLACK] = static_cast<Square>(
		  (pos.eval_list()->piece_list_fb()[PIECE_NUMBER_KING + BLACK]
		   - f_king) % SQ_NB);
		accumulator.stack_king_square[WHITE] = static_cast<Square>(
		  (pos.eval_list()->piece_list_fw()[PIECE_NUMBER_KING + WHITE]
		   - f_king) % SQ_NB);
#endif
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			const auto trigger = kRefreshTriggers[i];
			RawFeatures::AppendActiveIndices(pos, trigger, active_indices);
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
			if (i == 0) {
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
				std::int32_t sums[COLOR_NB]{};
				for (int c = 0; c < COLOR_NB; ++c)
					for (const auto index : active_indices[c])
						sums[c] += weights_[index * kWeightRowStride + kHalfDimensions];
				NnueKpProgressShadow::colocated_refresh(pos, sums);
#else
				NnueKpProgressShadow::refresh_from_active(pos, active_indices);
#endif
			}
#endif
			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
				const Square bucket = finny_bucket_square(pos, trigger, perspective);
				auto& entry = cache->entries[i][perspective][bucket];
				refresh_accumulator_using_finny_entry(
					accumulator.accumulation[perspective][i], entry,
					active_indices[perspective], i);
			}
		}

#if defined(NNUE_SIMPLE_PP3WIDE)
		refresh_pp3wide(pos);
#elif defined(NNUE_SIMPLE_PAIR64_ANY)
		refresh_pp3wide64(pos);
#endif

		accumulator.computed_accumulation = true;
	#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
		if (accumulator.stack_computed) *accumulator.stack_computed = true;
	#endif
		accumulator.computed_score = false;
	}
#endif

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
#if defined(USE_FINNY_TABLES)
		if constexpr (kUseFinnyTables) {
			refresh_accumulator_with_finny_cache(
				pos, const_cast<Accumulator&>(pos.state()->accumulator));
			return;
		}
#endif

		auto& accumulator = pos.state()->accumulator;
#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
		EnsureSimpleAccumulatorStorage(accumulator);
		accumulator.stack_king_square[BLACK] = static_cast<Square>(
		  (pos.eval_list()->piece_list_fb()[PIECE_NUMBER_KING + BLACK]
		   - f_king) % SQ_NB);
		accumulator.stack_king_square[WHITE] = static_cast<Square>(
		  (pos.eval_list()->piece_list_fw()[PIECE_NUMBER_KING + WHITE]
		   - f_king) % SQ_NB);
#endif
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
			if (i == 0) {
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
				std::int32_t sums[COLOR_NB]{};
				for (int c = 0; c < COLOR_NB; ++c)
					for (const auto index : active_indices[c])
						sums[c] += weights_[index * kWeightRowStride + kHalfDimensions];
				NnueKpProgressShadow::colocated_refresh(pos, sums);
#else
				NnueKpProgressShadow::refresh_from_active(pos, active_indices);
#endif
			}
#endif
			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
#if defined(VECTOR)
				auto* current = accumulator.accumulation[perspective][i];
				refresh_accumulator_from_scratch(current, active_indices[perspective], i);
#else
				refresh_accumulator_from_scratch(
					accumulator.accumulation[perspective][i], active_indices[perspective], i);
#endif
			}
		}

#if defined(NNUE_SIMPLE_PP3WIDE)
		refresh_pp3wide(pos);
#elif defined(NNUE_SIMPLE_PAIR64_ANY)
		refresh_pp3wide64(pos);
#endif

		accumulator.computed_accumulation = true;
	#if defined(NNUE_SIMPLE_ACCUMULATOR_STACK)
		if (accumulator.stack_computed) *accumulator.stack_computed = true;
	#endif
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto& prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
#if defined(NNUE_SIMPLE_PP3WIDE)
		// Prepare PP dirtiness once, before the 1536-wide tile loop.  The
		// lists are then consumed in the same load/store pass as HalfKA.
		const auto pp_before = pp_board_from_state(*pos.state(), false);
		const auto pp_after = pp_board_from_state(*pos.state(), true);
		bool pp_board_changed = false;
		for (int c = 0; c < COLOR_NB; ++c)
			for (int pc = 0; pc < 2; ++pc)
				pp_board_changed |= pp_before.pieces[c][pc]
				                 != pp_after.pieces[c][pc];
		PpFeature pp_removed[COLOR_NB], pp_added[COLOR_NB], pp_active[COLOR_NB];
		bool pp_active_ready[COLOR_NB]{};
		for (int c = 0; c < COLOR_NB; ++c) {
			const auto perspective = static_cast<Color>(c);
			const bool king_moved = pos.state()->dirtyPiece.pieceNo[0]
				== PIECE_NUMBER_KING + perspective;
			const Square king = pos.square<KING>(perspective);
			if (king_moved) {
				Features::Pp3WideShogi::append_active(
					pp_after, perspective, king, pp_active[c]);
				pp_active_ready[c] = true;
			} else if (pp_board_changed) {
				Features::Pp3WideShogi::make_local_dirty_diff(
					pp_before, pp_after, perspective, king,
					pp_removed[c], pp_added[c]);
				if (pp_removed[c].overflow || pp_added[c].overflow) {
					PpFeature old_active, new_active;
					pp_removed[c] = PpFeature{};
					pp_added[c] = PpFeature{};
					Features::Pp3WideShogi::append_active(
						pp_before, perspective, king, old_active);
					Features::Pp3WideShogi::append_active(
						pp_after, perspective, king, new_active);
					Features::Pp3WideShogi::make_diff(
						old_active, new_active, pp_removed[c], pp_added[c]);
				}
			}
		}
#endif
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
			if (i == 0) {
#if defined(KP_PROGRESS_SHADOW_INTERLEAVED)
				std::int32_t removed_sums[COLOR_NB]{}, added_sums[COLOR_NB]{};
				for (int c = 0; c < COLOR_NB; ++c) {
					for (const auto index : removed_indices[c])
						removed_sums[c] += weights_[index * kWeightRowStride + kHalfDimensions];
					for (const auto index : added_indices[c])
						added_sums[c] += weights_[index * kWeightRowStride + kHalfDimensions];
				}
				NnueKpProgressShadow::colocated_update(
					pos, removed_sums, added_sums, reset);
#else
				NnueKpProgressShadow::update_from_changed(
					pos, removed_indices, added_indices, reset);
#endif
			}
#endif
			for (int c = 0; c < COLOR_NB; ++c) {
				const Color perspective = static_cast<Color>(c);
#if defined(NNUE_SIMPLE_PP3WIDE)
				if (i == 0 && reset[perspective] && !pp_active_ready[c]) {
					Features::Pp3WideShogi::append_active(
						pp_after, perspective, pos.square<KING>(perspective),
						pp_active[c]);
					pp_active_ready[c] = true;
				}
#endif
#if defined(VECTOR)
				auto* current = accumulator.accumulation[perspective][i];
				if (reset[perspective]) {
					const auto* source = i == 0 ? biases_ : nullptr;
					update_accumulator_tiled(
						source, current,
						[&](vec_t* acc, IndexType tile_offset) {
							for (const auto index : added_indices[perspective])
								add_weight_to_tile(acc, index, tile_offset);
#if defined(NNUE_SIMPLE_PP3WIDE)
							if (i == 0)
								for (const auto index : pp_active[c])
									add_pp_weight_to_tile(acc, index, tile_offset);
#endif
						});
				} else {
					update_accumulator_tiled(
						prev_accumulator.accumulation[perspective][i],
						current,
						[&](vec_t* acc, IndexType tile_offset) {
							for (const auto index : removed_indices[perspective])
								sub_weight_from_tile(acc, index, tile_offset);
							for (const auto index : added_indices[perspective])
								add_weight_to_tile(acc, index, tile_offset);
#if defined(NNUE_SIMPLE_PP3WIDE)
							if (i == 0) {
								for (const auto index : pp_removed[c])
									sub_pp_weight_from_tile(acc, index, tile_offset);
								for (const auto index : pp_added[c])
									add_pp_weight_to_tile(acc, index, tile_offset);
							}
#endif
						});
				}
#else
				if (reset[perspective]) {
					if (i == 0) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kWeightRowStride * index;
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
					}
				}
				// Difference calculation for features that changed from 0 to 1
				// 0から1に変化した特徴量に関する差分計算
				for (const auto index : added_indices[perspective]) {
					const IndexType offset = kWeightRowStride * index;
					for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
					}
				}
#if defined(NNUE_SIMPLE_PP3WIDE)
				if (i == 0) {
					if (reset[perspective]) {
						for (const auto index : pp_active[c])
							pp_apply_row(accumulator.accumulation[perspective][i], index, +1);
					} else {
						for (const auto index : pp_removed[c])
							pp_apply_row(accumulator.accumulation[perspective][i], index, -1);
						for (const auto index : pp_added[c])
							pp_apply_row(accumulator.accumulation[perspective][i], index, +1);
					}
				}
#endif
#endif
			}
		}

#if defined(NNUE_SIMPLE_PAIR64_ANY)
		update_pp3wide64(pos);
#endif

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}


#if defined(ENABLE_TEST_CMD)
	public:
	void TestRefreshAccumulatorFromScratch(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active);
			for (int c = 0; c < COLOR_NB; ++c)
				refresh_accumulator_from_scratch(
					accumulator.accumulation[c][i], active[c], i);
		}
#if defined(NNUE_SIMPLE_PP3WIDE)
		refresh_pp3wide(pos);
#elif defined(NNUE_SIMPLE_PAIR64_ANY)
		refresh_pp3wide64(pos);
#endif
		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}
#if defined(NNUE_SIMPLE_PP3WIDE)
	void TestBuildPpAccumulator(const Position& pos, Color perspective,
	                            std::int16_t* destination) const {
		const auto board = pp_board_from_position(pos);
		PpFeature active;
		Features::Pp3WideShogi::append_active(
			board, perspective, pos.square<KING>(perspective), active);
		pp_refresh_rows(destination, active);
	}
#elif defined(NNUE_SIMPLE_PAIR64_ANY)
	void TestBuildPp64Accumulator(const Position& pos, Color perspective,
	                              std::int16_t* destination) const {
		const auto board = pp64_board_from_position(pos);
		Pp64Feature active;
		pp64_append_active(
			board, perspective, pos.square<KING>(perspective), active);
		pp64_refresh_rows(destination, active);
	}
#endif
#if defined(USE_FINNY_TABLES)
	void TestRefreshAccumulatorWithFinny(const Position& pos) const {
		refresh_accumulator_with_finny_cache(
			pos, const_cast<Accumulator&>(pos.state()->accumulator));
	}
	void TestResetFinnyCache() const {
		// The upstream simple cache is function-local and generation-keyed.
		// Incrementing the generation invalidates it without exposing storage.
		++const_cast<FeatureTransformer*>(this)->finny_generation_;
	}
#endif
	private:
#endif

	// parameter type
	// パラメータの型

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kWeightRowStride * kInputDimensions];
#if defined(NNUE_SIMPLE_PP3WIDE)
	alignas(kCacheLineSize) std::int8_t
		pp3wide_weights_[Features::Pp3WideShogi::kDimensions * kHalfDimensions];
#elif defined(NNUE_SIMPLE_PAIR64_ANY)
	alignas(kCacheLineSize) std::int8_t
		pp3wide64_weights_[kPp64Dimensions * kPp64Width];
	alignas(kCacheLineSize) std::int8_t
		pp3wide64_projection_[16 * kPp64Width];
#endif
#if defined(USE_FINNY_TABLES)
	std::uint64_t finny_generation_ = 0;
#endif
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
