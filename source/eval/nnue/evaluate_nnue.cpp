// NNUE評価関数の計算に関するコード

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#if defined(EVAL_HASH_VERIFY_HITS)
#include <unordered_map>
#endif
#include <vector>
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
#include <atomic>
#include <chrono>
#include <iomanip>
#endif

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../../incbin/incbin.h"

#include "../../types.h"
#include "../../evaluate.h"
#include "../../position.h"
#include "../../memory.h"
#include "../../usi.h"

#if defined(USE_EVAL_HASH)
#include "../evalhash.h"
#include "../evalhash_atomic64.h"
#endif

#include "evaluate_nnue.h"
#if defined(ENABLE_NNUE_SHOGI_THREAT_SPARSE_PROTOTYPE)
#include "nnue_shogi_threat_lazy.h"
#endif
#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
#define NNUE_SIDE_INPUT_KING_SQUARE(pos, color) (pos).square<KING>(color)
#define NNUE_SIDE_INPUT_NAMESPACE_BEGIN namespace YaneuraOu {
#define NNUE_SIDE_INPUT_NAMESPACE_END }
#include "nnue_side_input.h"
#undef NNUE_SIDE_INPUT_NAMESPACE_END
#undef NNUE_SIDE_INPUT_NAMESPACE_BEGIN
#undef NNUE_SIDE_INPUT_KING_SQUARE
#endif
#if defined(ENABLE_NNUE_SIDE_INPUT_MOBILITY_TACTICAL_V1)
#include "nnue_mobility_tactical.h"
#endif
#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
#include "nnue_pair_relation.h"
#endif
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
#include "kp_progress_shadow.h"
#endif
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_FT_PROXY)
#include "kp_progress_ft_proxy.h"
#endif

namespace YaneuraOu::Eval::NNUE {
extern int FV_SCALE;

extern int FMScale1;
extern int FMScale2;
extern int FMScale3;
extern int FMScale4;
extern int FMScale5;
extern int FMScale6;
extern int FMScale7;
extern int FMScale8;

}
 
// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL)
using namespace YaneuraOu;
void add_options_(OptionsMap& options, ThreadPool& threads);

namespace {
YaneuraOu::OptionsMap* options_ptr;
YaneuraOu::ThreadPool* threads_ptr;
}

// 📌 旧Options、旧Threadsとの互換性のための共通のマクロ 📌
#define Options (*options_ptr)
#define Threads (*threads_ptr)

namespace YaneuraOu::Eval {
void add_options(OptionsMap& options, ThreadPool& threads) {
    options_ptr = &options;
    threads_ptr = &threads;
    add_options_(options, threads);
}
}
// ============================================================

// 評価関数を読み込み済みであるか
bool        eval_loaded   = false;
std::string last_eval_dir = "None";

// 📌 この評価関数で追加したいエンジンオプションはここで追加する。
void add_options_(OptionsMap& options, ThreadPool& threads) {

#if defined(EVAL_LEARN)
    // isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
    // test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
    // このコマンドの実行前に異常終了してしまう。
    // そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
    // test evalconvertコマンドを叩く。
    Options("SkipLoadingEval", Option(false));
#endif

#if defined(NNUE_EMBEDDING_OFF)
    const char* default_eval_dir = "eval";
#else
	// メモリから読み込む。
    const char* default_eval_dir = "<internal>";
#endif
    Options.add("EvalDir", Option(default_eval_dir, [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    if (last_eval_dir != eval_dir)
                    {
                        // 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
                        last_eval_dir = eval_dir;
                        eval_loaded   = false;
                    }
                    return std::nullopt;
                }));

    // NNUEのFV_SCALEの値
    Options.add("FV_SCALE", Option(16, 1, 128, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FV_SCALE = int(o);
                    return std::nullopt;
                }));

    // NNUEのFMScaleの値
    Options.add("FMScale1", Option(42273, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale1 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale2", Option(42273, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale2 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale3", Option(13107, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale3 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale4", Option(13107, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale4 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale5", Option(16909, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale5 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale6", Option(16909, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale6 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale7", Option(5243, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale7 = int(o);
                    return std::nullopt;
                }));
    Options.add("FMScale8", Option(5243, 0, 1000000000, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FMScale8 = int(o);
                    return std::nullopt;
                }));

}
#endif

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.

// デフォルトの効率的に更新可能なニューラルネットワーク（NNUE）ファイルの
// データをエンジンのバイナリに埋め込むためのマクロ
// （Dale Weiler 氏の incbin.h を使用）。
// このマクロを使うことで、以下の3つの変数が宣言されます：
//     const unsigned char        gEmbeddedNNUEData[];  // 埋め込まれたデータへのポインタ
//     const unsigned char *const gEmbeddedNNUEEnd;     // データの終端を示すマーカー
//     const unsigned int         gEmbeddedNNUESize;    // 埋め込まれたファイルのサイズ
// なお、この方法は Microsoft Visual Studio では動作しません。

#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#else
const unsigned char        gEmbeddedNNUEData[1] = { 0x0 };
const unsigned char* const gEmbeddedNNUEEnd = &gEmbeddedNNUEData[1];
const unsigned int         gEmbeddedNNUESize = 1;
#endif

// NNUEの埋め込みデータ型

namespace {

	struct EmbeddedNNUE {
		EmbeddedNNUE(const unsigned char* embeddedData,
			const unsigned char* embeddedEnd,
			const unsigned int   embeddedSize) :
			data(embeddedData),
			end(embeddedEnd),
			size(embeddedSize) {
		}
		const unsigned char* data;
		const unsigned char* end;
		const unsigned int   size;
	};

	//EmbeddedNNUE get_embedded(EmbeddedNNUEType type) {
	//	if (type == EmbeddedNNUEType::BIG)
	//		return EmbeddedNNUE(gEmbeddedNNUEBigData, gEmbeddedNNUEBigEnd, gEmbeddedNNUEBigSize);
	//	else
	//		return EmbeddedNNUE(gEmbeddedNNUESmallData, gEmbeddedNNUESmallEnd, gEmbeddedNNUESmallSize);
	//}

	// ⇨  StockfishはNNUEとして大きなnetworkと小さなnetworkがある。

	EmbeddedNNUE get_embedded() {
		return EmbeddedNNUE(gEmbeddedNNUEData, gEmbeddedNNUEEnd, gEmbeddedNNUESize);
	}
}


namespace YaneuraOu {
namespace Eval {
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
void EvalHash_DiagnosticBeforeTransform(const Position& pos, bool refresh);
void EvalHash_DiagnosticOnPropagate();
void EvalHash_DiagnosticOnComplexRouter();
void EvalHash_DiagnosticOnComplexNetwork();
#if defined(EVAL_HASH_VERIFY_HITS)
Value EvalHash_DiagnosticVerifyHit(const Position& pos, Value cached,
                                   std::uint8_t cached_flags = 0);
#endif
#endif
namespace NNUE {

	int FV_SCALE = 16; // 水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

	int FMScale1 = 42273;
	int FMScale2 = 42273;
	int FMScale3 = 13107;
	int FMScale4 = 13107;
	int FMScale5 = 16909;
	int FMScale6 = 16909;
	int FMScale7 = 5243;
	int FMScale8 = 5243;

    // 入力特徴量変換器
	LargePagePtr<FeatureTransformer> feature_transformer;

#if !defined(NNUE_HALFKAHM2_SIMPLE)
    // --- [追加] ルーター (全バケット共通) ---
    AlignedPtr<Router> router;
#endif

    // 評価関数
#if defined(SFNNwoPSQT)
    AlignedPtr<Network> network[kLayerStacks];
#else
    AlignedPtr<Network> network;
#endif

    // 評価関数ファイル名
    const char* const kFileName = EvalFileDefaultName;

    // 評価関数の構造を表す文字列を取得する
    std::string GetArchitectureString() {
#if defined(NNUE_HALFKAHM2_SIMPLE)
#if defined(NNUE_SIMPLE_PP3WIDE)
        return "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2_NoDG(Friend)"
               "+PP3WidePL[73305+15552->1536x2],"
               "Network=SFNN-1536-HalfKAHM2-NoDG-PP3WPL-v3"
               "{LayerStack=9}";
#else
        return "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2_NoDG(Friend)"
               "[73305->1536x2],Network=SFNN-1536-HalfKAHM2-NoDG-v2"
               "{LayerStack=9}";
#endif
#else
        const std::string base = "Features=" + FeatureTransformer::GetStructureString() +
			",Network=" + Network::GetStructureString();
#if defined(SFNNwoPSQT)
		return "ModelType=SFNNWithoutPsqt;" + base + "{LayerStack=" + std::to_string(kLayerStacks) + "}";
#else
		return base;
#endif
#endif
    }

namespace {
	namespace Detail {

		// 評価関数パラメータを初期化する
		template <typename T>
		void Initialize(AlignedPtr<T>& pointer) {
			pointer = make_unique_aligned<T>();
		}

		template <typename T>
		void Initialize(LargePagePtr<T>& pointer) {
			// →　メモリはLarge Pageから確保することで高速化する。
			pointer = make_unique_large_page<T>();
		}

            // 評価関数パラメータを読み込む
            template <typename T>
            Tools::Result ReadParameters(std::istream& stream, const AlignedPtr<T>& pointer) {
            	std::uint32_t header;
            	stream.read(reinterpret_cast<char*>(&header), sizeof(header));
            	if (!stream)                     return Tools::ResultCode::FileReadError;
            	//if (header != T::GetHashValue()) return Tools::ResultCode::FileMismatch;
				// 🤔 hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。
				if (header != T::GetHashValue())
                    sync_cout << "info string Warning : nn.bin hash mismatch." << sync_endl;
            	return pointer->ReadParameters(stream);
            }

			// 評価関数パラメータを読み込む
			template <typename T>
			Tools::Result ReadParameters(std::istream& stream, const LargePagePtr<T>& pointer) {
				std::uint32_t header;
				stream.read(reinterpret_cast<char*>(&header), sizeof(header));
				if (!stream)                     return Tools::ResultCode::FileReadError;
				// 🤔 hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。
				if (header != T::GetHashValue())
                    sync_cout << "info string Warning : nn.bin hash mismatch." << sync_endl;
				return pointer->ReadParameters(stream);
			}

			// 評価関数パラメータを書き込む
            template <typename T>
            bool WriteParameters(std::ostream& stream, const AlignedPtr<T>& pointer) {
                constexpr std::uint32_t header = T::GetHashValue();
                stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
                return pointer->WriteParameters(stream);
            }

			// 評価関数パラメータを書き込む
			template <typename T>
			bool WriteParameters(std::ostream& stream, const LargePagePtr<T>& pointer) {
				constexpr std::uint32_t header = T::GetHashValue();
				stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
				return pointer->WriteParameters(stream);
			}

		}  // namespace Detail
	
		// 評価関数パラメータを初期化する
		void Initialize() {
			Detail::Initialize<FeatureTransformer>(feature_transformer);
#if !defined(NNUE_HALFKAHM2_SIMPLE)
			Detail::Initialize<Router>(router);
#endif

#if defined(SFNNwoPSQT)
			for (int i = 0; i < kLayerStacks; ++i) {
				Detail::Initialize<Network>(network[i]);
			}
#else
			Detail::Initialize<Network>(network);
#endif
		}
	
		}  // namespace
    // ヘッダを読み込む
    Tools::Result ReadHeader(std::istream& stream,
        std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out) {
        std::uint32_t version = 0, size = 0;
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (!stream) return Tools::ResultCode::FileReadError;
		if (version_out)
			*version_out = version;
        if (version != kVersion) {
			sync_cout << "info string NNUE header version mismatch: expected " << kVersion
				<< " got " << version << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
        architecture->resize(size);
        stream.read(&(*architecture)[0], size);
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    // ヘッダを書き込む
    bool WriteHeader(std::ostream& stream,
        std::uint32_t hash_value, const std::string& architecture) {
        stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
        const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(architecture.data(), size);
        return !stream.fail();
    }

    	// 評価関数パラメータを読み込む
    	Tools::Result ReadParameters(std::istream& stream) {
    		std::uint32_t hash_value;
    		std::string architecture;
		Tools::Result result = ReadHeader(stream, &hash_value, &architecture, nullptr);
		if (result.is_not_ok()) return result;
#if !defined(NNUE_HALFKAHM2_SIMPLE)
#if defined(USE_NNUE_ABS_SQR_REMOVED_160)
#if defined(USE_NNUE_LEGACY_PHASE6)
		if (architecture.find("-L2x160-NoAbsSqr") == std::string::npos
			|| architecture.find("-Phase5") != std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this diagnostic binary requires legacy L2x160-NoAbsSqr Phase6, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#else
#if defined(USE_NNUE_L2_PHYSICAL_128)
		if (architecture.find("-L2x128-NoAbsSqr-Phase5-FC1x64-Cross16-FMDiff24-FMAbsRaw24")
			== std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires the L2x128/Cross16/FMDiff24/FMAbsRaw24 architecture, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#else
		if (architecture.find("-L2x160-NoAbsSqr-Phase5") == std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires L2x160-NoAbsSqr-Phase5, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
#if defined(USE_NNUE_FC1_WIDTH_64)
		if (architecture.find("-Phase5-FC1x64") == std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires FC1x64, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#if defined(USE_NNUE_L2_PHYSICAL_128)
		// Full compact128 marker was checked above.
#elif defined(USE_NNUE_CROSS_WIDTH_24)
		if (architecture.find("-FC1x64-Cross24") == std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires Cross24, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#else
		if (architecture.find("-Cross24") != std::string::npos
			|| architecture.find("-L2x128-") != std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires Cross32, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
#else
		if (architecture.find("-Phase5-FC1x64") != std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: this binary requires FC1x96, got "
				<< architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
#endif
#else
		if (architecture.find("-L2x160-NoAbsSqr") != std::string::npos) {
			sync_cout << "info string NNUE architecture mismatch: the file requires an L2x160-NoAbsSqr binary"
				<< sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
#endif
		if (hash_value != kHashValue) {
    			// hash check廃止: 警告のみ出力して続行する
    			sync_cout << "info string Warning: NNUE hash mismatch: expected " << kHashValue
    				<< " got " << hash_value
    				<< " arch_in_file=" << architecture
    				<< " arch_expected=" << GetArchitectureString()
    				<< sync_endl;
		}
#if defined(NNUE_HALFKAHM2_SIMPLE)
		if (architecture != GetArchitectureString()) {
			sync_cout << "info string NNUE simple architecture mismatch: expected "
				<< GetArchitectureString() << " got " << architecture << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
    
    		result = Detail::ReadParameters<FeatureTransformer>(stream, feature_transformer);
    		if (result.is_not_ok()) {
    			sync_cout << "info string NNUE feature params read failed: " << result.to_string() << sync_endl;
    			return result;
    		}

#if !defined(NNUE_HALFKAHM2_SIMPLE)
			// Router の読み込み
			sync_cout << "router->ReadParameters(stream) START!!" << sync_endl;
			router->ReadParameters(stream);
			sync_cout << "router->ReadParameters(stream) END!!" << sync_endl;
#endif

#if defined(SFNNwoPSQT)
    		for (int i = 0; i < kLayerStacks; ++i) {
    			result = Detail::ReadParameters<Network>(stream, network[i]);
    			if (result.is_not_ok()) {
    				sync_cout << "info string NNUE network params read failed at stack " << i << ": " << result.to_string() << sync_endl;
    				return result;
    			}
    		}
#else
    		result = Detail::ReadParameters<Network>(stream, network);
    		if (result.is_not_ok()) {
    			sync_cout << "info string NNUE network params read failed: " << result.to_string() << sync_endl;
    			return result;
    		}
#endif

    		if (stream && stream.peek() == std::ios::traits_type::eof())
    			return Tools::ResultCode::Ok;
    		else
    			return Tools::ResultCode::FileCloseError;
    	}
    // 評価関数パラメータを書き込む
    bool WriteParameters(std::ostream& stream) {
        if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
        if (!Detail::WriteParameters<FeatureTransformer>(stream, feature_transformer)) return false;
#if defined(SFNNwoPSQT)
        for (int i = 0; i < kLayerStacks; ++i) {
            if (!Detail::WriteParameters<Network>(stream, network[i])) return false;
        }
#else
        if (!Detail::WriteParameters<Network>(stream, network)) return false;
#endif
        return !stream.fail();
    }

    // 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossible(const Position& pos) {
        feature_transformer->UpdateAccumulatorIfPossible(pos);
    }

    static void EnsureAccumulator(const Position& pos) {
        feature_transformer->EnsureAccumulator(pos);
    }

    static void ForceRefreshAccumulator(const Position& pos) {
        feature_transformer->ForceRefreshAccumulator(pos);
    }

#if defined(EVAL_HASH_VERIFY_HITS) && !defined(NNUE_HALFKAHM2_SIMPLE)
    static void EvalHashDebugRefreshAccumulatorFromScratch(const Position& pos) {
        feature_transformer->EvalHashDebugRefreshAccumulatorFromScratch(pos);
    }

    static void EvalHashDebugDescribeFeatures(const Position& pos,
                                              std::ostream& out) {
        feature_transformer->EvalHashDebugDescribeFeatures(pos, out);
    }
#endif

#if defined(EVAL_HASH_COMPLEX_SAFE)
    static std::uint64_t AccumulatorFingerprint(const Position& pos) {
        const auto& accumulator = pos.state()->accumulator;
        auto mix = [](std::uint64_t h, std::uint64_t value) {
            value ^= value >> 30;
            value *= UINT64_C(0xbf58476d1ce4e5b9);
            value ^= value >> 27;
            return (h ^ value) * UINT64_C(0x9e3779b185ebca87);
        };
        std::uint64_t h = UINT64_C(0x6a09e667f3bcc909);
        const auto* main_words = reinterpret_cast<const std::uint64_t*>(
          accumulator.accumulation);
        constexpr std::size_t main_count = sizeof(accumulator.accumulation) / 8;
        // Evenly sample the full Main accumulator, including both perspectives
        // and every refresh trigger.  This keeps the cache key sensitive to
        // path-dependent accumulator states without scanning several KiB.
        for (std::size_t i = 0; i < 48; ++i)
            h = mix(h, main_words[(i * main_count) / 48]);
        const auto* factor_words = reinterpret_cast<const std::uint64_t*>(
          accumulator.factors);
        constexpr std::size_t factor_count = sizeof(accumulator.factors) / 8;
        for (std::size_t i = 0; i < 16; ++i)
            h = mix(h, factor_words[(i * factor_count) / 16]);
        return h ^ (h >> 29);
    }
#endif

#if defined(SFNNwoPSQT)
    // レイヤースタックの選択。双方の玉の段に応じて9通りに分岐させる。
    static int stack_index_for_nnue(const Position& pos) {
#if defined(NNUE_HALFKAHM2_SIMPLE)
        constexpr int kFToIndex[] = { 0, 0, 0, 3, 3, 3, 6, 6, 6 };
        constexpr int kEToIndex[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_rank = stm == BLACK ? rank_of(f_king) : rank_of(Inv(f_king));
        const auto e_rank = stm == BLACK ? rank_of(Inv(e_king)) : rank_of(e_king);
        return kFToIndex[f_rank] + kEToIndex[e_rank];
#else
/*
        constexpr int kFToIndex[] = { 0, 0, 0, 3, 3, 3, 6, 6, 6 };
        constexpr int kEToIndex[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
        const auto stm = pos.side_to_move();
        const auto f_king = pos.square<KING>(stm);
        const auto e_king = pos.square<KING>(~stm);
        const auto f_rank = stm == BLACK ? rank_of(f_king) : rank_of(Inv(f_king));
        const auto e_rank = stm == BLACK ? rank_of(Inv(e_king)) : rank_of(e_king);
        int idx = kFToIndex[f_rank] + kEToIndex[e_rank];
        if (idx < 0) idx = 0;
        if (idx >= kLayerStacks) idx = kLayerStacks - 1;
        return idx;
*/
        // 駒割りの差の絶対値から算出する
        constexpr int index[24] = {0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11};
        return index[std::min((std::abs(pos.state()->materialValue) + 99) / 100, 23)];
#endif
    }
#endif

#if !defined(NNUE_HALFKAHM2_SIMPLE)
    // Router による動的バケット選択を行うヘルパー関数
    inline int SelectBucketWithRouter(
        const std::uint8_t* router_input
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        , NnueSignalSnapshot* signal = nullptr
#endif
#if defined(USE_NNUE_ROUTER_LMR)
        , NnueRouterLmrSignal* router_lmr_signal = nullptr
#endif
        )
    {
        alignas(kCacheLineSize) std::int32_t router_out[32]; // SIMD制約のため32確保

        // Router consumes the input synchronously and does not retain it.
        router->PropagatePrefix<12>(router_input, router_out);

        // 出力の中から最大値を持つバケットを選択 (Argmax)
        int chosen_bucket = 0;
        std::int32_t max_score = router_out[0];
        for (int b = 1; b < kLayerStacks; ++b) {
            if (router_out[b] > max_score) {
                max_score = router_out[b];
                chosen_bucket = b;
            }
        }

#if defined(ENABLE_NNUE_SIGNAL_LOG) || defined(USE_NNUE_ROUTER_LMR)
        bool need_second_score = false;
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        need_second_score |= signal != nullptr;
#endif
#if defined(USE_NNUE_ROUTER_LMR)
        need_second_score |= router_lmr_signal != nullptr;
#endif
        if (need_second_score) {
            std::int32_t second_score = std::numeric_limits<std::int32_t>::min();
            for (int b = 0; b < kLayerStacks; ++b)
                if (b != chosen_bucket)
                    second_score = std::max(second_score, router_out[b]);
            const std::int64_t margin = static_cast<std::int64_t>(max_score) - second_score;
#if defined(ENABLE_NNUE_SIGNAL_LOG)
            if (signal) {
            signal->selected_bucket = chosen_bucket;
            signal->router_top1_logit = max_score;
            signal->router_top2_logit = second_score;
            signal->router_margin = static_cast<std::int32_t>(
              std::min<std::int64_t>(margin, std::numeric_limits<std::int32_t>::max()));
            }
#endif
#if defined(USE_NNUE_ROUTER_LMR)
            if (router_lmr_signal) {
                router_lmr_signal->router_margin = static_cast<std::int32_t>(
                  std::min<std::int64_t>(margin, std::numeric_limits<std::int32_t>::max()));
#if defined(USE_NNUE_DECISION_RISK_LMR) || defined(ENABLE_NNUE_ASPIRATION_DIAGNOSTIC)
                router_lmr_signal->selected_bucket = static_cast<std::int8_t>(chosen_bucket);
#endif
            }
#endif
        }
#endif

        return chosen_bucket;
    }
#endif

    // 評価値を計算する
    static Value ComputeScore(const Position& pos, bool refresh = false) {
        auto& accumulator = pos.state()->accumulator;
        if (!refresh && accumulator.computed_score) {
#if defined(ENABLE_NNUE_SIGNAL_LOG)
            SetLastNnueSignalAccess(NnueSignalEvalSource::AccumulatorCached,
                                    accumulator.nnue_signal.valid ? &accumulator.nnue_signal : nullptr);
#endif
#if defined(USE_NNUE_ROUTER_LMR)
            SetLastNnueRouterLmrSignal(accumulator.nnue_router_lmr_signal.valid
                                         ? &accumulator.nnue_router_lmr_signal : nullptr);
#endif
            return accumulator.score;
        }

#if defined(NNUE_HALFKAHM2_SIMPLE)
        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        EvalHash_DiagnosticBeforeTransform(pos, refresh);
#endif
        feature_transformer->Transform(pos, transformed_features, refresh);
        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
        const int bucket = stack_index_for_nnue(pos);
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_FT_PROXY)
        if (bucket == 8)
            NnueKpProgressFtProxy::observe(transformed_features);
#endif
#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)
        // Shadow only: compute the candidate route for the dominant B08
        // cohort, but deliberately keep `bucket` and the final score intact.
        if (bucket == 8)
            NnueKpProgressShadow::observe(pos);
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        EvalHash_DiagnosticOnPropagate();
#endif
        const auto output = network[bucket]->Propagate(transformed_features, buffer);
        auto score = static_cast<Value>(output[0] / FV_SCALE);
        score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);
        accumulator.score = score;
        accumulator.computed_score = true;
        return accumulator.score;
#else

#if defined(ENABLE_NNUE_SHOGI_THREAT_SPARSE_PROTOTYPE)
        // Experiment-only shadow accumulator.  Its pseudo residual is never
        // consumed by Network or the returned score.
        NnueThreatLazy::on_evaluate(pos);
#endif

        // L1パス用 (1280次元)
        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];

        // Diffパス用 (128次元)
        alignas(kCacheLineSize) TransformedFeatureType diff_transformed[128];

        // Absパス用 (128次元)
        alignas(kCacheLineSize) TransformedFeatureType abs_transformed[128];

        const auto bucket_id1 = stack_index_for_nnue(pos);

#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        EvalHash_DiagnosticBeforeTransform(pos, refresh);
#endif
        feature_transformer->Transform(pos, transformed_features, diff_transformed, abs_transformed, refresh, bucket_id1);

        // Router and Phase have the same 384-byte input layout. Build it once
        // in the Network buffer, let Router consume it, then replace only the
        // Phase-specific material-bucket byte.
        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
        auto& network_buffer = *reinterpret_cast<Network::Buffer*>(buffer);
        for (int j = 0; j < 128; ++j) {
            const int32_t abs_val = static_cast<int32_t>(abs_transformed[j]);
            network_buffer.phase_input[j] = static_cast<std::uint8_t>(
                std::clamp((abs_val - 64) * 2, 0, 127));
            network_buffer.phase_input[j + 128] =
                static_cast<std::uint8_t>(diff_transformed[j]);
            network_buffer.phase_input[j + 256] =
                static_cast<std::uint8_t>(transformed_features[j]);
        }

        // Router による動的バケット選択
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        NnueSignalSnapshot signal{};
#endif
#if defined(USE_NNUE_ROUTER_LMR)
        NnueRouterLmrSignal router_lmr_signal{};
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        EvalHash_DiagnosticOnComplexRouter();
#endif
        const auto bucket_id2 = SelectBucketWithRouter(
          network_buffer.phase_input
#if defined(ENABLE_NNUE_SIGNAL_LOG)
          , &signal
#endif
#if defined(USE_NNUE_ROUTER_LMR)
          , &router_lmr_signal
#endif
        );

        network_buffer.phase_input[127] =
          static_cast<std::uint8_t>((bucket_id1 * 127) / 11);

#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
        std::array<std::uint16_t, NnuePairRelation::MaxRelations>
          pair_relation_indices{};
        const std::size_t pair_relation_count = std::min(
          NnuePairRelation::generate(pos, pair_relation_indices.data(),
                                     pair_relation_indices.size()),
          pair_relation_indices.size());
#endif
#if defined(ENABLE_NNUE_SIDE_INPUT_MOBILITY_TACTICAL_V1)
        const auto mobility_tactical_input = NnueMobilityTactical::normalize(
          NnueMobilityTactical::extract(pos));
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        EvalHash_DiagnosticOnComplexNetwork();
        EvalHash_DiagnosticOnPropagate();
#endif
        const auto output = network[bucket_id2]->Propagate<true, true>(
          transformed_features, diff_transformed, abs_transformed, bucket_id1, buffer
#if defined(ENABLE_NNUE_SIGNAL_LOG)
          , &signal
#endif
#if (defined(USE_NNUE_PHASE_FM_LMR) || defined(USE_NNUE_LCA_LMR)) \
          && !defined(ENABLE_NNUE_SIGNAL_LOG)
          , &router_lmr_signal
#endif
#if defined(ENABLE_NNUE_SIDE_INPUT_SAFE_ESCAPE)
          , NnueSideInput::safe_escape_mask16(pos)
#endif
#if defined(ENABLE_NNUE_SIDE_INPUT_MOBILITY_TACTICAL_V1)
          , mobility_tactical_input.data()
#endif
#if defined(ENABLE_NNUE_PAIR_RELATION_SIDE_INPUT)
          , pair_relation_indices.data(), pair_relation_count
#endif
        );


        // VALUE_MAX_EVALより大きな値が返ってくるとaspiration searchがfail highして
        // 探索が終わらなくなるのでVALUE_MAX_EVAL以下であることを保証すべき。

        // この現象が起きても、対局時に秒固定などだとそこで探索が打ち切られるので、
        // 1つ前のiterationのときの最善手がbestmoveとして指されるので見かけ上、
        // 問題ない。このVALUE_MAX_EVALが返ってくるような状況は、ほぼ詰みの局面であり、
        // そのような詰みの局面が出現するのは終盤で形勢に大差がついていることが多いので
        // 勝敗にはあまり影響しない。

        // しかし、教師生成時などdepth固定で探索するときに探索から戻ってこなくなるので
        // そのスレッドの計算時間を無駄にする。またdepth固定対局でtime-outするようになる。

        auto score = static_cast<Value>(output[0] / FV_SCALE);

        // 1) ここ、下手にclipすると学習時には影響があるような気もするが…。
        // 2) accumulator.scoreは、差分計算の時に用いないので書き換えて問題ない。
        score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

#if defined(ENABLE_NNUE_HAO_SEARCH_RISK_SIGNAL) \
    && !defined(DISABLE_NNUE_HAO_SEARCH_RISK_COMPUTE)
        // The offline context used Hao HalfKP static score.  This diagnostic
        // deliberately substitutes the current production NNUE static score,
        // which is a distribution shift and is labelled as such in the report.
        const int material_stm = pos.state()->materialValue
          * (pos.side_to_move() == BLACK ? 1 : -1);
        network[bucket_id2]->ComputeHaoSearchRiskSignals(
          network_buffer.ac_1_out, pos.game_ply(), material_stm,
          static_cast<int>(score), &signal);
#endif

        accumulator.score = score;
        accumulator.computed_score = true;
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        signal.valid = true;
        accumulator.nnue_signal = signal;
        SetLastNnueSignalAccess(NnueSignalEvalSource::FreshNetwork, &accumulator.nnue_signal);
#endif
#if defined(USE_NNUE_ROUTER_LMR)
        router_lmr_signal.valid = true;
        accumulator.nnue_router_lmr_signal = router_lmr_signal;
        SetLastNnueRouterLmrSignal(&accumulator.nnue_router_lmr_signal);
#endif
        return accumulator.score;
#endif
    }

}  // namespace NNUE

#if defined(USE_EVAL_HASH)

// HashTableに評価値を保存するために利用するクラス
struct alignas(16) ScoreKeyValue {
#if defined(USE_SSE2)
    ScoreKeyValue() = default;
    ScoreKeyValue(const ScoreKeyValue & other) {
        static_assert(sizeof(ScoreKeyValue) == sizeof(__m128i),
            "sizeof(ScoreKeyValue) should be equal to sizeof(__m128i)");
        _mm_store_si128(&as_m128i, other.as_m128i);
    }
    ScoreKeyValue& operator=(const ScoreKeyValue & other) {
        _mm_store_si128(&as_m128i, other.as_m128i);
        return *this;
    }
#endif

    // evaluate hashでatomicに操作できる必要があるのでそのための操作子
    void encode() {
#if defined(USE_SSE2)
        // ScoreKeyValue は atomic にコピーされるので key が合っていればデータも合っている。
#else
        key ^= score;
#endif
    }
    // decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
    void decode() { encode(); }

    union {
        struct {
            std::uint64_t key;
            std::uint64_t score;
        };
#if defined(USE_SSE2)
        __m128i as_m128i;
#endif
    };
};

// evaluateしたものを保存しておくHashTable(俗にいうehash)

struct EvaluateHashTable : HashTable<ScoreKeyValue> {};

#if defined(EVAL_HASH_ATOMIC64)
#if defined(EVAL_HASH_COMPLEX_SAFE)
using EvalHashCodec = ComplexEvalHashCodec;
#else
using EvalHashCodec = SimpleEvalHashCodec;
#endif
using EvaluateHashTableSelected = Atomic64HashTable;
constexpr unsigned kAtomicTagBits = EvalHashCodec::kTagBits;
inline std::uint64_t evalhash_tag(Key key) { return EvalHashCodec::tag(key); }
inline std::uint64_t evalhash_pack(Key key, Value score, std::uint8_t flags = 0) {
    return EvalHashCodec::pack(key, score, flags);
}
inline bool evalhash_unpack(Key key, std::uint64_t packed, Value& score,
                            std::uint8_t* flags = nullptr) {
    return EvalHashCodec::unpack(key, packed, score, flags);
}
#else
using EvaluateHashTableSelected = EvaluateHashTable;
#endif

EvaluateHashTableSelected g_evalTable;
#if defined(EVAL_HASH_DEFAULT_ON)
bool g_evalHashRequested = true;
#else
bool g_evalHashRequested = false;
#endif
bool g_evalHashInitialized = false;
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_ROUTER_LMR)
namespace {
constexpr std::uint8_t kEvalHashRouterFlag = 1u << 0;
constexpr std::uint8_t kEvalHashLcaFlag    = 1u << 1;
constexpr std::uint8_t kEvalHashCrossFlag  = 1u << 2;
constexpr int kEvalHashRouterThreshold = 256;
constexpr int kEvalHashCrossThreshold = 127;
#if defined(NNUE_LCA_LMR_FIXED_THRESHOLD)
std::atomic<int> g_evalHashLcaThreshold{NNUE_LCA_LMR_FIXED_THRESHOLD};
#else
std::atomic<int> g_evalHashLcaThreshold{1959};
#endif

std::uint8_t evalhash_signal_flags(const NNUE::NnueRouterLmrSignal& signal) {
    if (!signal.valid) return 0;
    std::uint8_t flags = 0;
    if (signal.router_margin <= kEvalHashRouterThreshold)
        flags |= kEvalHashRouterFlag;
#if defined(USE_NNUE_LCA_LMR)
    if (signal.lca_abs_delta_sum >=
        g_evalHashLcaThreshold.load(std::memory_order_relaxed))
        flags |= kEvalHashLcaFlag;
#endif
#if defined(USE_NNUE_CROSS_LMR)
    if (signal.cross_abs_max >= kEvalHashCrossThreshold)
        flags |= kEvalHashCrossFlag;
#endif
    return flags;
}

NNUE::NnueRouterLmrSignal evalhash_restore_signal(std::uint8_t flags) {
    NNUE::NnueRouterLmrSignal signal{};
    signal.valid = true;
    signal.router_margin = flags & kEvalHashRouterFlag
                         ? kEvalHashRouterThreshold : kEvalHashRouterThreshold + 1;
#if defined(USE_NNUE_LCA_LMR)
    const int lca = g_evalHashLcaThreshold.load(std::memory_order_relaxed);
    signal.lca_abs_delta_sum = flags & kEvalHashLcaFlag ? lca : std::max(0, lca - 1);
#endif
#if defined(USE_NNUE_CROSS_LMR)
    signal.cross_abs_max = static_cast<std::uint8_t>(
      flags & kEvalHashCrossFlag ? kEvalHashCrossThreshold
                                 : kEvalHashCrossThreshold - 1);
#endif
    return signal;
}
}
#endif
void EvalHash_Resize(size_t mbSize) {
    // A true default must not probe the table before isready has allocated it.
    g_evalHashInitialized = false;
    g_evalTable.resize(Threads, mbSize);
    // resize() itself zero-initializes the selected table.  Mark it usable
    // here as well as in Clear(), because an option callback may legally run
    // after isready without a following Clear().
    g_evalHashInitialized = g_evalTable.entry_count() != 0;
}
void EvalHash_Clear() {
    g_evalTable.clear(Threads);
    g_evalHashInitialized = true;
};
void EvalHash_SetEnabled(bool enabled) {
    // An OFF interval may span a network or threshold change.  Never expose
    // entries from that interval when the runtime option is enabled again.
    if (enabled && !g_evalHashRequested && g_evalHashInitialized)
        EvalHash_Clear();
    g_evalHashRequested = enabled;
}
bool EvalHash_IsEnabled() { return g_evalHashRequested && g_evalHashInitialized; }

#if defined(EVAL_HASH_ATOMIC64)
bool EvalHash_Atomic64CodecSelftest() {
    constexpr Key key = UINT64_C(0x123456789abcdef0);
    for (int value = VALUE_MIN_EVAL; value <= VALUE_MAX_EVAL; ++value) {
        Value decoded = VALUE_NONE;
        std::uint8_t decodedFlags = 0;
        const std::uint8_t flags = EvalHashCodec::kSignalMask;
        const auto packed = EvalHashCodec::pack(key, static_cast<Value>(value), flags);
        if (!packed || !EvalHashCodec::unpack(key, packed, decoded, &decodedFlags)
            || decoded != value || decodedFlags != flags)
            return false;
    }
    Value ignored = VALUE_NONE;
    return !EvalHashCodec::unpack(key ^ UINT64_C(0x100000000),
                                  EvalHashCodec::pack(key, Value(0)), ignored);
}
#endif
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_LCA_LMR)
void EvalHash_SetLcaLmrThreshold(int threshold) {
    const int previous = g_evalHashLcaThreshold.exchange(threshold, std::memory_order_relaxed);
    if (previous != threshold && g_evalHashInitialized)
        EvalHash_Clear();
}
#endif

#if defined(MEASURE_EVAL_HASH_BENCHMARK)
namespace {
struct EvalHashDiagnosticCounters {
    std::atomic<std::uint64_t> evaluate_calls{0};
    std::atomic<std::uint64_t> accumulator_score_hits{0};
    std::atomic<std::uint64_t> probes{0};
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> misses{0};
    std::atomic<std::uint64_t> stores{0};
    std::atomic<std::uint64_t> compute_score_calls{0};
    std::atomic<std::uint64_t> transform_calls{0};
    std::atomic<std::uint64_t> propagate_calls{0};
    std::atomic<std::uint64_t> accumulator_already_computed{0};
    std::atomic<std::uint64_t> accumulator_incremental_updates{0};
    std::atomic<std::uint64_t> accumulator_refreshes{0};
    std::atomic<std::uint64_t> complex_router_calls{0};
    std::atomic<std::uint64_t> complex_fm_calls{0};
    std::atomic<std::uint64_t> complex_phase_calls{0};
    std::atomic<std::uint64_t> complex_cross_calls{0};
    std::atomic<std::uint64_t> complex_lca_calls{0};
    std::atomic<std::uint64_t> verified_hits{0};
    std::atomic<std::uint64_t> score_mismatches{0};
    std::atomic<std::uint64_t> score_abs_diff_sum{0};
    std::atomic<std::uint64_t> score_abs_diff_max{0};
    std::atomic<std::uint64_t> cached_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> incremental_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> ft_main_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> halfka_fm_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> ksdg3_fm_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> router_bucket_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> continuity_updates{0};
    std::atomic<std::uint64_t> continuity_refreshes{0};
    std::atomic<std::uint64_t> continuity_ns{0};
    std::atomic<std::uint64_t> signal_mismatches{0};
    std::atomic<std::uint64_t> signal_vs_refresh_mismatches{0};
    std::atomic<std::uint64_t> router_flag_hits{0};
    std::atomic<std::uint64_t> lca_flag_hits{0};
    std::atomic<std::uint64_t> cross_flag_hits{0};
};
EvalHashDiagnosticCounters g_evalHashDiagnostic;
std::atomic<bool> g_evalHashDiagnosticEnabled{true};
#if defined(EVAL_HASH_VERIFY_HITS)
// T1 verifier only: distinguish a packed-tag false hit from the same position
// being stored with a path-dependent incremental score.
struct EvalHashShadowValue {
    int score; int material; int game_ply;
    bool current_accumulation; bool parent_accumulation;
};
std::unordered_map<std::uint64_t, EvalHashShadowValue> g_evalHashShadowScores;

std::uint64_t debug_hash_bytes(const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

template <typename T>
void debug_array_diff(std::ostream& out, const char* label,
                      const T* left, const T* right, const std::size_t count) {
    std::size_t differences = 0;
    std::size_t first = count;
    std::int64_t max_abs = 0;
    std::int64_t sum_abs = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (left[i] == right[i]) continue;
        if (first == count) first = i;
        ++differences;
        const auto delta = static_cast<std::int64_t>(left[i])
                         - static_cast<std::int64_t>(right[i]);
        const auto magnitude = std::llabs(delta);
        max_abs = std::max(max_abs, magnitude);
        sum_abs += magnitude;
    }
    out << ' ' << label << "_diff_count=" << differences
        << ' ' << label << "_first="
        << (first == count ? -1 : static_cast<std::int64_t>(first))
        << ' ' << label << "_max_abs=" << max_abs
        << ' ' << label << "_sum_abs=" << sum_abs
        << ' ' << label << "_left_hash=0x" << std::hex
        << debug_hash_bytes(left, count * sizeof(T))
        << ' ' << label << "_right_hash=0x"
        << debug_hash_bytes(right, count * sizeof(T)) << std::dec;
}

void debug_accumulator_diff(
  std::ostream& out, const NNUE::Accumulator& incremental,
  const NNUE::Accumulator& scratch) {
    const auto* incremental_main = &incremental.accumulation[0][0][0];
    const auto* scratch_main = &scratch.accumulation[0][0][0];
    constexpr auto main_count = sizeof(incremental.accumulation)
                              / sizeof(*incremental_main);
    debug_array_diff(out, "ft_main", incremental_main, scratch_main,
                     main_count);
    debug_array_diff(out, "fm_factors",
      reinterpret_cast<const std::int64_t*>(incremental.factors),
      reinterpret_cast<const std::int64_t*>(scratch.factors),
      sizeof(incremental.factors) / sizeof(std::int64_t));
    for (int perspective = 0; perspective < 2; ++perspective) {
        const auto& il = incremental.factors[perspective];
        const auto& sl = scratch.factors[perspective];
        const std::string prefix = "p" + std::to_string(perspective);
        debug_array_diff(out, (prefix + "_halfka_sum_v").c_str(),
                         il.halfka.sum_v, sl.halfka.sum_v, 32);
        debug_array_diff(out, (prefix + "_halfka_sum_v2").c_str(),
                         il.halfka.sum_v2, sl.halfka.sum_v2, 32);
        debug_array_diff(out, (prefix + "_ksdg_sum_v").c_str(),
                         il.ksdg.sum_v, sl.ksdg.sum_v, 32);
        debug_array_diff(out, (prefix + "_ksdg_sum_v2").c_str(),
                         il.ksdg.sum_v2, sl.ksdg.sum_v2, 32);
    }
}

void debug_state_chain(std::ostream& out, const Position& pos) {
    auto* state = pos.state();
    for (int depth = 0; state && depth < 16; ++depth, state = state->previous) {
        out << "\nstate_chain depth=" << depth
            << " ptr=" << static_cast<const void*>(state)
            << " prev=" << static_cast<const void*>(state->previous)
            << " key=0x" << std::hex << static_cast<std::uint64_t>(state->key())
            << " move_raw=0x" << state->accumulator.debug_move_raw << std::dec
            << " game_ply=" << state->accumulator.debug_game_ply
            << " null=" << state->accumulator.debug_was_null_move
            << " source=" << static_cast<int>(
                 state->accumulator.debug_accumulator_source)
            << " computed_acc=" << state->accumulator.computed_accumulation
            << " computed_score=" << state->accumulator.computed_score;
#if defined(USE_EVAL_LIST)
        out << " dirty_num=" << state->dirtyPiece.dirty_num;
        for (int i = 0; i < state->dirtyPiece.dirty_num; ++i)
            out << " dirty" << i << "_piece_no=" << state->dirtyPiece.pieceNo[i]
                << " dirty" << i << "_old_b="
                << state->dirtyPiece.changed_piece[i].old_piece.from[BLACK]
                << " dirty" << i << "_old_w="
                << state->dirtyPiece.changed_piece[i].old_piece.from[WHITE]
                << " dirty" << i << "_new_b="
                << state->dirtyPiece.changed_piece[i].new_piece.from[BLACK]
                << " dirty" << i << "_new_w="
                << state->dirtyPiece.changed_piece[i].new_piece.from[WHITE];
#endif
    }
}
#endif
inline void diag_inc(std::atomic<std::uint64_t>& value) {
    value.fetch_add(1, std::memory_order_relaxed);
}
}

void EvalHash_DiagnosticBeforeTransform(const Position& pos, bool refresh) {
    auto& d = g_evalHashDiagnostic;
    diag_inc(d.transform_calls);
    const auto* now = pos.state();
    if (!refresh && now->accumulator.computed_accumulation)
        diag_inc(d.accumulator_already_computed);
    else if (!refresh && now->previous && now->previous->accumulator.computed_accumulation)
        diag_inc(d.accumulator_incremental_updates);
    else
        diag_inc(d.accumulator_refreshes);
}

void EvalHash_DiagnosticOnPropagate() {
    diag_inc(g_evalHashDiagnostic.propagate_calls);
}

void EvalHash_DiagnosticOnComplexRouter() {
    diag_inc(g_evalHashDiagnostic.complex_router_calls);
}

void EvalHash_DiagnosticOnComplexNetwork() {
    auto& d = g_evalHashDiagnostic;
    // These paths are each evaluated once by the Complex network Propagate.
    diag_inc(d.complex_fm_calls);
    diag_inc(d.complex_phase_calls);
    diag_inc(d.complex_cross_calls);
    diag_inc(d.complex_lca_calls);
}

#if defined(EVAL_HASH_VERIFY_HITS)
Value EvalHash_DiagnosticVerifyHit(const Position& pos, Value cached,
                                   std::uint8_t cached_flags) {
    // The hit path has already materialized the FT/FM accumulator. Force only
    // the downstream score/signals to be recomputed, then compare a complete
    // refresh as an independent oracle.
    pos.state()->accumulator.computed_score = false;
    const Value fresh = NNUE::ComputeScore(pos);
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_ROUTER_LMR)
    const auto fresh_flags = evalhash_signal_flags(
      pos.state()->accumulator.nnue_router_lmr_signal);
#endif
    // The scratch oracle writes into the production StateInfo accumulator.
    // Preserve the complete incremental state, not only score/signal: otherwise
    // a diagnostic hit changes the parent used by descendants and can create
    // the very multi-ply divergence that this verifier is meant to detect.
    const auto incremental_accumulator = pos.state()->accumulator;
#if !defined(NNUE_HALFKAHM2_SIMPLE)
    NNUE::EvalHashDebugRefreshAccumulatorFromScratch(pos);
    const Value refreshed = NNUE::ComputeScore(pos);
#else
    const Value refreshed = NNUE::ComputeScore(pos, true);
#endif
    const auto scratch_accumulator = pos.state()->accumulator;
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_ROUTER_LMR)
    const auto refresh_flags = evalhash_signal_flags(
      pos.state()->accumulator.nnue_router_lmr_signal);
#endif
    auto& d = g_evalHashDiagnostic;
    if (std::memcmp(incremental_accumulator.accumulation,
                    scratch_accumulator.accumulation,
                    sizeof(incremental_accumulator.accumulation)) != 0)
        diag_inc(d.ft_main_vs_refresh_mismatches);
    bool halfka_fm_diff = false;
    bool ksdg3_fm_diff = false;
    for (const Color perspective : {BLACK, WHITE}) {
        const auto& incremental = incremental_accumulator.factors[perspective];
        const auto& scratch = scratch_accumulator.factors[perspective];
        halfka_fm_diff |=
            std::memcmp(&incremental.halfka, &scratch.halfka,
                        sizeof(incremental.halfka)) != 0;
        ksdg3_fm_diff |=
            std::memcmp(&incremental.ksdg, &scratch.ksdg,
                        sizeof(incremental.ksdg)) != 0;
    }
    if (halfka_fm_diff)
        diag_inc(d.halfka_fm_vs_refresh_mismatches);
    if (ksdg3_fm_diff)
        diag_inc(d.ksdg3_fm_vs_refresh_mismatches);
#if defined(ENABLE_NNUE_SIGNAL_LOG)
    if (incremental_accumulator.nnue_signal.selected_bucket
        != scratch_accumulator.nnue_signal.selected_bucket)
        diag_inc(d.router_bucket_vs_refresh_mismatches);
#endif
    pos.state()->accumulator = incremental_accumulator;
    diag_inc(d.verified_hits);
    const auto diff = static_cast<std::uint64_t>(std::abs(static_cast<int>(fresh-cached)));
    d.score_abs_diff_sum.fetch_add(diff, std::memory_order_relaxed);
    auto old = d.score_abs_diff_max.load(std::memory_order_relaxed);
    while (old < diff && !d.score_abs_diff_max.compare_exchange_weak(
             old, diff, std::memory_order_relaxed)) {}
    if (fresh != cached) {
        const auto mismatch_index = d.score_mismatches.fetch_add(1, std::memory_order_relaxed);
        if (mismatch_index < 20) {
            std::cout << "evalhash_score_mismatch index " << mismatch_index
                      << " key 0x" << std::hex << static_cast<std::uint64_t>(pos.state()->key())
                      << std::dec << " cached " << static_cast<int>(cached)
                      << " fresh " << static_cast<int>(fresh)
                      << " refresh " << static_cast<int>(refreshed)
                      << " material " << pos.state()->materialValue
                      << " game_ply " << pos.game_ply()
                      << " sfen " << pos.sfen() << std::endl;
        }
    }
    if (cached != refreshed) {
        const auto mismatch_index = d.cached_vs_refresh_mismatches.fetch_add(
          1, std::memory_order_relaxed);
        {
            const auto shadow_it = g_evalHashShadowScores.find(
              static_cast<std::uint64_t>(pos.state()->key()));
            const int shadow_score = shadow_it == g_evalHashShadowScores.end()
                                   ? 999999 : shadow_it->second.score;
            const int shadow_material = shadow_it == g_evalHashShadowScores.end()
                                      ? 999999 : shadow_it->second.material;
            const int shadow_ply = shadow_it == g_evalHashShadowScores.end()
                                 ? -1 : shadow_it->second.game_ply;
            const int shadow_current = shadow_it == g_evalHashShadowScores.end()
                                     ? -1 : shadow_it->second.current_accumulation;
            const int shadow_parent = shadow_it == g_evalHashShadowScores.end()
                                    ? -1 : shadow_it->second.parent_accumulation;
            std::cout << "evalhash_cached_refresh_mismatch index " << mismatch_index
                      << " key 0x" << std::hex
                      << static_cast<std::uint64_t>(pos.state()->key())
                      << std::dec << " cached " << static_cast<int>(cached)
                      << " fresh " << static_cast<int>(fresh)
                      << " refresh " << static_cast<int>(refreshed)
                      << " shadow " << shadow_score
                      << " shadow_material " << shadow_material
                      << " shadow_ply " << shadow_ply
                      << " shadow_current_acc " << shadow_current
                      << " shadow_parent_acc " << shadow_parent
                      << " material " << pos.state()->materialValue
                      << " game_ply " << pos.game_ply()
                      << " sfen " << pos.sfen() << std::endl;
            std::ostringstream details;
            details << "evalhash_incremental_scratch_detail index="
                    << mismatch_index
                    << " first_layer=";
            const bool main_diff = std::memcmp(
              incremental_accumulator.accumulation,
              scratch_accumulator.accumulation,
              sizeof(incremental_accumulator.accumulation)) != 0;
            const bool factor_diff = std::memcmp(
              incremental_accumulator.factors, scratch_accumulator.factors,
              sizeof(incremental_accumulator.factors)) != 0;
            details << (main_diff ? "FT_Main" : factor_diff ? "FM" : "Post_FT")
                    << " incremental_source=" << static_cast<int>(
                         incremental_accumulator.debug_accumulator_source)
                    << " scratch_source=" << static_cast<int>(
                         scratch_accumulator.debug_accumulator_source);
            debug_accumulator_diff(details, incremental_accumulator,
                                   scratch_accumulator);
#if defined(ENABLE_NNUE_SIGNAL_LOG)
            const auto& is = incremental_accumulator.nnue_signal;
            const auto& ss = scratch_accumulator.nnue_signal;
            details << " incremental_bucket=" << is.selected_bucket
                    << " scratch_bucket=" << ss.selected_bucket
                    << " incremental_router_top1=" << is.router_top1_logit
                    << " scratch_router_top1=" << ss.router_top1_logit
                    << " incremental_router_margin=" << is.router_margin
                    << " scratch_router_margin=" << ss.router_margin
                    << " incremental_phase0=" << is.phase_scale[0]
                    << " scratch_phase0=" << ss.phase_scale[0]
                    << " incremental_cross_max=" << static_cast<int>(is.cross_abs_max)
                    << " scratch_cross_max=" << static_cast<int>(ss.cross_abs_max)
                    << " incremental_lca_sum=" << is.lca_abs_delta_sum
                    << " scratch_lca_sum=" << ss.lca_abs_delta_sum
                    << " incremental_deep=" << is.deep_output
                    << " scratch_deep=" << ss.deep_output
                    << " incremental_bypass=" << is.bypass_output
                    << " scratch_bypass=" << ss.bypass_output;
#endif
            if (mismatch_index < 32) {
                debug_state_chain(details, pos);
                NNUE::EvalHashDebugDescribeFeatures(pos, details);
            }
            std::cout << details.str() << std::endl;
        }
    }
    if (fresh != refreshed) diag_inc(d.incremental_vs_refresh_mismatches);
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_ROUTER_LMR)
    if (cached_flags != fresh_flags) diag_inc(d.signal_mismatches);
    if (cached_flags != refresh_flags) diag_inc(d.signal_vs_refresh_mismatches);
    // Restore the exact hit contract after the verifier's recomputations.
    // The full accumulator assignment above already restored the incremental
    // FT/FM bytes.  Only publish the cached boundary-equivalent signal.
    auto restored = evalhash_restore_signal(cached_flags);
    auto& accumulator = pos.state()->accumulator;
    accumulator.score = cached;
    accumulator.computed_score = true;
    accumulator.nnue_router_lmr_signal = restored;
    NNUE::SetLastNnueRouterLmrSignal(&accumulator.nnue_router_lmr_signal);
#endif
    return cached;
}
#endif

void EvalHash_SetDiagnosticEnabled(bool enabled) {
    g_evalHashDiagnosticEnabled.store(enabled, std::memory_order_relaxed);
}

void EvalHash_DiagnosticReset() {
    auto& d = g_evalHashDiagnostic;
    d.evaluate_calls=0; d.accumulator_score_hits=0; d.probes=0;
    d.hits=0; d.misses=0; d.stores=0; d.compute_score_calls=0;
    d.transform_calls=0; d.propagate_calls=0;
    d.accumulator_already_computed=0; d.accumulator_incremental_updates=0;
    d.accumulator_refreshes=0;
    d.complex_router_calls=0; d.complex_fm_calls=0;
    d.complex_phase_calls=0; d.complex_cross_calls=0; d.complex_lca_calls=0;
    d.verified_hits=0; d.score_mismatches=0; d.score_abs_diff_sum=0; d.score_abs_diff_max=0;
    d.cached_vs_refresh_mismatches=0; d.incremental_vs_refresh_mismatches=0;
    d.ft_main_vs_refresh_mismatches=0;
    d.halfka_fm_vs_refresh_mismatches=0;
    d.ksdg3_fm_vs_refresh_mismatches=0;
    d.router_bucket_vs_refresh_mismatches=0;
    d.continuity_updates=0; d.continuity_refreshes=0; d.continuity_ns=0;
    d.signal_mismatches=0; d.signal_vs_refresh_mismatches=0;
    d.router_flag_hits=0; d.lca_flag_hits=0; d.cross_flag_hits=0;
#if defined(EVAL_HASH_VERIFY_HITS)
    g_evalHashShadowScores.clear();
#endif
}

void EvalHash_DiagnosticReport() {
    const auto& d = g_evalHashDiagnostic;
    const auto probes=d.probes.load(std::memory_order_relaxed);
    const auto hits=d.hits.load(std::memory_order_relaxed);
    std::cout << "[EvalHash Diagnostic]" << std::endl
              << "enabled " << g_evalHashDiagnosticEnabled.load(std::memory_order_relaxed) << std::endl
              << "runtime_requested " << g_evalHashRequested << std::endl
              << "table_initialized " << g_evalHashInitialized << std::endl
              << "runtime_effective " << EvalHash_IsEnabled() << std::endl
              << "entry_size " <<
#if defined(EVAL_HASH_ATOMIC64)
                 sizeof(std::atomic<std::uint64_t>) << std::endl
              << "tag_bits " << kAtomicTagBits << std::endl
#else
                 sizeof(ScoreKeyValue) << std::endl
#endif
              << "entry_count " << g_evalTable.entry_count() << std::endl
              << "table_bytes " << g_evalTable.byte_size() << std::endl
              << "evaluate_calls " << d.evaluate_calls.load() << std::endl
              << "accumulator_score_hits " << d.accumulator_score_hits.load() << std::endl
              << "probes " << probes << std::endl
              << "hits " << hits << std::endl
              << "misses " << d.misses.load() << std::endl
              << "hit_rate " << std::setprecision(10)
              << (probes ? static_cast<double>(hits)/probes : 0.0) << std::endl
              << "stores " << d.stores.load() << std::endl
              << "compute_score_calls " << d.compute_score_calls.load() << std::endl
              << "transform_calls " << d.transform_calls.load() << std::endl
              << "propagate_calls " << d.propagate_calls.load() << std::endl
              << "accumulator_already_computed " << d.accumulator_already_computed.load() << std::endl
              << "accumulator_incremental_updates " << d.accumulator_incremental_updates.load() << std::endl
              << "accumulator_refreshes " << d.accumulator_refreshes.load() << std::endl;
    std::cout << "complex_router_calls " << d.complex_router_calls.load() << std::endl
              << "complex_fm_calls " << d.complex_fm_calls.load() << std::endl
              << "complex_phase_calls " << d.complex_phase_calls.load() << std::endl
              << "complex_cross_calls " << d.complex_cross_calls.load() << std::endl
              << "complex_lca_calls " << d.complex_lca_calls.load() << std::endl;
    std::cout << "verified_hits " << d.verified_hits.load() << std::endl
              << "score_mismatches " << d.score_mismatches.load() << std::endl
              << "score_abs_diff_sum " << d.score_abs_diff_sum.load() << std::endl
              << "score_abs_diff_max " << d.score_abs_diff_max.load() << std::endl;
    std::cout << "cached_vs_refresh_mismatches " << d.cached_vs_refresh_mismatches.load() << std::endl
              << "incremental_vs_refresh_mismatches "
              << d.incremental_vs_refresh_mismatches.load() << std::endl
              << "ft_main_vs_refresh_mismatches "
              << d.ft_main_vs_refresh_mismatches.load() << std::endl
              << "halfka_fm_vs_refresh_mismatches "
              << d.halfka_fm_vs_refresh_mismatches.load() << std::endl
              << "ksdg3_fm_vs_refresh_mismatches "
              << d.ksdg3_fm_vs_refresh_mismatches.load() << std::endl
              << "router_bucket_vs_refresh_mismatches "
              << d.router_bucket_vs_refresh_mismatches.load() << std::endl
              << "continuity_updates " << d.continuity_updates.load() << std::endl
              << "continuity_refreshes " << d.continuity_refreshes.load() << std::endl
              << "continuity_ns " << d.continuity_ns.load() << std::endl
              << "signal_mismatches " << d.signal_mismatches.load() << std::endl
              << "signal_vs_refresh_mismatches " << d.signal_vs_refresh_mismatches.load() << std::endl
              << "router_flag_hits " << d.router_flag_hits.load() << std::endl
              << "lca_flag_hits " << d.lca_flag_hits.load() << std::endl
              << "cross_flag_hits " << d.cross_flag_hits.load() << std::endl;
}

void EvalHash_Microbench(std::uint64_t repetitions) {
    using Clock=std::chrono::steady_clock;
    volatile std::uint64_t checksum=0;
    auto measure=[&](const char* name, auto&& operation) {
        const auto begin=Clock::now();
        for (std::uint64_t i=0;i<repetitions;++i) checksum ^= operation(i);
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
        std::cout << "micro_" << name << "_ns "
                  << static_cast<double>(ns)/repetitions << std::endl;
    };
    constexpr Key base=UINT64_C(0x9e3779b97f4a7c15);
#if defined(EVAL_HASH_ATOMIC64)
    for (std::uint64_t i=0;i<4096;++i) {
        const Key key=base+i*UINT64_C(0x100000001b3);
        g_evalTable[key].store(evalhash_pack(key, static_cast<Value>(i&1023)), std::memory_order_relaxed);
    }
    measure("hit",[&](std::uint64_t i){const Key key=base+(i&4095)*UINT64_C(0x100000001b3);Value v;return evalhash_unpack(key,g_evalTable[key].load(std::memory_order_relaxed),v)?static_cast<std::uint64_t>(v):0;});
    measure("miss",[&](std::uint64_t i){const Key key=(base^UINT64_C(0xd1b54a32d192ed03))+(i&4095)*UINT64_C(0x100000001b3);Value v;return evalhash_unpack(key,g_evalTable[key].load(std::memory_order_relaxed),v)?static_cast<std::uint64_t>(v):0;});
    measure("store",[&](std::uint64_t i){const Key key=base+(i&4095)*UINT64_C(0x100000001b3);g_evalTable[key].store(evalhash_pack(key,static_cast<Value>(i&1023)),std::memory_order_relaxed);return i;});
#else
    for (std::uint64_t i=0;i<4096;++i) {
        const Key key=base+i*UINT64_C(0x100000001b3);
        ScoreKeyValue entry;
        entry.key=key; entry.score=static_cast<std::uint64_t>(i);
        entry.encode();*g_evalTable[key]=entry;
    }
    measure("hit",[&](std::uint64_t i){const Key key=base+(i&4095)*UINT64_C(0x100000001b3);auto e=*g_evalTable[key];e.decode();return e.key==key?e.score:0;});
    measure("miss",[&](std::uint64_t i){const Key key=(base^UINT64_C(0xd1b54a32d192ed03))+(i&4095)*UINT64_C(0x100000001b3);auto e=*g_evalTable[key];e.decode();return e.key==key?e.score:0;});
    measure("store",[&](std::uint64_t i){const Key key=base+(i&4095)*UINT64_C(0x100000001b3);ScoreKeyValue e;e.key=key;e.score=i;e.encode();*g_evalTable[key]=e;return i;});
#endif
    std::cout << "micro_checksum " << checksum << std::endl;
}

#if defined(EVAL_HASH_ATOMIC64)
void EvalHash_Atomic64Selftest(std::uint64_t collisionTrials) {
    std::uint64_t errors=0;
    constexpr Key key=UINT64_C(0x123456789abcdef0);
    for (int value=VALUE_MIN_EVAL; value<=VALUE_MAX_EVAL; ++value) {
        Value decoded=VALUE_NONE;
        const auto packed=evalhash_pack(key,value);
        if (!evalhash_unpack(key,packed,decoded) || decoded!=value || packed==0) ++errors;
    }
    std::uint64_t state=UINT64_C(0x9e3779b97f4a7c15), collisions=0;
    constexpr unsigned indexBits=17; // 1 MiB packed64 production candidate.
    constexpr std::uint64_t indexMask=(UINT64_C(1)<<indexBits)-1;
    for (std::uint64_t i=0;i<collisionTrials;++i) {
        state ^= state<<7; state ^= state>>9; state ^= state<<8;
        const Key a=state;
        state ^= state<<7; state ^= state>>9; state ^= state<<8;
        const Key b=(state&~indexMask)|(a&indexMask);
        collisions += evalhash_tag(a)==evalhash_tag(b);
    }
    std::cout << "atomic64_score_roundtrip_errors " << errors << std::endl
              << "atomic64_value_min " << VALUE_MIN_EVAL << std::endl
              << "atomic64_value_max " << VALUE_MAX_EVAL << std::endl
              << "atomic64_tag_bits " << kAtomicTagBits << std::endl
              << "atomic64_collision_trials " << collisionTrials << std::endl
              << "atomic64_tag_collisions " << collisions << std::endl;
}
#endif
#endif

// prefetchする関数も用意しておく。
void prefetch_evalhash(const Key key) {
    constexpr auto mask = ~((u64)0x1f);
#if defined(EVAL_HASH_ATOMIC64)
    prefetch((void*)((u64)&g_evalTable[key] & mask));
#else
    prefetch((void*)((u64)g_evalTable[key] & mask));
#endif
}
#endif

// 評価関数ファイルを読み込む
void load_eval() {
    // 評価関数パラメーターを読み込み済みであるなら帰る。
    if (eval_loaded)
        return;

	// 初期化もここでやる。
	NNUE::Initialize();

#if defined(EVAL_LEARN)
    if (!Options["SkipLoadingEval"])
#endif
    {
        const std::string dir_name = Options["EvalDir"];
    #if !defined(__EMSCRIPTEN__)
		const std::string file_name = NNUE::kFileName;
#else
		// WASM
        const std::string file_name = Options["EvalFile"];
    #endif
        const Tools::Result result = [&] {
            if (dir_name != "<internal>") {
                auto abs_eval_path = Path::Combine(Directory::GetBinaryFolder(), dir_name);
                const std::string file_path = Path::Combine(abs_eval_path, file_name);
                std::ifstream stream(file_path, std::ios::binary);
                sync_cout << "info string loading eval file : " << file_path << sync_endl;
				if (!stream.is_open())
					return Tools::Result(Tools::ResultCode::FileNotFound);

                return NNUE::ReadParameters(stream);
            }
            else {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };

			    const auto embedded = get_embedded(/* embeddedType */);

                MemoryBuffer buffer(
                              const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                              size_t(embedded.size));

                std::istream stream(&buffer);
                sync_cout << "info string loading eval file : <internal>" << sync_endl;

                return NNUE::ReadParameters(stream);
            }
        }();

        //      ASSERT(result);

        if (result.is_not_ok())
        {
            // 読み込みエラーのとき終了してくれないと困る。
            sync_cout << "Error! : failed to read " << file_name << " : " << result.to_string() << sync_endl;
            Tools::exit();
        }

		// 評価関数ファイルの読み込みが完了した。
		eval_loaded = true;
    }
}


// 評価関数。差分計算ではなく全計算する。
// Position::set()で一度だけ呼び出される。(以降は差分計算)
// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
// なので、この関数の最適化は頑張らない。
Value compute_eval(const Position& pos) {
    return NNUE::ComputeScore(pos, true);
}

// 評価関数
Value evaluate(const Position& pos) {
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    diag_inc(g_evalHashDiagnostic.evaluate_calls);
#endif
    const auto& accumulator = pos.state()->accumulator;
    if (accumulator.computed_score) {
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        diag_inc(g_evalHashDiagnostic.accumulator_score_hits);
#endif
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        NNUE::SetLastNnueSignalAccess(
          NNUE::NnueSignalEvalSource::AccumulatorCached,
          accumulator.nnue_signal.valid ? &accumulator.nnue_signal : nullptr);
#endif
#if defined(USE_NNUE_ROUTER_LMR)
        NNUE::SetLastNnueRouterLmrSignal(
          accumulator.nnue_router_lmr_signal.valid ? &accumulator.nnue_router_lmr_signal : nullptr);
#endif
        return accumulator.score;
    }

#if defined(USE_GLOBAL_OPTIONS) && !defined(EVAL_HASH_RUNTIME_OPTION_OVERRIDES_GLOBAL)
    // GlobalOptionsでeval hashを用いない設定になっているなら
    // eval hashへの照会をskipする。
    if (!GlobalOptions.use_eval_hash) {
        ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
        return NNUE::ComputeScore(pos);
    }
#endif

#if defined(USE_EVAL_HASH)
    // Runtime option; DISABLE_EVAL_HASH removes this branch at compile time.
    if (!EvalHash_IsEnabled())
        return NNUE::ComputeScore(pos);

    const Key key = pos.state()->key();
#if defined(EVAL_HASH_COMPLEX_SAFE)
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    const bool continuity_refresh =
      !pos.state()->accumulator.computed_accumulation
      && !(pos.state()->previous
           && pos.state()->previous->accumulator.computed_accumulation);
    const auto continuity_begin = std::chrono::steady_clock::now();
#endif
#if defined(EVAL_HASH_FORCE_REFRESH_ON_HIT)
    // Diagnostic candidate: canonicalize every probed state.
    NNUE::ForceRefreshAccumulator(pos);
#else
    NNUE::EnsureAccumulator(pos);
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    const auto continuity_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - continuity_begin).count();
    diag_inc(g_evalHashDiagnostic.continuity_updates);
    if (continuity_refresh) diag_inc(g_evalHashDiagnostic.continuity_refreshes);
    g_evalHashDiagnostic.continuity_ns.fetch_add(
      static_cast<std::uint64_t>(continuity_elapsed), std::memory_order_relaxed);
#endif
    const Key lookupKey = static_cast<Key64>(key)
                        ^ NNUE::AccumulatorFingerprint(pos);
#else
    const Key lookupKey = key;
#endif
#if !defined(EVAL_HASH_ATOMIC64)
    ScoreKeyValue entry;
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    if (g_evalHashDiagnosticEnabled.load(std::memory_order_relaxed)) {
#endif
    // evaluate hash tableにはあるかも。
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    diag_inc(g_evalHashDiagnostic.probes);
#endif
#if defined(EVAL_HASH_ATOMIC64)
    Value cachedScore;
    std::uint8_t cachedFlags = 0;
    const auto packed = g_evalTable[lookupKey].load(std::memory_order_relaxed);
    if (evalhash_unpack(lookupKey, packed, cachedScore, &cachedFlags)) {
#else
    entry = *g_evalTable[key]; entry.decode();
    if (entry.key == key) {
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        diag_inc(g_evalHashDiagnostic.hits);
#endif
        // あった！
#if defined(ENABLE_NNUE_SIGNAL_LOG)
        NNUE::SetLastNnueSignalAccess(NNUE::NnueSignalEvalSource::EvalHashHit);
#endif
#if defined(EVAL_HASH_MAINTAIN_ACCUMULATOR) && !defined(EVAL_HASH_COMPLEX_SAFE)
        // A cached score cannot replace the FT/FM state: child positions use
        // this node as the parent of their incremental accumulator chain.
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        const bool continuity_refresh =
          !pos.state()->accumulator.computed_accumulation
          && !(pos.state()->previous
               && pos.state()->previous->accumulator.computed_accumulation);
        const auto continuity_begin = std::chrono::steady_clock::now();
#endif
#if defined(EVAL_HASH_FORCE_REFRESH_ON_HIT)
        NNUE::ForceRefreshAccumulator(pos);
#else
        NNUE::EnsureAccumulator(pos);
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        const auto continuity_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - continuity_begin).count();
        diag_inc(g_evalHashDiagnostic.continuity_updates);
        if (continuity_refresh) diag_inc(g_evalHashDiagnostic.continuity_refreshes);
        g_evalHashDiagnostic.continuity_ns.fetch_add(
          static_cast<std::uint64_t>(continuity_elapsed), std::memory_order_relaxed);
#endif
#endif
#if defined(USE_NNUE_ROUTER_LMR)
#if defined(EVAL_HASH_COMPLEX_SAFE)
        auto restoredSignal = evalhash_restore_signal(cachedFlags);
        auto& mutableAccumulator = pos.state()->accumulator;
        mutableAccumulator.score = cachedScore;
        mutableAccumulator.computed_score = true;
        mutableAccumulator.nnue_router_lmr_signal = restoredSignal;
        NNUE::SetLastNnueRouterLmrSignal(
          &mutableAccumulator.nnue_router_lmr_signal);
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
        if (cachedFlags & kEvalHashRouterFlag) diag_inc(g_evalHashDiagnostic.router_flag_hits);
        if (cachedFlags & kEvalHashLcaFlag) diag_inc(g_evalHashDiagnostic.lca_flag_hits);
        if (cachedFlags & kEvalHashCrossFlag) diag_inc(g_evalHashDiagnostic.cross_flag_hits);
#endif
#else
        // Score-only variants intentionally expose the missing-signal behavior.
        NNUE::SetLastNnueRouterLmrSignal();
#endif
#endif
#if defined(EVAL_HASH_ATOMIC64)
#if defined(MEASURE_EVAL_HASH_BENCHMARK) && defined(EVAL_HASH_VERIFY_HITS)
        return EvalHash_DiagnosticVerifyHit(pos, cachedScore, cachedFlags);
#else
        return cachedScore;
#endif
#else
        return Value(entry.score);
#endif
    }
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    diag_inc(g_evalHashDiagnostic.misses);
    }
#endif
#endif

#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    diag_inc(g_evalHashDiagnostic.compute_score_calls);
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK) && defined(EVAL_HASH_VERIFY_HITS)
    const bool shadowCurrentAccumulation = pos.state()->accumulator.computed_accumulation;
    const bool shadowParentAccumulation = pos.state()->previous
      && pos.state()->previous->accumulator.computed_accumulation;
#endif
    Value score = NNUE::ComputeScore(pos);
#if defined(USE_EVAL_HASH)
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    if (g_evalHashDiagnosticEnabled.load(std::memory_order_relaxed)) {
#endif
    // せっかく計算したのでevaluate hash tableに保存しておく。
#if defined(EVAL_HASH_ATOMIC64)
    std::uint8_t storeFlags = 0;
#if defined(EVAL_HASH_COMPLEX_SAFE) && defined(USE_NNUE_ROUTER_LMR)
    storeFlags = evalhash_signal_flags(
      pos.state()->accumulator.nnue_router_lmr_signal);
#endif
    g_evalTable[lookupKey].store(evalhash_pack(lookupKey, score, storeFlags), std::memory_order_relaxed);
#if defined(MEASURE_EVAL_HASH_BENCHMARK) && defined(EVAL_HASH_VERIFY_HITS)
    g_evalHashShadowScores[static_cast<std::uint64_t>(key)] = {
      static_cast<int>(score), pos.state()->materialValue, pos.game_ply(),
      shadowCurrentAccumulation, shadowParentAccumulation};
#endif
#else
    entry.key = key; entry.score = score; entry.encode(); *g_evalTable[key] = entry;
#endif
#if defined(MEASURE_EVAL_HASH_BENCHMARK)
    diag_inc(g_evalHashDiagnostic.stores);
    }
#endif
#endif

    return score;
}

// 差分計算ができるなら進める
void evaluate_with_no_return(const Position& pos) {
    NNUE::UpdateAccumulatorIfPossible(pos);
}

// 現在の局面の評価値の内訳を表示する
void print_eval_stat(Position& /*pos*/) {
    std::cout << "--- EVAL STAT: not implemented" << std::endl;
}

} // namespace Eval
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
