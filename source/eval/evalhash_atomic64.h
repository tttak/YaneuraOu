#ifndef EVALHASH_ATOMIC64_H_INCLUDED
#define EVALHASH_ATOMIC64_H_INCLUDED

#include "../types.h"

#include <cstdint>

namespace YaneuraOu::Eval {

// One-word EvalHash publication contract.  A relaxed atomic load/store is
// sufficient: the entry does not publish any other memory, and readers either
// observe the complete old word or the complete new word.  A false hit can
// therefore only come from a truncated verification-tag collision.
template <unsigned SignalBits, bool StrongTagMix>
struct EvalHashAtomic64Codec {
    static_assert(SignalBits <= 3);
    static constexpr unsigned kScoreBits   = 16;
    static constexpr unsigned kPayloadBits = kScoreBits + SignalBits;
    static constexpr unsigned kTagBits     = 64 - kPayloadBits;
    static constexpr std::uint64_t kTagMask =
      (UINT64_C(1) << kTagBits) - 1;
    static constexpr std::uint8_t kSignalMask =
      static_cast<std::uint8_t>((UINT64_C(1) << SignalBits) - 1);

    static inline std::uint64_t tag(Key key) {
        std::uint64_t mixed = static_cast<std::uint64_t>(key);
        if constexpr (StrongTagMix) {
            mixed ^= mixed >> 30;
            mixed *= UINT64_C(0xbf58476d1ce4e5b9);
            mixed ^= mixed >> 27;
            mixed *= UINT64_C(0x94d049bb133111eb);
            mixed ^= mixed >> 31;
        } else {
            mixed = (mixed ^ (mixed >> 32)) >> 16;
        }
        return mixed & kTagMask;
    }

    static inline std::uint16_t encode_score(Value score) {
        ASSERT_LV3(score != VALUE_NONE);
        ASSERT_LV3(score >= VALUE_MIN_EVAL && score <= VALUE_MAX_EVAL);
        const auto code = static_cast<std::uint16_t>(score + 32768);
        ASSERT_LV3(code != 0);  // packed zero is the empty-table sentinel.
        return code;
    }

    static inline Value decode_score(std::uint16_t code) {
        return static_cast<Value>(static_cast<int>(code) - 32768);
    }

    static inline std::uint64_t pack(Key key, Value score,
                                     std::uint8_t signals = 0) {
        ASSERT_LV3((signals & ~kSignalMask) == 0);
        return (tag(key) << kPayloadBits)
             | (static_cast<std::uint64_t>(signals & kSignalMask) << kScoreBits)
             | encode_score(score);
    }

    static inline bool unpack(Key key, std::uint64_t packed, Value& score,
                              std::uint8_t* signals = nullptr) {
        if (!packed || (packed >> kPayloadBits) != tag(key))
            return false;
        score = decode_score(static_cast<std::uint16_t>(packed));
        if (signals)
            *signals = static_cast<std::uint8_t>(
              (packed >> kScoreBits) & kSignalMask);
        return true;
    }
};

using SimpleEvalHashCodec  = EvalHashAtomic64Codec<0, false>;  // 48-bit tag.
using ComplexEvalHashCodec = EvalHashAtomic64Codec<3, true>;   // 45-bit tag.

static_assert(SimpleEvalHashCodec::kTagBits == 48);
static_assert(ComplexEvalHashCodec::kTagBits == 45);

}  // namespace YaneuraOu::Eval

#endif  // EVALHASH_ATOMIC64_H_INCLUDED
