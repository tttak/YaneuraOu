// Experiment 85: canonical HalfKA_hm2/no-DG SFNN baseline.
#ifndef NNUE_HALFKAHM2_SIMPLE_1536_H_INCLUDED
#define NNUE_HALFKAHM2_SIMPLE_1536_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_ka_hm2.h"
#include "../layers/affine_transform_explicit.h"
#include "../layers/affine_transform_sparse_input_explicit.h"
#include "../layers/clipped_relu_explicit.h"
#include "../layers/sqr_clipped_relu.h"
#include <cstring>

namespace YaneuraOu::Eval::NNUE {

using RawFeatures = Features::FeatureSet<
    Features::HalfKA_hm2<Features::Side::kFriend>>;
constexpr IndexType kTransformedFeatureDimensions = 1536;
constexpr int LayerStacks = 9;
constexpr IndexType kInputDims = 1536;
constexpr IndexType kHidden1Dims = 15;
constexpr IndexType kHidden2Dims = 32;

struct Network {
    Layers::AffineTransformSparseInputExplicit<kInputDims, 16> fc_0;
    Layers::ClippedReLUExplicit<16> ac_0;
    Layers::SqrClippedReLU<16> ac_sqr_0;
    Layers::AffineTransformExplicit<30, 32> fc_1;
    Layers::ClippedReLUExplicit<32> ac_1;
    Layers::AffineTransformExplicit<32, 1> fc_2;

    using OutputType = std::int32_t;
    static constexpr IndexType kOutputDimensions = 1;
    static constexpr std::uint32_t GetHashValue() {
        return (0x6333718Au ^ 0x484D3202u)
#if defined(NNUE_SIMPLE_PP3WIDE)
            ^ 0x50335031u
#endif
            ;
    }
    static std::string GetStructureString() {
#if defined(NNUE_SIMPLE_PP3WIDE)
        return "SFNN-1536-HalfKAHM2-NoDG-PP3WPL-v3";
#else
        return "SFNN-1536-HalfKAHM2-NoDG-v2";
#endif
    }
    Tools::Result ReadParameters(std::istream& stream) {
        const bool ok = fc_0.ReadParameters(stream).is_ok()
            && ac_0.ReadParameters(stream).is_ok()
            && ac_sqr_0.ReadParameters(stream).is_ok()
            && fc_1.ReadParameters(stream).is_ok()
            && ac_1.ReadParameters(stream).is_ok()
            && fc_2.ReadParameters(stream).is_ok();
        return ok ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }
    bool WriteParameters(std::ostream& stream) const {
        return fc_0.WriteParameters(stream) && ac_0.WriteParameters(stream)
            && ac_sqr_0.WriteParameters(stream) && fc_1.WriteParameters(stream)
            && ac_1.WriteParameters(stream) && fc_2.WriteParameters(stream);
    }
    struct alignas(kCacheLineSize) Buffer {
        alignas(kCacheLineSize) typename decltype(fc_0)::OutputBuffer fc0;
        alignas(kCacheLineSize) typename decltype(ac_0)::OutputBuffer ac0;
        alignas(kCacheLineSize) typename decltype(ac_sqr_0)::OutputType concat[
            CeilToMultiple<IndexType>(30, 32)];
        alignas(kCacheLineSize) typename decltype(fc_1)::OutputBuffer fc1;
        alignas(kCacheLineSize) typename decltype(ac_1)::OutputBuffer ac1;
        alignas(kCacheLineSize) typename decltype(fc_2)::OutputBuffer fc2;
    };
    static constexpr std::size_t kBufferSize = sizeof(Buffer);
    const OutputType* Propagate(const TransformedFeatureType* input,
                                char* storage) const {
        auto& b = *reinterpret_cast<Buffer*>(storage);
        fc_0.Propagate(input, b.fc0);
        ac_0.Propagate(b.fc0, b.ac0);
        ac_sqr_0.Propagate(b.fc0, b.concat);
        std::memcpy(b.concat + 15, b.ac0, 15 * sizeof(b.ac0[0]));
        fc_1.Propagate(b.concat, b.fc1);
        ac_1.Propagate(b.fc1, b.ac1);
        fc_2.Propagate(b.ac1, b.fc2);
        b.fc2[0] += b.fc0[15];
        return b.fc2;
    }
};

} // namespace YaneuraOu::Eval::NNUE
#endif
