#ifndef YANEURAOU_NNUE_KP_PROGRESS_FT_PROXY_H_INCLUDED
#define YANEURAOU_NNUE_KP_PROGRESS_FT_PROXY_H_INCLUDED

#include "../../config.h"

#if defined(USE_EXPERIMENTAL_KP_PROGRESS_FT_PROXY)
#include <cstdint>
#include <iosfwd>
#include "nnue_common.h"

namespace YaneuraOu::Eval::NNUE::NnueKpProgressFtProxy {
void observe(const TransformedFeatureType* transformed);
void benchmark(std::uint64_t repeats, std::ostream& out);
}
#endif
#endif
