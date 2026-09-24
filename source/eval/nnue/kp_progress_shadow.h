#ifndef YANEURAOU_NNUE_KP_PROGRESS_SHADOW_H_INCLUDED
#define YANEURAOU_NNUE_KP_PROGRESS_SHADOW_H_INCLUDED

#include "../../config.h"

#if defined(USE_EXPERIMENTAL_KP_PROGRESS_SHADOW)

#include <cstdint>
#include <iosfwd>
#include <string>

#include "features/index_list.h"

namespace YaneuraOu { class Position; }

namespace YaneuraOu::Eval::NNUE::NnueKpProgressShadow {

struct ProgressShadowResult {
  double logit = 0.0;
  int bucket3 = 0;
};

bool load(const std::string& path, std::ostream* error = nullptr);
bool load_from_environment(std::ostream* error = nullptr);
bool enabled();
ProgressShadowResult fresh(const Position& pos);
ProgressShadowResult incremental(const Position& pos);
ProgressShadowResult fixed_fresh(const Position& pos, int scale);
void refresh_from_active(const Position& pos,
                         const Features::IndexList active[2]);
void update_from_changed(const Position& pos,
                         const Features::IndexList removed[2],
                         const Features::IndexList added[2],
                         const bool reset[2]);
std::int32_t runtime_weight(IndexType index);
void colocated_refresh(const Position& pos, const std::int32_t sums[2]);
void colocated_update(const Position& pos,
                      const std::int32_t removed_sums[2],
                      const std::int32_t added_sums[2],
                      const bool reset[2]);
void observe(const Position& pos);
void test_position(const Position& pos, const std::string& path,
                   std::uint64_t repeats, std::ostream& out);
void self_test(const std::string& path, std::uint64_t games, int max_ply,
               std::ostream& out);

}  // namespace YaneuraOu::Eval::NNUE::NnueKpProgressShadow

#endif
#endif
