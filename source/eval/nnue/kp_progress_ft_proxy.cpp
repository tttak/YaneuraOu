#include "kp_progress_ft_proxy.h"

#if defined(USE_EXPERIMENTAL_KP_PROGRESS_FT_PROXY)

#include <array>
#include <chrono>
#include <iomanip>
#include <ostream>

namespace YaneuraOu::Eval::NNUE::NnueKpProgressFtProxy {
namespace {
constexpr std::array<int,32> Indices = {
  450,1218,243,1011,478,1246,332,1339,571,1100,1286,518,93,861,1071,1364,
  1361,593,570,303,18,786,596,1338,550,1318,648,1416,368,715,95,1231};
// Python input is in [0,1], while native transformed FT bytes use Q0.7.
constexpr std::array<float,32> Weights = {
 -0.2551232278f/127,-0.2302225977f/127,-0.5972347856f/127,-0.5177194476f/127,
 -0.4426254630f/127,-0.4480269253f/127,-0.2827780247f/127,-0.02400998957f/127,
  0.06323057413f/127,-0.2753007710f/127,-0.2192846090f/127,-0.2562185526f/127,
 -0.2040497661f/127,-0.1739098877f/127,-0.01875161752f/127,-0.07685273141f/127,
  0.2812597454f/127,0.2511866987f/127,-0.3558382690f/127,0.01363633387f/127,
 -0.7798849344f/127,-0.7746444345f/127,-0.07529319078f/127,-0.3329061866f/127,
 -0.1912360936f/127,-0.1714290977f/127,1.622112274f/127,1.790902853f/127,
 -0.3228856325f/127,-0.04220939428f/127,-0.03787202016f/127,-0.7225556970f/127};
constexpr float Bias = 0.5199973390f;
constexpr float Low = -1.0986122887f;
constexpr float Mid = -0.2006706955f;
volatile std::uint64_t sink = 0;

inline float predict(const TransformedFeatureType* transformed) {
  float value=Bias;
  for(std::size_t i=0;i<Indices.size();++i)
    value += Weights[i]*transformed[Indices[i]];
  return value;
}
inline int route(float value) { return value<Low ? 0 : value<Mid ? 1 : 2; }
}

void observe(const TransformedFeatureType* transformed) {
  sink ^= static_cast<std::uint64_t>(route(predict(transformed)));
}

void benchmark(std::uint64_t repeats, std::ostream& out) {
  alignas(64) std::array<TransformedFeatureType,1536> input{};
  for(std::size_t i=0;i<input.size();++i)
    input[i]=static_cast<TransformedFeatureType>((i*37+11)&127);
  const auto begin=std::chrono::steady_clock::now();
  float checksum=0;
  for(std::uint64_t n=0;n<repeats;++n) {
    input[Indices[n & 31]] ^= 1;
    checksum += predict(input.data());
  }
  const auto middle=std::chrono::steady_clock::now();
  int route_checksum=0;
  for(std::uint64_t n=0;n<repeats;++n) {
    input[Indices[n & 31]] ^= 1;
    route_checksum += route(predict(input.data()));
  }
  const auto end=std::chrono::steady_clock::now();
  out << std::setprecision(12) << "kp_progress_ft_proxy repeats " << repeats
      << " predict_ns " << std::chrono::duration<double,std::nano>(middle-begin).count()/repeats
      << " predict_route_ns " << std::chrono::duration<double,std::nano>(end-middle).count()/repeats
      << " checksum " << checksum << ' ' << route_checksum << '\n';
}
}
#endif
