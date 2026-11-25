// MiniMax-H3's canvas tiers, on the CPU (video/ops/Geometry.hpp): the canvases a tier admits and those refused, and
// the clip lengths a geometry refuses. canvasRefusal decides the engine's refusal of a request frame.
#include "TestChecks.hpp"
#include "video/ops/Geometry.hpp"

#include <array>
#include <format>
#include <iostream>
#include <vector>

namespace {

using splash::test::require;
using namespace splash::video;

void checkRefusals() {
  for (auto [w, h, tier] : std::vector<std::array<int32_t, 3>>{{1344, 768, 768}, {768, 1344, 768},  {768, 768, 768},
                                                                {1536, 768, 768}, {832, 480, 480},   {864, 480, 480},
                                                                {480, 832, 480},  {640, 480, 480},   {960, 480, 480},
                                                                {1280, 720, 0},   {1920, 1080, 0},   {1024, 576, 0},
                                                                {1600, 768, 0},   {992, 480, 0},     {832, 464, 0},
                                                                {848, 480, 0},    {0, 0, 0},         {-768, 1344, 0}}) {
    require(ops::canvasTier(w, h) == tier, std::format("canvas {} x {}: tier {}, not {}", w, h,
                                                       ops::canvasTier(w, h), tier));
    require(ops::canvasRefusal(w, h).empty() == (tier != 0), std::format("canvas {} x {}: refusal", w, h));
  }
  // 241 frames is not 17 n + 5: the 10 s clip is 243
  splash::test::rejects([] { (void)ops::VideoGeometry::of(241, 1344, 768); }, "17 n + 5", "241 frames were taken");
}

} // namespace

int main() {
  try {
    checkRefusals();
    std::cout << "video_canvas_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "video_canvas_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
