// Scope the unchanged independent equations to the plain-buffer path used
// by the16-row research model. The full Table16suite failure is retained.
#define main full_gdn_suite_main
#include DSPINE_GDN_ORACLE_SOURCE
#undef main
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: gdn-plain16 METALLIB");
    static_assert(kRows == 16);
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        tiledHeadOrder(backend, shape, lanes, LinearInput::Plain, true);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
          runDecode(backend, shape, lanes, float32);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (GdnHeadOrder order : {GdnHeadOrder::Grouped, GdnHeadOrder::Tiled}) {
          gateMatchesPrefill(backend, shape, float32, order);
          replayAcrossBatches(backend, shape, float32, order, LinearInput::Plain);
        }
    for (const GdnShape &shape : kShapes)
      for (const uint32_t lanes : {1U, kMaxLanes})
        for (bool float32 : familyNorms(shape))
          bfloatState(backend, shape, lanes, float32, LinearInput::Plain);
    std::cout << "gdn_plain16_oracle: PASS (Table16 excluded; known unsupported path)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_plain16_oracle: FAIL: " << error.what() << '\n';
    return 1;
  }
}
