#include "/Users/linson/Documents/ai/splash-dspine-investigation-20261110/precision/runtime/model/GgufPreparation.cpp"
#include <fstream>
#include <iterator>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  try {
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)), {});
    if (!input || data.empty() || data.size() % 128) throw std::runtime_error("invalid BF16 fixture");
    for (const uint32_t bits : {2u,3u,4u,5u,6u,8u}) {
      const uint32_t blockBytes = 4 + 64 * bits / 8;
      std::vector<uint8_t> out(data.size() / 128 * blockBytes);
      for (size_t group = 0; group < data.size()/128; ++group)
        splash::model::quantizeGroup(data.data()+group*128, out.data()+group*blockBytes, bits);
      std::ofstream output(std::string(argv[2])+"-"+std::to_string(bits)+".bin",std::ios::binary);
      output.write(reinterpret_cast<const char *>(out.data()),out.size());
      if (!output) throw std::runtime_error("cannot write quantization fixture");
    }
  } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
