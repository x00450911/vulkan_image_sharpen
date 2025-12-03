#include "sr/VideoSuperRes.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct CmdOptions {
  std::string model_path;
  std::string input_video;
  std::string output_video = "out.mp4";
  std::string device = "cuda:0";
  bool use_fp16 = false;
  int warmup_iterations = 8;
  std::string fourcc = "mp4v";
  std::array<float, 3> mean = {0.f, 0.f, 0.f};
  std::array<float, 3> std = {1.f, 1.f, 1.f};
  bool apply_norm = false;
};

void PrintUsage(const char* binary) {
  std::cerr << "Usage: " << binary
            << " --model model.ts --input input.mp4 --output upscale.mp4 [options]\n"
            << "Options:\n"
            << "  --device <cuda:0|cpu>   Select inference device (default cuda:0)\n"
            << "  --fp16                  Enable mixed precision (default false)\n"
            << "  --warmup <N>            Warmup iterations before recording (default 8)\n"
            << "  --fourcc <mp4v>         FourCC used by OpenCV VideoWriter (default mp4v)\n"
            << "  --norm <m1,m2,m3:s1,s2,s3>  Apply (value-mean)/std normalization\n";
}

CmdOptions ParseArgs(int argc, char** argv) {
  CmdOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto read_value = [&](std::string* target) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + arg);
      }
      *target = argv[++i];
    };

    if (arg == "--model") {
      read_value(&opts.model_path);
    } else if (arg == "--input") {
      read_value(&opts.input_video);
    } else if (arg == "--output") {
      read_value(&opts.output_video);
    } else if (arg == "--device") {
      read_value(&opts.device);
    } else if (arg == "--fp16") {
      opts.use_fp16 = true;
    } else if (arg == "--warmup") {
      std::string value;
      read_value(&value);
      opts.warmup_iterations = std::stoi(value);
    } else if (arg == "--fourcc") {
      read_value(&opts.fourcc);
      if (opts.fourcc.size() != 4) {
        throw std::runtime_error("--fourcc expects 4 characters");
      }
    } else if (arg == "--norm") {
      std::string value;
      read_value(&value);
      auto sep = value.find(':');
      if (sep == std::string::npos) {
        throw std::runtime_error("--norm expects mean:std (e.g. 0.5,0.5,0.5:0.5,0.5,0.5)");
      }
      auto parse_triplet = [](const std::string& part, std::array<float, 3>* target) {
        std::stringstream ss(part);
        std::string item;
        int idx = 0;
        while (std::getline(ss, item, ',')) {
          if (idx >= 3) break;
          (*target)[idx++] = std::stof(item);
        }
        if (idx != 3) {
          throw std::runtime_error("norm tuple must contain 3 values");
        }
      };
      parse_triplet(value.substr(0, sep), &opts.mean);
      parse_triplet(value.substr(sep + 1), &opts.std);
      opts.apply_norm = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.model_path.empty() || opts.input_video.empty()) {
    PrintUsage(argv[0]);
    throw std::runtime_error("Missing mandatory --model/--input argument");
  }

  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto cmd = ParseArgs(argc, argv);
    sr::VideoSRConfig config;
    config.model_path = cmd.model_path;
    config.device = cmd.device;
    config.use_fp16 = cmd.use_fp16;
    config.warmup_iterations = cmd.warmup_iterations;
    config.fourcc = cmd.fourcc;
    config.apply_normalization = cmd.apply_norm;
    config.mean = cmd.mean;
    config.std = cmd.std;

    sr::VideoSuperRes engine(config);
    engine.ProcessVideo(cmd.input_video, cmd.output_video);
  } catch (const std::exception& ex) {
    std::cerr << "[ERROR] " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
