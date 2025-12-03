// VideoSuperRes.h
#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <torch/script.h>

namespace sr {

struct VideoSRConfig {
  std::string model_path;
  std::string device = "cuda:0";
  bool use_fp16 = false;
  int warmup_iterations = 8;
  std::array<float, 3> mean = {0.f, 0.f, 0.f};
  std::array<float, 3> std = {1.f, 1.f, 1.f};
  bool apply_normalization = false;
  std::string fourcc = "mp4v";
};

class VideoSuperRes {
 public:
  explicit VideoSuperRes(VideoSRConfig config);

  cv::Mat Upscale(const cv::Mat& frame);
  void ProcessVideo(const std::string& input_path, const std::string& output_path);
  const VideoSRConfig& config() const { return config_; }

 private:
  cv::Mat TensorToImage(const torch::Tensor& tensor);

  VideoSRConfig config_;
  torch::Device device_;
  torch::jit::script::Module module_;
};

}  // namespace sr
