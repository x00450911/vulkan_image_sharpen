#include "sr/VideoSuperRes.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace sr {

namespace {
torch::Tensor MaybeNormalize(const torch::Tensor& tensor,
                             const VideoSRConfig& config) {
  auto norm = tensor.div(255.0);
  if (!config.apply_normalization) {
    return norm;
  }

  auto options = norm.options();
  auto mean = torch::tensor(
                  {config.mean[0], config.mean[1], config.mean[2]}, options)
                  .view({1, 3, 1, 1});
  auto std = torch::tensor(
                 {config.std[0], config.std[1], config.std[2]}, options)
                 .view({1, 3, 1, 1});
  return norm.sub(mean).div(std);
}
}  // namespace

VideoSuperRes::VideoSuperRes(VideoSRConfig config)
    : config_(std::move(config)),
      device_(torch::Device(config_.device.empty() ? "cuda:0" : config_.device)) {
  if (!std::filesystem::exists(config_.model_path)) {
    throw std::runtime_error("Model path does not exist: " + config_.model_path);
  }

  module_ = torch::jit::load(config_.model_path, device_);
  module_.eval();
  module_.to(device_);
}

cv::Mat VideoSuperRes::Upscale(const cv::Mat& frame) {
  if (frame.empty()) {
    throw std::runtime_error("Received empty frame");
  }

  cv::Mat rgb;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
  } else if (frame.channels() == 1) {
    cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
  } else {
    throw std::runtime_error("Frame must have 1 or 3 channels");
  }

  auto tensor = torch::from_blob(
      rgb.data, {1, rgb.rows, rgb.cols, 3}, torch::TensorOptions().dtype(torch::kUInt8));
  tensor = tensor.permute({0, 3, 1, 2}).contiguous();
  tensor = MaybeNormalize(tensor, config_);
  tensor = tensor.to(device_, config_.use_fp16 ? torch::kHalf : torch::kFloat);

  torch::NoGradGuard no_grad;
  auto output = module_.forward({tensor}).toTensor();

  if (config_.use_fp16 && output.scalar_type() == torch::kHalf) {
    output = output.to(torch::kFloat);
  }

  return TensorToImage(output);
}

void VideoSuperRes::ProcessVideo(const std::string& input_path, const std::string& output_path) {
  cv::VideoCapture capture(input_path);
  if (!capture.isOpened()) {
    throw std::runtime_error("Failed to open input video: " + input_path);
  }

  cv::Mat frame;
  if (!capture.read(frame)) {
    throw std::runtime_error("Failed to read first frame");
  }

  // Warm-up iterations to stabilize CUDA kernels.
  capture.set(cv::CAP_PROP_POS_FRAMES, 0);
  for (int i = 0; i < config_.warmup_iterations && capture.read(frame); ++i) {
    Upscale(frame);
  }
  capture.set(cv::CAP_PROP_POS_FRAMES, 0);

  cv::VideoWriter writer;
  double fps = capture.get(cv::CAP_PROP_FPS);
  if (fps <= 1e-2) {
    fps = 30.0;
  }

  // Determine output resolution from first processed frame.
  if (!capture.read(frame)) {
    throw std::runtime_error("Failed to decode first frame after warmup");
  }
  auto first_upscaled = Upscale(frame);
  if (first_upscaled.empty()) {
    throw std::runtime_error("Upscale result is empty");
  }

  if (config_.fourcc.size() != 4) {
    throw std::runtime_error("fourcc must be 4 characters");
  }

  writer.open(output_path,
              cv::VideoWriter::fourcc(
                  config_.fourcc[0], config_.fourcc[1], config_.fourcc[2], config_.fourcc[3]),
              fps,
              first_upscaled.size());
  if (!writer.isOpened()) {
    throw std::runtime_error("Failed to open VideoWriter for: " + output_path);
  }
  writer.write(first_upscaled);

  while (capture.read(frame)) {
    writer.write(Upscale(frame));
  }

  writer.release();
}

cv::Mat VideoSuperRes::TensorToImage(const torch::Tensor& tensor) {
  auto squeezed = tensor.squeeze();
  if (squeezed.dim() != 3) {
    throw std::runtime_error("Model output must be BCHW");
  }

  torch::Tensor clamped = squeezed.clamp(0, 1).mul(255).to(torch::kU8);
  clamped = clamped.permute({1, 2, 0}).contiguous();  // HWC

  cv::Mat rgb(clamped.size(0), clamped.size(1), CV_8UC3);
  std::memcpy(rgb.data, clamped.data_ptr(), clamped.numel() * clamped.element_size());

  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr;
}

}  // namespace sr
