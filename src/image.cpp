#include "image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
float clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}
}

Image::Image(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels),
      data_(static_cast<size_t>(width) * height * channels, 0.f) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }
}

float &Image::at(int x, int y, int c) {
    return data_.at(index(x, y, c));
}

const float &Image::at(int x, int y, int c) const {
    return data_.at(index(x, y, c));
}

void Image::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

Image Image::resizedBilinear(int targetWidth, int targetHeight) const {
    if (empty()) {
        return Image();
    }
    Image output(targetWidth, targetHeight, channels_);
    const float xScale = static_cast<float>(width_) / targetWidth;
    const float yScale = static_cast<float>(height_) / targetHeight;

    for (int y = 0; y < targetHeight; ++y) {
        const float srcY = (y + 0.5f) * yScale - 0.5f;
        const int y0 = clampInt(static_cast<int>(std::floor(srcY)), 0, height_ - 1);
        const int y1 = clampInt(y0 + 1, 0, height_ - 1);
        const float yLerp = srcY - y0;

        for (int x = 0; x < targetWidth; ++x) {
            const float srcX = (x + 0.5f) * xScale - 0.5f;
            const int x0 = clampInt(static_cast<int>(std::floor(srcX)), 0, width_ - 1);
            const int x1 = clampInt(x0 + 1, 0, width_ - 1);
            const float xLerp = srcX - x0;

            for (int c = 0; c < channels_; ++c) {
                const float top = at(x0, y0, c) * (1.f - xLerp) + at(x1, y0, c) * xLerp;
                const float bottom = at(x0, y1, c) * (1.f - xLerp) + at(x1, y1, c) * xLerp;
                output.at(x, y, c) = top * (1.f - yLerp) + bottom * yLerp;
            }
        }
    }
    return output;
}

Image Image::boxBlurred(int radius) const {
    if (radius <= 0 || empty()) {
        return *this;
    }
    Image output(width_, height_, channels_);
    const int diameter = radius * 2 + 1;
    const float invArea = 1.f / static_cast<float>(diameter * diameter);

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            for (int c = 0; c < channels_; ++c) {
                float accum = 0.f;
                for (int ky = -radius; ky <= radius; ++ky) {
                    const int py = clampInt(y + ky, 0, height_ - 1);
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const int px = clampInt(x + kx, 0, width_ - 1);
                        accum += at(px, py, c);
                    }
                }
                output.at(x, y, c) = accum * invArea;
            }
        }
    }
    return output;
}

Image LoadSyntheticGradient(int width, int height, int channels) {
    Image img(width, height, channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x) / width;
            const float fy = static_cast<float>(y) / height;
            for (int c = 0; c < channels; ++c) {
                img.at(x, y, c) = clamp(std::sin(fx * 3.1415f + c) * 0.5f + 0.5f + fy * 0.25f, 0.f, 1.f);
            }
        }
    }
    return img;
}

void SaveAsPGM(const Image &image, const std::string &path) {
    if (image.channels() == 0) {
        throw std::runtime_error("Cannot save empty image");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    ofs << "P5\n" << image.width() << " " << image.height() << "\n255\n";
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const float v = image.at(x, y, 0);
            const uint8_t b = static_cast<uint8_t>(clamp(v, 0.f, 1.f) * 255.f);
            ofs.write(reinterpret_cast<const char *>(&b), 1);
        }
    }
}
