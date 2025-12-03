#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Image {
public:
    Image() = default;
    Image(int width, int height, int channels);

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }

    bool empty() const { return data_.empty(); }

    float &at(int x, int y, int c);
    const float &at(int x, int y, int c) const;

    Image resizedBilinear(int targetWidth, int targetHeight) const;
    Image boxBlurred(int radius) const;

    void fill(float value);
    std::vector<float> &raw() { return data_; }
    const std::vector<float> &raw() const { return data_; }

private:
    int width_{0};
    int height_{0};
    int channels_{0};
    std::vector<float> data_;

    inline size_t index(int x, int y, int c) const {
        return static_cast<size_t>((y * width_ + x) * channels_ + c);
    }
};

Image LoadSyntheticGradient(int width, int height, int channels);
void SaveAsPGM(const Image &image, const std::string &path);
