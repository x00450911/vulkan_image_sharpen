#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Simple video frame structure
struct VideoFrame {
    std::vector<uint8_t> data;
    uint32_t width;
    uint32_t height;
    uint32_t frameNumber;
    double timestamp;
};

// Video reader interface (can be extended for FFmpeg, OpenCV, etc.)
class VideoReader {
public:
    virtual ~VideoReader() = default;
    virtual bool Open(const std::string& filename) = 0;
    virtual void Close() = 0;
    virtual bool ReadFrame(VideoFrame& frame) = 0;
    virtual uint32_t GetWidth() const = 0;
    virtual uint32_t GetHeight() const = 0;
    virtual uint32_t GetFrameCount() const = 0;
    virtual double GetFPS() const = 0;
    virtual bool SeekToFrame(uint32_t frameNumber) = 0;
};

// Video writer interface
class VideoWriter {
public:
    virtual ~VideoWriter() = default;
    virtual bool Open(const std::string& filename, uint32_t width, uint32_t height, double fps) = 0;
    virtual void Close() = 0;
    virtual bool WriteFrame(const VideoFrame& frame) = 0;
};

// Simple PPM image reader/writer (for testing)
class PPMVideoReader : public VideoReader {
public:
    PPMVideoReader();
    ~PPMVideoReader() override;
    bool Open(const std::string& filename) override;
    void Close() override;
    bool ReadFrame(VideoFrame& frame) override;
    uint32_t GetWidth() const override { return width_; }
    uint32_t GetHeight() const override { return height_; }
    uint32_t GetFrameCount() const override { return 1; }
    double GetFPS() const override { return 30.0; }
    bool SeekToFrame(uint32_t frameNumber) override { return frameNumber == 0; }

private:
    std::string filename_;
    uint32_t width_;
    uint32_t height_;
    bool opened_;
};

class PPMVideoWriter : public VideoWriter {
public:
    PPMVideoWriter();
    ~PPMVideoWriter() override;
    bool Open(const std::string& filename, uint32_t width, uint32_t height, double fps) override;
    void Close() override;
    bool WriteFrame(const VideoFrame& frame) override;

private:
    std::string baseFilename_;
    uint32_t width_;
    uint32_t height_;
    uint32_t frameCounter_;
    bool opened_;
};
