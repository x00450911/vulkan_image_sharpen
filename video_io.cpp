#include "video_io.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

// PPMVideoReader implementation
PPMVideoReader::PPMVideoReader() : width_(0), height_(0), opened_(false) {
}

PPMVideoReader::~PPMVideoReader() {
    Close();
}

bool PPMVideoReader::Open(const std::string& filename) {
    filename_ = filename;
    opened_ = true;
    
    // Read header to get dimensions
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    std::string magic;
    file >> magic;
    if (magic != "P6" && magic != "P3") {
        return false;
    }
    
    file >> width_ >> height_;
    int maxVal;
    file >> maxVal;
    file.get(); // Skip newline
    
    file.close();
    return true;
}

void PPMVideoReader::Close() {
    opened_ = false;
}

bool PPMVideoReader::ReadFrame(VideoFrame& frame) {
    if (!opened_) return false;
    
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    std::string magic;
    file >> magic;
    if (magic != "P6" && magic != "P3") {
        return false;
    }
    
    uint32_t width, height;
    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.get(); // Skip newline
    
    frame.width = width;
    frame.height = height;
    frame.data.resize(width * height * 4);
    
    if (magic == "P6") {
        // Binary PPM
        std::vector<uint8_t> rgbData(width * height * 3);
        file.read(reinterpret_cast<char*>(rgbData.data()), width * height * 3);
        
        for (size_t i = 0; i < width * height; i++) {
            frame.data[i * 4 + 0] = rgbData[i * 3 + 0]; // R
            frame.data[i * 4 + 1] = rgbData[i * 3 + 1]; // G
            frame.data[i * 4 + 2] = rgbData[i * 3 + 2]; // B
            frame.data[i * 4 + 3] = 255; // A
        }
    } else {
        // ASCII PPM
        for (size_t i = 0; i < width * height; i++) {
            int r, g, b;
            file >> r >> g >> b;
            frame.data[i * 4 + 0] = static_cast<uint8_t>(r);
            frame.data[i * 4 + 1] = static_cast<uint8_t>(g);
            frame.data[i * 4 + 2] = static_cast<uint8_t>(b);
            frame.data[i * 4 + 3] = 255;
        }
    }
    
    frame.frameNumber = 0;
    frame.timestamp = 0.0;
    
    return true;
}

// PPMVideoWriter implementation
PPMVideoWriter::PPMVideoWriter() : width_(0), height_(0), frameCounter_(0), opened_(false) {
}

PPMVideoWriter::~PPMVideoWriter() {
    Close();
}

bool PPMVideoWriter::Open(const std::string& filename, uint32_t width, uint32_t height, double fps) {
    baseFilename_ = filename;
    width_ = width;
    height_ = height;
    frameCounter_ = 0;
    opened_ = true;
    return true;
}

void PPMVideoWriter::Close() {
    opened_ = false;
}

bool PPMVideoWriter::WriteFrame(const VideoFrame& frame) {
    if (!opened_) return false;
    
    std::ostringstream filename;
    filename << baseFilename_ << "_" << std::setfill('0') << std::setw(6) << frameCounter_ << ".ppm";
    
    std::ofstream file(filename.str(), std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file << "P6\n" << frame.width << " " << frame.height << "\n255\n";
    
    for (size_t i = 0; i < frame.width * frame.height; i++) {
        file.write(reinterpret_cast<const char*>(&frame.data[i * 4]), 3); // Write RGB, skip Alpha
    }
    
    frameCounter_++;
    return true;
}
