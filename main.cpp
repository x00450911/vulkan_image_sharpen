#include "vulkan_denoiser.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>

// Simple video frame structure
struct VideoFrame {
    std::vector<uint8_t> data;
    uint32_t width;
    uint32_t height;
    uint32_t frameNumber;
};

// Simple function to generate test frames with noise
void GenerateTestFrame(VideoFrame& frame, uint32_t width, uint32_t height, uint32_t frameNum) {
    frame.width = width;
    frame.height = height;
    frame.frameNumber = frameNum;
    frame.data.resize(width * height * 4); // RGBA
    
    // Generate a simple pattern with noise
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t index = (y * width + x) * 4;
            
            // Base pattern (moving gradient)
            float fx = float(x) / float(width);
            float fy = float(y) / float(height);
            float time = float(frameNum) * 0.1f;
            
            uint8_t r = static_cast<uint8_t>((sin(fx * 3.14159f + time) * 0.5f + 0.5f) * 255.0f);
            uint8_t g = static_cast<uint8_t>((cos(fy * 3.14159f + time) * 0.5f + 0.5f) * 255.0f);
            uint8_t b = static_cast<uint8_t>((sin((fx + fy) * 3.14159f + time) * 0.5f + 0.5f) * 255.0f);
            
            // Add noise
            int noise = (rand() % 40) - 20;
            r = static_cast<uint8_t>(std::max(0, std::min(255, int(r) + noise)));
            g = static_cast<uint8_t>(std::max(0, std::min(255, int(g) + noise)));
            b = static_cast<uint8_t>(std::max(0, std::min(255, int(b) + noise)));
            
            frame.data[index + 0] = r;
            frame.data[index + 1] = g;
            frame.data[index + 2] = b;
            frame.data[index + 3] = 255; // Alpha
        }
    }
}

// Save frame as PPM (simple image format)
bool SaveFrameAsPPM(const std::string& filename, const uint8_t* data, uint32_t width, uint32_t height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file << "P6\n" << width << " " << height << "\n255\n";
    
    for (uint32_t i = 0; i < width * height; i++) {
        file.write(reinterpret_cast<const char*>(&data[i * 4]), 3); // Write RGB, skip Alpha
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t numFrames = 5; // Number of frames to use for denoising
    const uint32_t totalFrames = 30; // Total frames to process
    
    std::cout << "Vulkan Multi-Frame Video Denoiser" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Temporal window: " << numFrames << " frames" << std::endl;
    std::cout << "Total frames to process: " << totalFrames << std::endl;
    
    // Initialize denoiser
    VulkanDenoiser denoiser(width, height, numFrames);
    
    if (!denoiser.Initialize()) {
        std::cerr << "Failed to initialize Vulkan denoiser" << std::endl;
        return 1;
    }
    
    // Set denoising parameters
    denoiser.SetDenoisingStrength(0.7f);  // 70% denoising strength
    denoiser.SetTemporalWeight(0.8f);     // 80% temporal weight
    
    std::cout << "Denoising parameters:" << std::endl;
    std::cout << "  Strength: 0.7" << std::endl;
    std::cout << "  Temporal weight: 0.8" << std::endl;
    
    // Process frames
    std::vector<VideoFrame> inputFrames;
    std::vector<VideoFrame> outputFrames;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (uint32_t frameNum = 0; frameNum < totalFrames; frameNum++) {
        // Generate or load input frame
        VideoFrame inputFrame;
        GenerateTestFrame(inputFrame, width, height, frameNum);
        inputFrames.push_back(inputFrame);
        
        // Prepare output frame
        VideoFrame outputFrame;
        outputFrame.width = width;
        outputFrame.height = height;
        outputFrame.frameNumber = frameNum;
        outputFrame.data.resize(width * height * 4);
        
        // Process frame
        std::cout << "Processing frame " << frameNum + 1 << "/" << totalFrames << "...";
        std::cout.flush();
        
        if (!denoiser.ProcessFrame(inputFrame.data.data(), outputFrame.data.data(), frameNum)) {
            std::cerr << "\nFailed to process frame " << frameNum << std::endl;
            denoiser.Cleanup();
            return 1;
        }
        
        outputFrames.push_back(outputFrame);
        
        // Save sample frames
        if (frameNum == 0 || frameNum == totalFrames / 2 || frameNum == totalFrames - 1) {
            std::string inputFilename = "frame_" + std::to_string(frameNum) + "_input.ppm";
            std::string outputFilename = "frame_" + std::to_string(frameNum) + "_denoised.ppm";
            
            SaveFrameAsPPM(inputFilename, inputFrame.data.data(), width, height);
            SaveFrameAsPPM(outputFilename, outputFrame.data.data(), width, height);
            
            std::cout << " [saved samples]";
        }
        
        std::cout << " done" << std::endl;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "\nProcessing complete!" << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per frame: " << (duration.count() / float(totalFrames)) << " ms" << std::endl;
    std::cout << "FPS: " << (1000.0f / (duration.count() / float(totalFrames))) << std::endl;
    
    // Cleanup
    denoiser.Cleanup();
    
    std::cout << "\nSample frames saved:" << std::endl;
    std::cout << "  - frame_0_input.ppm / frame_0_denoised.ppm" << std::endl;
    std::cout << "  - frame_" << totalFrames / 2 << "_input.ppm / frame_" << totalFrames / 2 << "_denoised.ppm" << std::endl;
    std::cout << "  - frame_" << totalFrames - 1 << "_input.ppm / frame_" << totalFrames - 1 << "_denoised.ppm" << std::endl;
    
    return 0;
}
