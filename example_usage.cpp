// Example usage of VulkanDenoiser with different scenarios
// This file demonstrates various ways to use the denoiser

#include "vulkan_denoiser.h"
#include "video_io.h"
#include <iostream>
#include <vector>
#include <memory>

// Example 1: Basic usage with generated frames
void Example1_BasicUsage() {
    std::cout << "=== Example 1: Basic Usage ===" << std::endl;
    
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t numFrames = 5;
    
    // Initialize denoiser
    VulkanDenoiser denoiser(width, height, numFrames);
    if (!denoiser.Initialize()) {
        std::cerr << "Failed to initialize denoiser" << std::endl;
        return;
    }
    
    // Set parameters
    denoiser.SetDenoisingStrength(0.7f);
    denoiser.SetTemporalWeight(0.8f);
    
    // Process frames
    for (uint32_t i = 0; i < 10; i++) {
        // Allocate frame buffers
        std::vector<uint8_t> inputFrame(width * height * 4);
        std::vector<uint8_t> outputFrame(width * height * 4);
        
        // Fill input frame with your data...
        // (In real usage, load from video file or camera)
        
        // Process
        if (!denoiser.ProcessFrame(inputFrame.data(), outputFrame.data(), i)) {
            std::cerr << "Failed to process frame " << i << std::endl;
            break;
        }
        
        // Use output frame...
        // (Save to file, display, encode to video, etc.)
    }
    
    denoiser.Cleanup();
    std::cout << "Example 1 complete" << std::endl;
}

// Example 2: Adjusting parameters dynamically
void Example2_DynamicParameters() {
    std::cout << "\n=== Example 2: Dynamic Parameters ===" << std::endl;
    
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    
    VulkanDenoiser denoiser(width, height, 5);
    if (!denoiser.Initialize()) {
        return;
    }
    
    // Start with light denoising
    denoiser.SetDenoisingStrength(0.3f);
    denoiser.SetTemporalWeight(0.5f);
    
    // Process first few frames with light denoising
    for (uint32_t i = 0; i < 5; i++) {
        std::vector<uint8_t> input(width * height * 4);
        std::vector<uint8_t> output(width * height * 4);
        // ... fill input ...
        denoiser.ProcessFrame(input.data(), output.data(), i);
    }
    
    // Increase denoising for noisy scenes
    denoiser.SetDenoisingStrength(0.9f);
    denoiser.SetTemporalWeight(0.9f);
    
    // Process remaining frames with stronger denoising
    for (uint32_t i = 5; i < 10; i++) {
        std::vector<uint8_t> input(width * height * 4);
        std::vector<uint8_t> output(width * height * 4);
        // ... fill input ...
        denoiser.ProcessFrame(input.data(), output.data(), i);
    }
    
    denoiser.Cleanup();
    std::cout << "Example 2 complete" << std::endl;
}

// Example 3: Using with PPM images (for testing)
void Example3_PPMImages() {
    std::cout << "\n=== Example 3: PPM Image Processing ===" << std::endl;
    
    PPMVideoReader reader;
    PPMVideoWriter writer;
    
    if (!reader.Open("input_frame.ppm")) {
        std::cerr << "Failed to open input image" << std::endl;
        return;
    }
    
    uint32_t width = reader.GetWidth();
    uint32_t height = reader.GetHeight();
    
    VulkanDenoiser denoiser(width, height, 3);
    if (!denoiser.Initialize()) {
        return;
    }
    
    VideoFrame frame;
    if (reader.ReadFrame(frame)) {
        // For single frame, we need multiple copies for temporal denoising
        // In practice, you'd process a sequence of frames
        std::vector<uint8_t> output(frame.data.size());
        
        // Process the same frame multiple times (not ideal, but demonstrates usage)
        for (uint32_t i = 0; i < 3; i++) {
            denoiser.ProcessFrame(frame.data.data(), output.data(), i);
            // Copy output back to input for next iteration
            if (i < 2) {
                frame.data = output;
            }
        }
        
        // Save result
        VideoFrame outputFrame;
        outputFrame.width = width;
        outputFrame.height = height;
        outputFrame.data = output;
        
        writer.Open("output_frame", width, height, 30.0);
        writer.WriteFrame(outputFrame);
        writer.Close();
    }
    
    reader.Close();
    denoiser.Cleanup();
    std::cout << "Example 3 complete" << std::endl;
}

// Example 4: Processing video sequence
void Example4_VideoSequence() {
    std::cout << "\n=== Example 4: Video Sequence Processing ===" << std::endl;
    
    // This example shows the structure for processing a video sequence
    // In practice, you'd use FFmpeg or OpenCV to read/write video files
    
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t numFrames = 5;
    
    VulkanDenoiser denoiser(width, height, numFrames);
    if (!denoiser.Initialize()) {
        return;
    }
    
    denoiser.SetDenoisingStrength(0.7f);
    denoiser.SetTemporalWeight(0.8f);
    
    // Simulated video processing loop
    std::vector<uint8_t> inputFrame(width * height * 4);
    std::vector<uint8_t> outputFrame(width * height * 4);
    
    for (uint32_t frameIndex = 0; frameIndex < 100; frameIndex++) {
        // In real implementation:
        // 1. Read frame from video file (FFmpeg/OpenCV)
        // 2. Convert to RGBA format if needed
        // 3. Copy to inputFrame buffer
        
        // Process frame
        if (!denoiser.ProcessFrame(inputFrame.data(), outputFrame.data(), frameIndex)) {
            std::cerr << "Failed to process frame " << frameIndex << std::endl;
            break;
        }
        
        // In real implementation:
        // 1. Convert outputFrame to required format (BGR, YUV, etc.)
        // 2. Write to output video file (FFmpeg/OpenCV)
        
        if (frameIndex % 10 == 0) {
            std::cout << "Processed " << frameIndex << " frames" << std::endl;
        }
    }
    
    denoiser.Cleanup();
    std::cout << "Example 4 complete" << std::endl;
}

// Example 5: Different resolutions
void Example5_DifferentResolutions() {
    std::cout << "\n=== Example 5: Different Resolutions ===" << std::endl;
    
    // Process different resolutions
    struct Resolution {
        uint32_t width;
        uint32_t height;
        const char* name;
    };
    
    Resolution resolutions[] = {
        {1280, 720, "720p"},
        {1920, 1080, "1080p"},
        {3840, 2160, "4K"}
    };
    
    for (const auto& res : resolutions) {
        std::cout << "Processing " << res.name << " (" 
                  << res.width << "x" << res.height << ")..." << std::endl;
        
        VulkanDenoiser denoiser(res.width, res.height, 5);
        if (!denoiser.Initialize()) {
            continue;
        }
        
        std::vector<uint8_t> input(res.width * res.height * 4);
        std::vector<uint8_t> output(res.width * res.height * 4);
        
        // Process a few frames
        for (uint32_t i = 0; i < 3; i++) {
            // ... fill input ...
            denoiser.ProcessFrame(input.data(), output.data(), i);
        }
        
        denoiser.Cleanup();
        std::cout << "  " << res.name << " processing complete" << std::endl;
    }
    
    std::cout << "Example 5 complete" << std::endl;
}

int main() {
    std::cout << "Vulkan Denoiser Usage Examples\n" << std::endl;
    
    // Uncomment the example you want to run:
    
    // Example1_BasicUsage();
    // Example2_DynamicParameters();
    // Example3_PPMImages();
    // Example4_VideoSequence();
    // Example5_DifferentResolutions();
    
    std::cout << "\nNote: These are example code structures." << std::endl;
    std::cout << "Uncomment and modify examples in example_usage.cpp to use them." << std::endl;
    
    return 0;
}
