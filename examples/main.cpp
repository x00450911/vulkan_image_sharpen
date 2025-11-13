#include "vulkan_video_denoiser.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <chrono>

// Simple PPM image loader/saver for demonstration
struct Image {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> data; // RGBA format
};

// Load PPM image
bool loadPPM(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    
    std::string magic;
    file >> magic;
    if (magic != "P6") {
        std::cerr << "Not a valid PPM file" << std::endl;
        return false;
    }
    
    file >> img.width >> img.height;
    int maxval;
    file >> maxval;
    file.ignore(1); // Skip newline
    
    std::vector<uint8_t> rgb(img.width * img.height * 3);
    file.read(reinterpret_cast<char*>(rgb.data()), rgb.size());
    
    // Convert RGB to RGBA
    img.data.resize(img.width * img.height * 4);
    for (size_t i = 0; i < img.width * img.height; i++) {
        img.data[i * 4 + 0] = rgb[i * 3 + 0]; // R
        img.data[i * 4 + 1] = rgb[i * 3 + 1]; // G
        img.data[i * 4 + 2] = rgb[i * 3 + 2]; // B
        img.data[i * 4 + 3] = 255;             // A
    }
    
    return true;
}

// Save PPM image
bool savePPM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to create " << filename << std::endl;
        return false;
    }
    
    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    
    // Convert RGBA to RGB
    for (size_t i = 0; i < img.width * img.height; i++) {
        file.put(img.data[i * 4 + 0]); // R
        file.put(img.data[i * 4 + 1]); // G
        file.put(img.data[i * 4 + 2]); // B
    }
    
    return true;
}

// Add synthetic noise to image for testing
void addNoise(Image& img, float noiseLevel) {
    for (size_t i = 0; i < img.width * img.height; i++) {
        for (int c = 0; c < 3; c++) {
            float noise = (rand() / static_cast<float>(RAND_MAX) - 0.5f) * noiseLevel * 255.0f;
            int val = static_cast<int>(img.data[i * 4 + c]) + static_cast<int>(noise);
            img.data[i * 4 + c] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }
}

// Example 1: Single image denoising (applying temporal filter with frame history)
void example1_singleImage() {
    std::cout << "\n=== Example 1: Single Image Denoising ===" << std::endl;
    
    // Create synthetic noisy image
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    
    Image inputImage;
    inputImage.width = width;
    inputImage.height = height;
    inputImage.data.resize(width * height * 4);
    
    // Generate gradient pattern
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t idx = (y * width + x) * 4;
            inputImage.data[idx + 0] = static_cast<uint8_t>((x * 255) / width);     // R
            inputImage.data[idx + 1] = static_cast<uint8_t>((y * 255) / height);    // G
            inputImage.data[idx + 2] = static_cast<uint8_t>(128);                    // B
            inputImage.data[idx + 3] = 255;                                          // A
        }
    }
    
    // Add noise
    addNoise(inputImage, 0.1f);
    
    // Initialize denoiser
    vk_denoise::VulkanVideoDenoiser denoiser;
    vk_denoise::DenoiseParams params;
    params.temporalWeight = 0.7f;
    params.spatialSigma = 1.5f;
    params.colorSigma = 0.15f;
    params.noiseThreshold = 0.1f;
    params.frameCount = 3;
    params.kernelSize = 5;
    
    if (!denoiser.initialize(width, height, params)) {
        std::cerr << "Failed to initialize denoiser" << std::endl;
        return;
    }
    
    // Process frame multiple times to build up frame history
    Image outputImage;
    outputImage.width = width;
    outputImage.height = height;
    outputImage.data.resize(width * height * 4);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 5; i++) {
        denoiser.processFrame(inputImage.data.data(), outputImage.data.data());
        std::cout << "Processed frame " << (i + 1) << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average per frame: " << (duration.count() / 5.0f) << " ms" << std::endl;
    std::cout << "FPS: " << (5000.0f / duration.count()) << std::endl;
    
    std::cout << "Denoising complete!" << std::endl;
}

// Example 2: Video sequence denoising
void example2_videoSequence() {
    std::cout << "\n=== Example 2: Video Sequence Denoising ===" << std::endl;
    
    const uint32_t width = 1280;
    const uint32_t height = 720;
    const int numFrames = 30;
    
    // Initialize denoiser
    vk_denoise::VulkanVideoDenoiser denoiser;
    vk_denoise::DenoiseParams params;
    params.temporalWeight = 0.8f;      // Higher temporal weight for video
    params.spatialSigma = 1.2f;
    params.colorSigma = 0.12f;
    params.noiseThreshold = 0.08f;
    params.frameCount = 4;             // Use all 4 previous frames
    params.adaptiveStrength = 0.6f;    // Adaptive to preserve details
    params.kernelSize = 5;
    
    if (!denoiser.initialize(width, height, params)) {
        std::cerr << "Failed to initialize denoiser" << std::endl;
        return;
    }
    
    // Create synthetic video frames
    std::vector<Image> inputFrames(numFrames);
    std::vector<Image> outputFrames(numFrames);
    
    for (int f = 0; f < numFrames; f++) {
        inputFrames[f].width = width;
        inputFrames[f].height = height;
        inputFrames[f].data.resize(width * height * 4);
        
        outputFrames[f].width = width;
        outputFrames[f].height = height;
        outputFrames[f].data.resize(width * height * 4);
        
        // Generate animated pattern
        float phase = (f * 2.0f * 3.14159f) / numFrames;
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                uint32_t idx = (y * width + x) * 4;
                
                // Moving wave pattern
                float wave = sin(x * 0.01f + phase) * 0.5f + 0.5f;
                
                inputFrames[f].data[idx + 0] = static_cast<uint8_t>(wave * 255);
                inputFrames[f].data[idx + 1] = static_cast<uint8_t>((y * 255) / height);
                inputFrames[f].data[idx + 2] = static_cast<uint8_t>((x * 255) / width);
                inputFrames[f].data[idx + 3] = 255;
            }
        }
        
        // Add noise
        addNoise(inputFrames[f], 0.15f);
    }
    
    // Process video sequence
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int f = 0; f < numFrames; f++) {
        denoiser.processFrame(inputFrames[f].data.data(), outputFrames[f].data.data());
        
        if (f % 10 == 0) {
            std::cout << "Processed frame " << f << "/" << numFrames << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\nVideo denoising complete!" << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average per frame: " << (duration.count() / static_cast<float>(numFrames)) << " ms" << std::endl;
    std::cout << "Throughput: " << (numFrames * 1000.0f / duration.count()) << " FPS" << std::endl;
}

// Example 3: Comparing different parameter settings
void example3_parameterComparison() {
    std::cout << "\n=== Example 3: Parameter Comparison ===" << std::endl;
    
    const uint32_t width = 1024;
    const uint32_t height = 768;
    
    // Create test image
    Image testImage;
    testImage.width = width;
    testImage.height = height;
    testImage.data.resize(width * height * 4);
    
    // Checkerboard pattern
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t idx = (y * width + x) * 4;
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            uint8_t val = checker ? 200 : 50;
            
            testImage.data[idx + 0] = val;
            testImage.data[idx + 1] = val;
            testImage.data[idx + 2] = val;
            testImage.data[idx + 3] = 255;
        }
    }
    
    addNoise(testImage, 0.2f);
    
    // Test different configurations
    struct Config {
        std::string name;
        vk_denoise::DenoiseParams params;
    };
    
    std::vector<Config> configs = {
        {"Light Denoising", {0.5f, 1.0f, 0.1f, false, 0.1f, 2, 0.3f, 3}},
        {"Medium Denoising", {0.7f, 1.5f, 0.15f, false, 0.1f, 3, 0.5f, 5}},
        {"Heavy Denoising", {0.9f, 2.0f, 0.2f, false, 0.15f, 4, 0.7f, 7}},
        {"Detail Preserving", {0.6f, 1.0f, 0.08f, false, 0.05f, 3, 0.8f, 3}}
    };
    
    Image outputImage;
    outputImage.width = width;
    outputImage.height = height;
    outputImage.data.resize(width * height * 4);
    
    for (const auto& config : configs) {
        std::cout << "\nTesting: " << config.name << std::endl;
        
        vk_denoise::VulkanVideoDenoiser denoiser;
        if (!denoiser.initialize(width, height, config.params)) {
            std::cerr << "Failed to initialize denoiser" << std::endl;
            continue;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process several times to build frame history
        for (int i = 0; i < 4; i++) {
            denoiser.processFrame(testImage.data.data(), outputImage.data.data());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Processing time: " << (duration.count() / 1000.0f) << " ms" << std::endl;
        std::cout << "  Temporal weight: " << config.params.temporalWeight << std::endl;
        std::cout << "  Spatial sigma: " << config.params.spatialSigma << std::endl;
        std::cout << "  Kernel size: " << config.params.kernelSize << std::endl;
    }
}

// Example 4: Real-time parameter adjustment
void example4_dynamicParameters() {
    std::cout << "\n=== Example 4: Dynamic Parameter Adjustment ===" << std::endl;
    
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    
    Image testImage;
    testImage.width = width;
    testImage.height = height;
    testImage.data.resize(width * height * 4);
    
    // Generate test pattern
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t idx = (y * width + x) * 4;
            testImage.data[idx + 0] = static_cast<uint8_t>((x + y) % 256);
            testImage.data[idx + 1] = static_cast<uint8_t>(x % 256);
            testImage.data[idx + 2] = static_cast<uint8_t>(y % 256);
            testImage.data[idx + 3] = 255;
        }
    }
    
    addNoise(testImage, 0.15f);
    
    // Initialize denoiser
    vk_denoise::VulkanVideoDenoiser denoiser;
    vk_denoise::DenoiseParams params;
    params.temporalWeight = 0.5f;
    params.spatialSigma = 1.0f;
    params.colorSigma = 0.1f;
    params.frameCount = 3;
    
    if (!denoiser.initialize(width, height, params)) {
        std::cerr << "Failed to initialize denoiser" << std::endl;
        return;
    }
    
    Image outputImage;
    outputImage.width = width;
    outputImage.height = height;
    outputImage.data.resize(width * height * 4);
    
    // Gradually increase denoising strength
    for (int frame = 0; frame < 20; frame++) {
        params.temporalWeight = 0.3f + (frame / 20.0f) * 0.6f;
        params.spatialSigma = 0.8f + (frame / 20.0f) * 1.5f;
        
        denoiser.updateParameters(params);
        denoiser.processFrame(testImage.data.data(), outputImage.data.data());
        
        if (frame % 5 == 0) {
            std::cout << "Frame " << frame << ": temporal=" << params.temporalWeight 
                     << ", spatial=" << params.spatialSigma << std::endl;
        }
    }
    
    std::cout << "Dynamic parameter adjustment complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Vulkan Multi-Frame Video Denoiser - Examples" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        // Run examples
        example1_singleImage();
        example2_videoSequence();
        example3_parameterComparison();
        example4_dynamicParameters();
        
        std::cout << "\n\nAll examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
