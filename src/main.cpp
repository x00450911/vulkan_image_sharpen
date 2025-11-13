#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "vulkan_context.h"
#include "frame_manager.h"
#include "denoiser.h"

class VideoDenoiser {
public:
    VideoDenoiser() : context_(std::make_unique<VulkanContext>()) {}
    
    bool initialize(uint32_t width, uint32_t height, uint32_t frameBufferCount) {
        if (!context_->initialize()) {
            std::cerr << "Failed to initialize Vulkan context!" << std::endl;
            return false;
        }

        frameManager_ = std::make_unique<FrameManager>(context_.get());
        if (!frameManager_->initialize(width, height, frameBufferCount)) {
            std::cerr << "Failed to initialize frame manager!" << std::endl;
            return false;
        }

        denoiser_ = std::make_unique<Denoiser>(context_.get(), frameManager_.get());
        if (!denoiser_->initialize()) {
            std::cerr << "Failed to initialize denoiser!" << std::endl;
            return false;
        }

        return true;
    }

    void processVideo(const std::string& inputPath, const std::string& outputPath, float alpha = 0.2f) {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file: " << inputPath << std::endl;
            return;
        }

        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "Video info: " << width << "x" << height << " @ " << fps << " fps, " << totalFrames << " frames" << std::endl;

        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(width, height));

        cv::Mat frame;
        uint32_t frameIndex = 0;
        std::vector<uint8_t> frameData(width * height * 4);

        auto startTime = std::chrono::high_resolution_clock::now();

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            // Convert BGR to RGBA
            cv::Mat rgbaFrame;
            cv::cvtColor(frame, rgbaFrame, cv::COLOR_BGR2RGBA);
            
            // Copy frame data
            memcpy(frameData.data(), rgbaFrame.data, width * height * 4);

            // Upload frame to GPU
            frameManager_->uploadFrame(frameIndex % frameManager_->getFrameCount(), frameData.data(), frameData.size());

            // Denoise frame
            if (frameIndex > 0) {
                denoiser_->denoiseFrame(
                    frameIndex % frameManager_->getFrameCount(),
                    std::min(frameIndex, static_cast<uint32_t>(frameManager_->getFrameCount())),
                    alpha
                );
            }

            // Download denoised frame
            frameManager_->downloadFrame(frameIndex % frameManager_->getFrameCount(), frameData.data(), frameData.size());

            // Convert RGBA back to BGR
            cv::Mat outputFrame(height, width, CV_8UC4, frameData.data());
            cv::Mat bgrFrame;
            cv::cvtColor(outputFrame, bgrFrame, cv::COLOR_RGBA2BGR);

            // Write frame
            writer.write(bgrFrame);

            frameIndex++;

            if (frameIndex % 30 == 0) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
                double fps = (frameIndex * 1000.0) / elapsed;
                std::cout << "Processed " << frameIndex << " / " << totalFrames 
                          << " frames (" << fps << " fps)" << std::endl;
            }
        }

        cap.release();
        writer.release();

        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "Processing complete! Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << (frameIndex * 1000.0) / totalTime << std::endl;
    }

private:
    std::unique_ptr<VulkanContext> context_;
    std::unique_ptr<FrameManager> frameManager_;
    std::unique_ptr<Denoiser> denoiser_;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <input_video> <output_video> [alpha]" << std::endl;
    std::cout << "  input_video:  Path to input video file" << std::endl;
    std::cout << "  output_video: Path to output video file" << std::endl;
    std::cout << "  alpha:        Temporal blending factor (0.0-1.0, default: 0.2)" << std::endl;
    std::cout << "                Lower values = more temporal smoothing (less noise, more motion blur)" << std::endl;
    std::cout << "                Higher values = less temporal smoothing (more noise, less motion blur)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    float alpha = 0.2f;

    if (argc >= 4) {
        alpha = std::stof(argv[3]);
        alpha = std::max(0.0f, std::min(1.0f, alpha)); // Clamp to [0, 1]
    }

    std::cout << "Vulkan Multi-Frame Video Denoising" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Input:  " << inputPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "Alpha:  " << alpha << std::endl;
    std::cout << std::endl;

    // Read first frame to get dimensions
    cv::VideoCapture testCap(inputPath);
    if (!testCap.isOpened()) {
        std::cerr << "Error opening video file: " << inputPath << std::endl;
        return 1;
    }

    cv::Mat testFrame;
    testCap.read(testFrame);
    testCap.release();

    if (testFrame.empty()) {
        std::cerr << "Error reading first frame!" << std::endl;
        return 1;
    }

    uint32_t width = testFrame.cols;
    uint32_t height = testFrame.rows;
    uint32_t frameBufferCount = 4; // Number of frames to keep in GPU memory

    std::cout << "Initializing Vulkan..." << std::endl;
    VideoDenoiser denoiser;
    if (!denoiser.initialize(width, height, frameBufferCount)) {
        std::cerr << "Failed to initialize video denoiser!" << std::endl;
        return 1;
    }

    std::cout << "Processing video..." << std::endl;
    denoiser.processVideo(inputPath, outputPath, alpha);

    std::cout << "Done!" << std::endl;
    return 0;
}
