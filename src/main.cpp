#include "multi_frame_denoiser.h"
#include "vulkan_context.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct CliOptions {
    std::filesystem::path inputDir;
    std::filesystem::path outputDir = "denoised_output";
    std::filesystem::path shaderPath;
    uint32_t historyLength = 4;
    float blendFactor = 0.1f;
    float luminanceSigma = 0.2f;
    bool enableValidation = false;
};

struct FrameData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> pixels;  // RGBA32F
};

bool parseArguments(int argc, char** argv, CliOptions& outOptions) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            outOptions.inputDir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outOptions.outputDir = argv[++i];
        } else if (arg == "--shader" && i + 1 < argc) {
            outOptions.shaderPath = argv[++i];
        } else if (arg == "--history" && i + 1 < argc) {
            outOptions.historyLength = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--blend" && i + 1 < argc) {
            outOptions.blendFactor = std::stof(argv[++i]);
        } else if (arg == "--sigma" && i + 1 < argc) {
            outOptions.luminanceSigma = std::stof(argv[++i]);
        } else if (arg == "--validation") {
            outOptions.enableValidation = true;
        } else if (arg == "--help") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return false;
        }
    }

    if (outOptions.inputDir.empty()) {
        std::cerr << "Missing required argument: --input <directory>\n";
        return false;
    }

    if (outOptions.shaderPath.empty()) {
        outOptions.shaderPath = std::filesystem::current_path() / "shaders" / "denoise.comp.spv";
    }

    return true;
}

std::string toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return str;
}

bool isPfmFile(const std::filesystem::path& path) {
    return toLower(path.extension().string()) == ".pfm";
}

FrameData loadPfm(const std::filesystem::path& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open PFM file: " + filePath.string());
    }

    auto readNonCommentLine = [&]() -> std::string {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty() && line[0] == '#') {
                continue;
            }
            if (!line.empty()) {
                return line;
            }
        }
        throw std::runtime_error("Unexpected end of file while reading header in: " + filePath.string());
    };

    const std::string signature = readNonCommentLine();
    if (signature != "PF") {
        throw std::runtime_error("Unsupported PFM format (expected 'PF'): " + filePath.string());
    }

    const std::string dimensionLine = readNonCommentLine();
    std::istringstream dimsStream(dimensionLine);
    int width = 0;
    int height = 0;
    dimsStream >> width >> height;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid PFM dimensions in: " + filePath.string());
    }

    const std::string scaleLine = readNonCommentLine();
    const float scale = std::stof(scaleLine);
    if (scale >= 0.0f) {
        throw std::runtime_error("PFM scale must be negative for little-endian data: " + filePath.string());
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    std::vector<float> rgb(pixelCount * 3u);
    file.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size() * sizeof(float)));
    if (file.gcount() != static_cast<std::streamsize>(rgb.size() * sizeof(float))) {
        throw std::runtime_error("Unexpected EOF when reading pixel data from: " + filePath.string());
    }

    FrameData frame;
    frame.width = static_cast<uint32_t>(width);
    frame.height = static_cast<uint32_t>(height);
    frame.pixels.resize(pixelCount * 4u);

    for (uint32_t y = 0; y < frame.height; ++y) {
        for (uint32_t x = 0; x < frame.width; ++x) {
            const size_t srcRow = static_cast<size_t>(frame.height - 1 - y);
            const size_t srcIndex = (srcRow * frame.width + x) * 3u;
            const size_t dstIndex = (static_cast<size_t>(y) * frame.width + x) * 4u;
            frame.pixels[dstIndex + 0] = rgb[srcIndex + 0];
            frame.pixels[dstIndex + 1] = rgb[srcIndex + 1];
            frame.pixels[dstIndex + 2] = rgb[srcIndex + 2];
            frame.pixels[dstIndex + 3] = 1.0f;
        }
    }

    return frame;
}

void writePfm(const std::filesystem::path& filePath, uint32_t width, uint32_t height, const std::vector<float>& pixels) {
    const size_t expected = static_cast<size_t>(width) * height * 4u;
    if (pixels.size() != expected) {
        throw std::runtime_error("writePfm: pixel count mismatch for " + filePath.string());
    }

    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output PFM file: " + filePath.string());
    }

    file << "PF\n";
    file << width << " " << height << "\n";
    file << "-1.0\n";

    std::vector<float> rgb(static_cast<size_t>(width) * height * 3u);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t dstRow = static_cast<size_t>(height - 1 - y);
            const size_t dstIndex = (dstRow * width + x) * 3u;
            const size_t srcIndex = (static_cast<size_t>(y) * width + x) * 4u;
            rgb[dstIndex + 0] = pixels[srcIndex + 0];
            rgb[dstIndex + 1] = pixels[srcIndex + 1];
            rgb[dstIndex + 2] = pixels[srcIndex + 2];
        }
    }

    file.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size() * sizeof(float)));
}

int main(int argc, char** argv) {
    CliOptions options;
    if (!parseArguments(argc, argv, options)) {
        std::cout << "Usage: denoiser --input <dir> [--output <dir>] [--shader <path>] "
                     "[--history N] [--blend X] [--sigma Y] [--validation]\n";
        std::cout << "Frames must be provided as .pfm images in RGB floating-point format.\n";
        return EXIT_FAILURE;
    }

    try {
        if (!std::filesystem::exists(options.inputDir)) {
            throw std::runtime_error("Input directory does not exist: " + options.inputDir.string());
        }

        std::filesystem::create_directories(options.outputDir);

        std::vector<std::filesystem::path> frameFiles;
        for (const auto& entry : std::filesystem::directory_iterator(options.inputDir)) {
            if (entry.is_regular_file() && isPfmFile(entry.path())) {
                frameFiles.push_back(entry.path());
            }
        }
        std::sort(frameFiles.begin(), frameFiles.end());

        if (frameFiles.empty()) {
            throw std::runtime_error("No .pfm frames found in " + options.inputDir.string());
        }

        FrameData firstFrame = loadPfm(frameFiles.front());
        const uint32_t width = firstFrame.width;
        const uint32_t height = firstFrame.height;

        VulkanContext context;
        context.initialize(options.enableValidation);

        DenoiserConfig config;
        config.width = width;
        config.height = height;
        config.historyLength = options.historyLength;
        config.frameBlendFactor = options.blendFactor;
        config.luminanceSigma = options.luminanceSigma;
        config.shaderSpirvPath = options.shaderPath;

        MultiFrameDenoiser denoiser;
        denoiser.initialize(context, config);

        std::vector<float> outputPixels;

        for (size_t frameIdx = 0; frameIdx < frameFiles.size(); ++frameIdx) {
            FrameData frame = (frameIdx == 0) ? firstFrame : loadPfm(frameFiles[frameIdx]);
            if (frame.width != width || frame.height != height) {
                throw std::runtime_error("Frame resolution mismatch in " + frameFiles[frameIdx].string());
            }

            denoiser.processFrame(frame.pixels.data(), frame.pixels.size(), outputPixels);

            const std::filesystem::path outputPath = options.outputDir / frameFiles[frameIdx].filename();
            writePfm(outputPath, width, height, outputPixels);

            std::cout << "Processed frame " << (frameIdx + 1) << " / " << frameFiles.size()
                      << ": " << frameFiles[frameIdx].filename().string() << '\n';
        }

        denoiser.cleanup();
        context.cleanup();

        std::cout << "Denoising complete. Output written to " << options.outputDir << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
