#include "FrameProvider.h"
#include "FrameWriter.h"
#include "VideoDenoiser.h"
#include "VulkanContext.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct AppConfig {
    uint32_t width = 640;
    uint32_t height = 360;
    uint32_t frameCount = 120;
    float blendFactor = 0.82f;
    float sigmaColor = 18.0f;       // in 0-255 space
    float sigmaTemporal = 25.0f;    // in 0-255 space
    bool useSpatialFilter = true;
    float noiseAmplitude = 0.25f;
    std::string outputDirectory = "output";
    bool enableValidationLayers =
#ifdef NDEBUG
        false;
#else
        true;
#endif
};

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [options]\n"
              << "Options:\n"
              << "  --width <pixels>            Frame width (default 640)\n"
              << "  --height <pixels>           Frame height (default 360)\n"
              << "  --frames <count>            Number of frames to synthesize (default 120)\n"
              << "  --blend <value>             Temporal blend factor (0-1, default 0.82)\n"
              << "  --sigma-color <value>       Sigma for color difference weighting in 0-255 space (default 18.0)\n"
              << "  --sigma-temporal <value>    Sigma for temporal difference weighting in 0-255 space (default 25.0)\n"
              << "  --noise <value>             Noise amplitude for synthetic input (0-1, default 0.25)\n"
              << "  --output <path>             Output directory for denoised frames (default ./output)\n"
              << "  --no-spatial                Disable spatial bilateral accumulation\n"
              << "  --no-validation             Disable Vulkan validation layers\n"
              << "  --help                      Show this message\n";
}

void parseArguments(int argc, char** argv, AppConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto requireValue = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for argument ") + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--width") {
            config.width = static_cast<uint32_t>(std::stoul(requireValue("--width")));
        } else if (arg == "--height") {
            config.height = static_cast<uint32_t>(std::stoul(requireValue("--height")));
        } else if (arg == "--frames") {
            config.frameCount = static_cast<uint32_t>(std::stoul(requireValue("--frames")));
        } else if (arg == "--blend") {
            config.blendFactor = std::stof(requireValue("--blend"));
        } else if (arg == "--sigma-color") {
            config.sigmaColor = std::stof(requireValue("--sigma-color"));
        } else if (arg == "--sigma-temporal") {
            config.sigmaTemporal = std::stof(requireValue("--sigma-temporal"));
        } else if (arg == "--noise") {
            config.noiseAmplitude = std::stof(requireValue("--noise"));
        } else if (arg == "--output") {
            config.outputDirectory = requireValue("--output");
        } else if (arg == "--no-spatial") {
            config.useSpatialFilter = false;
        } else if (arg == "--no-validation") {
            config.enableValidationLayers = false;
        } else if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    AppConfig config;

    try {
        parseArguments(argc, argv, config);

        std::cout << "Initialising Vulkan context..." << std::endl;
        VulkanContext context;
        context.initialize(config.enableValidationLayers);

        std::cout << "Creating denoiser pipeline for " << config.width << "x" << config.height << " frames." << std::endl;
        VideoDenoiser denoiser(context);
        denoiser.initialize(config.width, config.height);

        SyntheticFrameProvider frameProvider({config.width, config.height}, config.frameCount, config.noiseAmplitude);

        std::filesystem::create_directories(config.outputDirectory);

        std::vector<uint8_t> noisyFrame;
        std::vector<uint8_t> denoisedFrame;

        uint32_t frameIndex = 0;
        while (frameIndex < config.frameCount && frameProvider.getNextFrame(noisyFrame)) {
            denoiser.processFrame(noisyFrame, denoisedFrame, config.blendFactor,
                                  config.sigmaColor, config.sigmaTemporal, config.useSpatialFilter);

            const std::string filename =
                config.outputDirectory + "/denoised_" + std::to_string(frameIndex) + ".ppm";
            FrameWriter::writeFrameAsPPM(filename, config.width, config.height, denoisedFrame);

            std::cout << "Processed frame " << frameIndex + 1 << " / " << config.frameCount << std::endl;
            ++frameIndex;
        }

        std::cout << "Denoising complete. Frames written to " << config.outputDirectory << std::endl;

        denoiser.cleanup();
        context.cleanup();
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
