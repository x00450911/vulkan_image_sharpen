# Implementation Details

## Architecture Overview

This Vulkan-based multi-frame video denoising system consists of four main components:

### 1. VulkanContext (`vulkan_context.h/cpp`)
Manages all Vulkan resources and initialization:
- **Instance Creation**: Sets up Vulkan instance with required extensions
- **Device Selection**: Finds and selects a suitable GPU with compute capabilities
- **Resource Management**: Creates and manages buffers, images, command pools, and descriptor pools
- **Memory Management**: Handles GPU memory allocation and deallocation
- **Shader Loading**: Loads and compiles SPIR-V shaders

**Key Functions:**
- `initialize()`: Sets up Vulkan instance, device, queues, and pools
- `createImage()`: Creates GPU images with proper memory allocation
- `createBuffer()`: Creates GPU buffers
- `transitionImageLayout()`: Handles image layout transitions for compute/transfer operations
- `loadShaderModule()`: Loads compiled SPIR-V shaders

### 2. FrameManager (`frame_manager.h/cpp`)
Manages video frame buffers on the GPU:
- **Frame Storage**: Maintains multiple frame buffers in GPU memory
- **Upload/Download**: Handles CPU-GPU data transfers via staging buffers
- **Memory Management**: Tracks image and buffer memory for cleanup

**Key Functions:**
- `initialize()`: Creates frame buffers with images and staging buffers
- `uploadFrame()`: Transfers frame data from CPU to GPU
- `downloadFrame()`: Transfers frame data from GPU to CPU
- `getFrame()`: Accesses frame buffers by index

**Frame Buffer Structure:**
```cpp
struct FrameBuffer {
    VkImage image;              // GPU image storage
    VkDeviceMemory imageMemory; // Image memory
    VkImageView view;           // Image view for shader access
    VkBuffer stagingBuffer;      // CPU-visible staging buffer
    VkDeviceMemory stagingMemory; // Staging buffer memory
    uint32_t width, height;     // Frame dimensions
};
```

### 3. Denoiser (`denoiser.h/cpp`)
Implements the denoising pipeline:
- **Pipeline Creation**: Sets up compute pipeline with denoising shader
- **Descriptor Management**: Creates and updates descriptor sets for shader bindings
- **Denoising Execution**: Dispatches compute shaders to process frames

**Key Functions:**
- `initialize()`: Creates descriptor set layout, pipeline layout, and compute pipeline
- `denoiseFrame()`: Executes denoising on a frame using temporal blending
- `updateDescriptorSets()`: Updates shader bindings for current/previous/output frames

**Denoising Algorithm:**
1. Loads current frame and previous denoised frame
2. Applies bilateral filter to current frame (spatial denoising)
3. Blends filtered current frame with previous frame (temporal denoising)
4. Adaptively adjusts blending based on variance (motion detection)

### 4. Main Application (`main.cpp`)
Orchestrates video processing:
- **Video I/O**: Uses OpenCV to read and write video files
- **Frame Processing Loop**: Processes each frame through the denoising pipeline
- **Progress Tracking**: Reports processing statistics

**Processing Flow:**
```
1. Initialize Vulkan context
2. Initialize frame manager with N frame buffers
3. Initialize denoiser
4. For each video frame:
   a. Read frame from video file
   b. Convert BGR to RGBA
   c. Upload frame to GPU
   d. Denoise frame (if not first frame)
   e. Download denoised frame
   f. Convert RGBA to BGR
   g. Write frame to output video
```

## Compute Shaders

### denoise.comp
Main denoising shader implementing:
- **Bilateral Filtering**: Spatial denoising with Gaussian weights based on:
  - Spatial distance (Gaussian kernel)
  - Color difference (edge preservation)
- **Temporal Blending**: Exponential moving average:
  ```
  output = alpha * current + (1 - alpha) * previous
  ```
- **Adaptive Filtering**: Increases alpha in high-variance regions to preserve motion/detail

**Push Constants:**
- `alpha`: Temporal blending factor (0-1)
- `noiseThreshold`: Variance threshold for adaptive filtering
- `imageSize`: Frame dimensions

**Bindings:**
- Binding 0: Current frame (read-only storage image)
- Binding 1: Previous denoised frame (read-only storage image)
- Binding 2: Output frame (write-only storage image)

### accumulate.comp
Multi-frame accumulation shader (for future enhancements):
- Combines multiple frames with weighted averaging
- Supports up to 4 input frames
- Useful for advanced temporal techniques

## Memory Layout

### Image Layouts
Frames transition through these layouts:
1. `UNDEFINED` → `GENERAL`: Initial setup for compute shader access
2. `GENERAL` → `TRANSFER_DST_OPTIMAL`: Before uploading frame data
3. `TRANSFER_DST_OPTIMAL` → `GENERAL`: After upload, ready for compute
4. `GENERAL` → `TRANSFER_SRC_OPTIMAL`: Before downloading frame data
5. `TRANSFER_SRC_OPTIMAL` → `GENERAL`: After download, ready for next frame

### Buffer Usage
- **Staging Buffers**: Host-visible, used for CPU-GPU transfers
- **Images**: Device-local, optimal for compute shader access

## Performance Considerations

### Current Implementation
- Synchronous transfers (blocks until complete)
- Single command buffer per operation
- Sequential frame processing

### Optimization Opportunities
1. **Asynchronous Transfers**: Use separate transfer queue and semaphores
2. **Command Buffer Pooling**: Pre-allocate and reuse command buffers
3. **Pipelining**: Process frame N+1 while transferring frame N
4. **Multiple Queues**: Use separate compute and transfer queues
5. **Memory Aliasing**: Reuse memory for frames that are no longer needed

## Error Handling

The implementation includes error handling for:
- Vulkan initialization failures
- Resource creation failures
- Shader loading failures
- Invalid frame indices
- Unsupported layout transitions

All errors are reported via `std::cerr` and functions return `false` or throw exceptions on failure.

## Extensibility

The architecture supports easy extension:
- **Additional Shaders**: Add new compute shaders and pipelines
- **Different Algorithms**: Modify shader code without changing C++ code
- **Multiple Passes**: Chain multiple compute passes
- **Motion Estimation**: Add optical flow computation
- **Different Formats**: Extend to support other pixel formats

## Build System

CMake configuration:
- Automatically compiles GLSL shaders to SPIR-V using `glslc`
- Links against Vulkan, OpenCV, and GLFW libraries
- Copies compiled shaders to build directory
- Supports cross-platform compilation

## Testing

To test the implementation:
1. Build the project
2. Prepare a noisy video file
3. Run with different alpha values to find optimal settings
4. Compare output quality vs. processing time

Typical alpha values:
- **0.1**: Very aggressive denoising, significant motion blur
- **0.2**: Balanced (default), good noise reduction with acceptable motion preservation
- **0.5**: Light denoising, preserves motion but retains some noise
