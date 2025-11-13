## Vulkan Multi-Frame Denoising Pipeline

### Goals
- Denoise a sequence of video frames by combining information from several consecutive frames.
- Balance temporal stability with responsiveness to scene changes.
- Provide a Vulkan-based reference implementation that can be adapted to real-time or offline workflows.

### High-Level Flow
1. Initialize a Vulkan compute pipeline with storage images and uniform buffers describing the frame history.
2. For each video frame:
   - Upload the new frame into a staging buffer, then into the GPU frame history image.
   - Bind the compute pipeline with descriptors referencing the current frame, the history images, and temporal accumulation buffers.
   - Dispatch the compute shader to produce the denoised output.
   - Read the output back for encoding or display.
3. Rotate the history buffers to prepare for the next frame.

### Data Structures
- **FrameHistory**: Ring buffer storing the last `N` raw frames as `VK_FORMAT_R16G16B16A16_SFLOAT` images for high precision.
- **AccumulationBuffer**: Stores the previous denoised output to provide temporal feedback.
- **Uniforms**:
  - `frameIndex`: current frame number.
  - `frameBlendFactor`: blend factor when no motion is detected.
  - `luminanceSigma`: range weight controlling aggressiveness of temporal blending.
  - `historyLength`: number of frames available in history.

### Compute Shader Algorithm
1. Load the current pixel from the newest frame.
2. Iterate over the previous frames in the history:
   - Compute luminance difference between the current pixel and the historical pixel.
   - Compute a weight using the luminance difference and `luminanceSigma`.
   - Accumulate weighted color and weight sum.
3. Combine the temporal accumulation buffer with the new frame via `frameBlendFactor`.
4. Write the normalized result to the output image and update the accumulation buffer.

### Synchronization
- Use a single compute queue to simplify synchronization.
- Insert image memory barriers when transitioning between transfer and compute stages.
- Use `vkQueueWaitIdle` in the sample for simplicity; real applications should use fences and semaphores.

### Extensibility
- Replace the simple color-difference weighting with motion-compensated sampling using optical flow data.
- Switch to asynchronous compute queues for better performance.
- Integrate with a video decoder to stream frames directly into Vulkan images.
