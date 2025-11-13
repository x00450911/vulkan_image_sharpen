# Architecture Documentation

## System Overview

The Vulkan Multi-Frame Video Denoiser is designed as a modular, GPU-accelerated video processing pipeline. The system leverages Vulkan compute shaders for parallel processing of video frames.

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  (User code: FFmpeg, OpenCV, custom video processing)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VulkanVideoDenoiser API                        │
│  • initialize()  • processFrame()  • updateParameters()     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vulkan Pipeline                            │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐              │
│  │  Upload  │──▶│ Compute  │──▶│ Download  │              │
│  │  Stage   │   │  Shader  │   │   Stage   │              │
│  └──────────┘   └──────────┘   └───────────┘              │
└─────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU Hardware                             │
│           (Vulkan-capable Graphics Card)                    │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. VulkanVideoDenoiser Class

**Responsibility**: Main API interface and resource management

**Key Components**:
- Vulkan instance and device management
- Memory allocation and buffer management
- Pipeline state management
- Frame history ring buffer

**Lifecycle**:
1. Construction: Initialize member variables
2. `initialize()`: Create Vulkan resources
3. `processFrame()`: Execute denoising pipeline (can be called repeatedly)
4. `cleanup()` / Destruction: Release all resources

### 2. Memory Management

#### Frame Buffers
```
Ring Buffer Layout (4 frames):
┌─────────┐
│ Frame 0 │ ◄─── Current frame (just uploaded)
├─────────┤
│ Frame 1 │ ◄─── Previous frame (t-1)
├─────────┤
│ Frame 2 │ ◄─── Frame (t-2)
├─────────┤
│ Frame 3 │ ◄─── Frame (t-3)
└─────────┘

After rotation:
┌─────────┐
│ Frame 1 │ ◄─── New current frame
├─────────┤
│ Frame 2 │ ◄─── Previous frame (t-1)
├─────────┤
│ Frame 3 │ ◄─── Frame (t-2)
├─────────┤
│ Frame 0 │ ◄─── Frame (t-3) - will be overwritten next
└─────────┘
```

#### Memory Types
- **Device Local**: Frame buffers, output buffer (GPU-only access, fastest)
- **Host Visible**: Staging buffers (CPU-GPU transfer, slower but accessible)
- **Host Coherent**: Automatic synchronization between CPU and GPU

### 3. Compute Shader Pipeline

```
Shader Execution Flow:
────────────────────────────────────────────────────────

Input Binding:
  • binding=0: Current frame (RGBA8)
  • binding=1: Previous frame 1 (RGBA8)
  • binding=2: Previous frame 2 (RGBA8)
  • binding=3: Previous frame 3 (RGBA8)
  • binding=5: Motion vectors (RG16F)

        │
        ▼
┌──────────────────────┐
│  Workgroup (16x16)   │
│  Per-pixel compute   │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Temporal Filtering   │
│  • Load current px   │
│  • Load history px   │
│  • Compute weights   │
│  • Accumulate        │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Spatial Filtering    │
│  • Bilateral filter  │
│  • 3x3, 5x5, or 7x7  │
│  • Edge-preserving   │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Adaptive Weighting   │
│  • Variance estimate │
│  • Adjust strength   │
│  • Preserve details  │
└──────────────────────┘
        │
        ▼
Output Binding:
  • binding=4: Denoised frame (RGBA8)
```

### 4. Processing Pipeline

#### Single Frame Processing Flow

```
1. Upload Phase:
   CPU ──[staging buffer]──▶ GPU Device Memory
   
   • Map staging buffer
   • memcpy() input data
   • Unmap staging buffer
   • vkCmdCopyBufferToImage()
   • Transition image layout to GENERAL

2. Compute Phase:
   GPU Memory ──[compute shader]──▶ GPU Memory
   
   • Bind pipeline
   • Bind descriptor sets
   • Push constants (parameters)
   • Dispatch compute (workgroups)
   • Memory barriers (ensure completion)

3. Download Phase:
   GPU Device Memory ──[staging buffer]──▶ CPU
   
   • Transition image layout to TRANSFER_SRC
   • vkCmdCopyImageToBuffer()
   • Transition back to GENERAL
   • Map staging buffer
   • memcpy() output data
   • Unmap staging buffer

4. Frame Rotation:
   Ring Buffer Index = (Index + 1) % 4
```

### 5. Synchronization

#### Fences
- Used to wait for GPU operations to complete
- One fence per command buffer submission
- Reset before each submission, wait after

#### Memory Barriers
- Ensure compute shader writes are visible
- Coordinate image layout transitions
- Prevent read-after-write hazards

#### Pipeline Stages
```
VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    ↓
VK_PIPELINE_STAGE_TRANSFER_BIT
    ↓
VK_PIPELINE_STAGE_HOST_BIT
```

## Algorithm Details

### Temporal Filtering Algorithm

```cpp
for each pixel (x, y):
    current_color = load_pixel(current_frame, x, y)
    
    // Estimate if this is a detail/edge region
    local_variance = compute_variance_3x3(x, y)
    is_detail = (local_variance > threshold)
    
    // Adjust temporal weight for details
    effective_weight = temporal_weight * (1 - detail_strength * is_detail)
    
    accumulated = current_color
    total_weight = 1.0
    
    for i in [1, 2, 3]:  // Previous frames
        // Optionally apply motion compensation
        if motion_compensation:
            motion = load_motion_vector(x, y)
            prev_pos = (x, y) - motion * i
        else:
            prev_pos = (x, y)
        
        prev_color = load_pixel(previous_frame[i], prev_pos)
        
        // Compute color difference
        color_diff = distance(current_color, prev_color)
        
        // Reject outliers (scene changes, occlusions)
        if color_diff < rejection_threshold:
            // Exponential decay for older frames
            frame_weight = effective_weight * pow(0.7, i)
            
            // Bilateral weight based on color similarity
            similarity = gaussian(color_diff, color_sigma)
            
            weight = frame_weight * similarity
            accumulated += prev_color * weight
            total_weight += weight
    
    temporal_result = accumulated / total_weight
```

### Spatial Bilateral Filtering

```cpp
for each pixel (x, y):
    center_color = temporal_result(x, y)
    
    accumulated = center_color
    total_weight = 1.0
    
    for dy in [-kernel_radius, kernel_radius]:
        for dx in [-kernel_radius, kernel_radius]:
            if (dx == 0 && dy == 0):
                continue
            
            neighbor_pos = (x + dx, y + dy)
            neighbor_color = temporal_result(neighbor_pos)
            
            // Spatial distance weight
            spatial_dist = sqrt(dx*dx + dy*dy)
            spatial_weight = gaussian(spatial_dist, spatial_sigma)
            
            // Color similarity weight
            color_dist = distance(center_color, neighbor_color)
            color_weight = gaussian(color_dist, color_sigma)
            
            weight = spatial_weight * color_weight
            accumulated += neighbor_color * weight
            total_weight += weight
    
    final_result = accumulated / total_weight
```

## Performance Considerations

### GPU Workload Distribution

**Workgroup Size**: 16x16 = 256 threads per workgroup
- Good balance between occupancy and resource usage
- Matches typical warp/wavefront sizes (32-64)
- Allows efficient memory coalescing

**Grid Calculation**:
```cpp
num_workgroups_x = ceil(image_width / 16)
num_workgroups_y = ceil(image_height / 16)
total_workgroups = num_workgroups_x * num_workgroups_y
```

For 1920x1080:
- Workgroups: 120 × 68 = 8,160
- Total threads: 8,160 × 256 = 2,088,960

### Memory Bandwidth

**Per-frame transfers** (1920×1080 RGBA8):
- Upload: 8.3 MB
- Download: 8.3 MB
- Total: 16.6 MB per frame

At 60 FPS: ~1 GB/s bandwidth required

**Optimization**: Minimize CPU-GPU transfers
- Process multiple frames without downloading intermediate results
- Use GPU-to-GPU pipelines when possible

### Compute Performance

**Operations per pixel**:
- Temporal: ~10-20 texture reads, ~50 ALU ops
- Spatial: ~20-50 texture reads (depending on kernel), ~100 ALU ops
- Total: ~30-70 reads, ~150 ops

**Bottleneck**: Usually memory bandwidth (texture reads)
- L2 cache hit rate is critical
- Spatial locality helps (neighboring pixels access nearby data)

### Optimization Strategies

1. **Memory Access Patterns**
   - Coalesced reads (16x16 workgroup reads contiguous memory)
   - Cache-friendly kernel sizes
   - Minimize redundant texture fetches

2. **Arithmetic Optimization**
   - Vectorized operations (RGBA processed together)
   - Fast math operations where appropriate
   - Precompute weights when possible

3. **Occupancy**
   - Balance register usage vs. threads per SM
   - Avoid excessive local memory spills
   - Match workgroup size to hardware warp size

## Error Handling

### Initialization Errors
- **No Vulkan support**: Return false, log error
- **Insufficient VRAM**: Fail gracefully, suggest lower resolution
- **Shader compilation failure**: Check for shader files, validate SPIR-V

### Runtime Errors
- **Invalid input data**: Validate pointer and size
- **GPU timeout**: Increase timeout, check for infinite loops in shader
- **Out of memory**: Free unused resources, reduce frame history

### Recovery Strategies
- Automatic cleanup on failure
- No partial state (either fully initialized or not at all)
- Explicit error codes for debugging

## Extension Points

### Adding New Denoise Methods

1. Create new compute shader in `shaders/`
2. Add shader compilation to CMakeLists.txt
3. Create new pipeline in `VulkanVideoDenoiser`
4. Add switch/selection mechanism for algorithms

### Supporting Different Formats

1. Add format to `createImage()` calls
2. Update shader input/output formats
3. Adjust data conversion in upload/download
4. Handle different channel counts (RGB, YUV, etc.)

### Multi-GPU Support

1. Create multiple `VulkanVideoDenoiser` instances
2. Select different physical devices in `selectPhysicalDevice()`
3. Distribute frames across GPUs
4. Synchronize results on CPU side

## Testing Strategy

### Unit Tests
- Buffer creation and cleanup
- Image format conversions
- Parameter validation
- Memory leak detection

### Integration Tests
- Full pipeline execution
- Frame history management
- Parameter updates during processing
- Resource cleanup

### Performance Tests
- Throughput benchmarks (FPS)
- Latency measurements
- Memory usage profiling
- GPU utilization monitoring

## Future Enhancements

1. **Async Processing**: Overlap compute and transfer
2. **Dynamic Resolution**: Adapt to available VRAM
3. **ML Integration**: Neural network-based denoising
4. **HDR Support**: 16-bit and 32-bit float formats
5. **Multi-scale Processing**: Coarse-to-fine pyramid
6. **Improved Motion Estimation**: Optical flow on GPU
7. **Temporal Stability**: Flicker reduction
8. **Adaptive Quality**: Dynamic parameter adjustment based on content
