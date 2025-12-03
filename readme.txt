=============================
RTX 4060 Ti Video Super-Res
=============================

本工程提供了一个使用 C++17 + LibTorch（CUDA 12.x）+ OpenCV 的视频超分推理示例，允许直接加载由 PyTorch 训练得到的超分模型（TorchScript `.pth/.pt`）并在 RTX 4060 Ti 上通过 CUDA 加速处理整段视频。

-----------------
1. 环境依赖
-----------------
- Ubuntu 20.04+，NVIDIA 驱动 ≥ 550，CUDA 12.1（4060 Ti 对应 SM 89）
- CMake ≥ 3.18，Ninja 或 Make
- Compiler: GCC ≥ 9（需完整 C++17 支持）
- OpenCV ≥ 4.5（需包含 `core/imgproc/videoio` 组件）
- LibTorch（CUDA 12.x 版本），例如：
  `wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip`

---------------------------
2. 将 PyTorch 模型转为 TorchScript
---------------------------
LibTorch 只能加载 TorchScript 模型，请先在 Python 中将 `.pth` state_dict 转为 `.ts`（或 `.pth` 但已 script 化）：

```python
import torch
from sr_model import Net  # 替换为你的模型定义

checkpoint = torch.load("original_weights.pth", map_location="cpu")
model = Net().eval()
model.load_state_dict(checkpoint)

example = torch.rand(1, 3, 270, 480)
scripted = torch.jit.trace(model, example).half()  # 若需要半精度
scripted.save("video_sr.ts")  # C++ 侧直接加载该文件
```

------------------------
3. 目录结构与源代码
------------------------
- `CMakeLists.txt`：项目配置，默认输出 `video_sr` 可执行文件。
- `include/sr/VideoSuperRes.h`：封装推理接口，负责模型加载、单帧超分及视频处理。
- `src/VideoSuperRes.cpp`：实现 CUDA 推理、图像预处理/后处理、视频编解码。
- `src/main.cpp`：命令行入口，解析参数后调用推理引擎。

-------------------
4. 工程配置步骤
-------------------
```bash
# 1) 下载并解压 LibTorch（确保是 cu12x 版本）
export TORCH_HOME=$HOME/libtorch-cu121
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip -d $TORCH_HOME

# 2) 配置 OpenCV
sudo apt-get install libopencv-dev

# 3) 生成并编译
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=$TORCH_HOME/libtorch/share/cmake/Torch \
  -DTORCH_CUDA_ARCH_LIST="8.9" \
  -G Ninja
ninja -C build

# 4) 运行前确保动态库可见
export LD_LIBRARY_PATH=$TORCH_HOME/libtorch/lib:$LD_LIBRARY_PATH
```

CMake 中已经默认在缺省情况下将 `TORCH_CUDA_ARCH_LIST` 设为 8.9（RTX 4060 Ti），也可按需覆盖。

--------------------------
5. 运行视频超分推理
--------------------------
```bash
./build/video_sr \
  --model /path/to/video_sr.ts \
  --input input_1080p.mp4 \
  --output output_4k.mp4 \
  --device cuda:0 \
  --fp16 \
  --warmup 16
```

- `--model`：TorchScript 文件路径。
- `--input / --output`：输入输出视频文件。
- `--device`：`cuda:0` 或 `cpu`。
- `--fp16`：启用半精度，加速 Ada GPU 的 Tensor Core。
- `--warmup`：CUDA kernel 热身次数，避免前几帧性能抖动。
- `--fourcc`：自定义 OpenCV 写文件的编码格式（默认 `mp4v`，若系统装有 `H264/AVC` 可设置为 `avc1`）。
- `--norm mean:std`：按通道进行归一化（示例：`--norm 0.5,0.5,0.5:0.5,0.5,0.5`）。

-----------------------
6. 性能与调优建议
-----------------------
- 确保 `nvidia-smi` 显示进程运行在 8.9 SM 上（Ada）。
- 如果模型内部仍为 FP32，可在 Python 端手动调用 `.half()` 后重新保存。
- 视频 I/O 的性能取决于 OpenCV 编解码后端，可根据系统切换为 GStreamer/FFmpeg。
- 需要短时多次调用时，可以在 C++ 层复用 `VideoSuperRes` 实例，以免重复加载模型。

-----------------------
7. 常见故障排查
-----------------------
- **`libtorch_cuda.so: undefined symbol`**：LibTorch 版本与 CUDA 不匹配，重新下载 cu12x 版本。
- **`VideoWriter` 无法打开**：系统缺少 H264/AVC 编码器，尝试修改 `VideoWriter::fourcc` 为 `mp4v` 或使用 `FFMPEG` 后端。
- **`Expected Tensor but got None`**：模型输出为空，确认 TorchScript 导出时 forward 返回 tensor。

通过上述流程即可在 RTX 4060 Ti 上以 C++/CUDA 高效运行 `.pth/.ts` 格式的视频超分网络。
