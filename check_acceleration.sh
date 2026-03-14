#!/bin/bash
# 数字人项目加速诊断脚本
# 用法: bash check_acceleration.sh

echo "=============================================="
echo "  数字人项目加速诊断工具"
echo "=============================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 NVIDIA GPU
echo "【1】检查 NVIDIA GPU"
echo "----------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi 可用${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    echo "GPU 编码器支持:"
    nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps --format=csv,noheader 2>/dev/null || echo "  (编码器统计不可用)"
else
    echo -e "${RED}✗ nvidia-smi 不可用，请检查 NVIDIA 驱动${NC}"
fi
echo ""

# 2. 检查 TensorRT
echo "【2】检查 TensorRT 环境"
echo "----------------------------------------------"
if python3 -c "import tensorrt; print(f'TensorRT 版本: {tensorrt.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ TensorRT Python 绑定已安装${NC}"
else
    echo -e "${RED}✗ TensorRT Python 绑定未安装${NC}"
    echo "  安装方法: pip install tensorrt"
fi
echo ""

# 3. 检查项目配置文件
echo "【3】检查项目 TensorRT 配置"
echo "----------------------------------------------"
CONFIG_FILE="FasterLivePortrait/configs/trt_infer.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✓ 配置文件存在: $CONFIG_FILE${NC}"
    echo ""
    echo "模型 predict_type 配置:"
    grep -A1 "predict_type" "$CONFIG_FILE" | head -20
    echo ""

    # 检查是否使用 trt
    trt_count=$(grep -c 'predict_type: "trt"' "$CONFIG_FILE" 2>/dev/null || echo "0")
    ort_count=$(grep -c 'predict_type: "ort"' "$CONFIG_FILE" 2>/dev/null || echo "0")
    echo "统计: TRT 模型 $trt_count 个, ONNX 模型 $ort_count 个"
else
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
fi
echo ""

# 4. 检查 TRT 模型文件
echo "【4】检查 TensorRT 模型文件 (.trt)"
echo "----------------------------------------------"
TRT_DIR="FasterLivePortrait/checkpoints/liveportrait_onnx"
if [ -d "$TRT_DIR" ]; then
    echo "TRT 模型目录: $TRT_DIR"
    echo ""
    trt_files=$(find "$TRT_DIR" -name "*.trt" 2>/dev/null)
    trt_count=$(echo "$trt_files" | grep -c ".trt" 2>/dev/null || echo "0")
    if [ "$trt_count" -gt 0 ]; then
        echo -e "${GREEN}✓ 找到 $trt_count 个 TRT 模型文件:${NC}"
        echo "$trt_files" | while read -r file; do
            if [ -n "$file" ]; then
                size=$(du -h "$file" 2>/dev/null | cut -f1)
                echo "  - $(basename "$file") ($size)"
            fi
        done
    else
        echo -e "${YELLOW}⚠ 未找到 .trt 文件${NC}"
        echo "  需要将 ONNX 模型转换为 TRT 引擎"
    fi
else
    echo -e "${RED}✗ 模型目录不存在: $TRT_DIR${NC}"
fi
echo ""

# 5. 检查 JoyVASA 是否使用 TRT
echo "【5】检查 JoyVASA 加速状态"
echo "----------------------------------------------"
JOYVASA_MODEL="FasterLivePortrait/checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
if [ -f "$JOYVASA_MODEL" ]; then
    echo -e "${YELLOW}⚠ JoyVASA 使用 PyTorch 模型 (.pt)${NC}"
    echo "  文件: $JOYVASA_MODEL"
    size=$(du -h "$JOYVASA_MODEL" 2>/dev/null | cut -f1)
    echo "  大小: $size"
    echo ""
    echo "  建议: 将 JoyVASA 转换为 TensorRT 以加速"
else
    echo -e "${RED}✗ JoyVASA 模型不存在${NC}"
fi
echo ""

# 6. 检查 FFmpeg
echo "【6】检查 FFmpeg 及硬件编码支持"
echo "----------------------------------------------"
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg 已安装${NC}"
    ffmpeg -version | head -1
    echo ""

    echo "硬件编码器支持:"
    # 检查 NVIDIA 编码器
    if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
        echo -e "  ${GREEN}✓ h264_nvenc (NVIDIA GPU 编码)${NC}"
    else
        echo -e "  ${RED}✗ h264_nvenc 不支持${NC}"
    fi

    if ffmpeg -encoders 2>/dev/null | grep -q "hevc_nvenc"; then
        echo -e "  ${GREEN}✓ hevc_nvenc (NVIDIA HEVC 编码)${NC}"
    else
        echo -e "  ${YELLOW}✗ hevc_nvenc 不支持${NC}"
    fi

    # 检查其他硬件编码器
    if ffmpeg -encoders 2>/dev/null | grep -q "h264_vaapi"; then
        echo -e "  ${GREEN}✓ h264_vaapi (VAAPI 硬件编码)${NC}"
    fi

    if ffmpeg -encoders 2>/dev/null | grep -q "h264_qsv"; then
        echo -e "  ${GREEN}✓ h264_qsv (Intel QSV 编码)${NC}"
    fi

    echo ""
    echo "可用编码器列表 (GPU 相关):"
    ffmpeg -encoders 2>/dev/null | grep -E "(nvenc|vaapi|qsv|cuda)" || echo "  (无硬件编码器)"

    echo ""
    echo "FFmpeg 编译配置 (GPU 相关):"
    ffmpeg -buildconf 2>/dev/null | grep -E "(nvenc|cuda|vaapi)" | head -5 || echo "  (未找到 GPU 相关配置)"
else
    echo -e "${RED}✗ FFmpeg 未安装${NC}"
fi
echo ""

# 7. 检查 Python 环境关键依赖
echo "【7】检查 Python 关键依赖"
echo "----------------------------------------------"
python3 << 'EOF'
import sys
packages = [
    ("torch", "PyTorch"),
    ("tensorrt", "TensorRT"),
    ("onnxruntime", "ONNX Runtime"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
]

for pkg, name in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        # 检查 CUDA 支持
        if pkg == "torch":
            cuda_available = mod.cuda.is_available()
            cuda_version = mod.version.cuda if hasattr(mod.version, 'cuda') else 'N/A'
            print(f"✓ {name}: {version}")
            print(f"    CUDA 可用: {cuda_available}, CUDA 版本: {cuda_version}")
        elif pkg == "onnxruntime":
            providers = mod.get_available_providers()
            print(f"✓ {name}: {version}")
            print(f"    可用 Providers: {providers}")
        else:
            print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name}: 未安装")
EOF
echo ""

# 8. 检查实际运行状态
echo "【8】检查服务运行状态"
echo "----------------------------------------------"
# 检查是否有运行的 Python 进程
if pgrep -f "start_batch_server.py" > /dev/null; then
    echo -e "${GREEN}✓ start_batch_server.py 正在运行${NC}"
    pid=$(pgrep -f "start_batch_server.py")
    echo "  PID: $pid"
else
    echo -e "${YELLOW}⚠ start_batch_server.py 未运行${NC}"
fi

# 检查 GPU 使用情况
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 进程:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "  (无 GPU 进程)"
fi
echo ""

# 9. 总结和建议
echo "=============================================="
echo "  诊断总结"
echo "=============================================="
echo ""

# 读取关键状态
has_gpu=$(command -v nvidia-smi &> /dev/null && echo "yes" || echo "no")
has_trt=$(python3 -c "import tensorrt" 2>/dev/null && echo "yes" || echo "no")
has_nvenc=$(ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc" && echo "yes" || echo "no")
trt_configured=$([ -f "$CONFIG_FILE" ] && grep -q 'predict_type: "trt"' "$CONFIG_FILE" && echo "yes" || echo "no")

echo "状态汇总:"
echo "  - NVIDIA GPU:      $([ "$has_gpu" = "yes" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
echo "  - TensorRT:        $([ "$has_trt" = "yes" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
echo "  - TRT 配置:        $([ "$trt_configured" = "yes" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
echo "  - NVENC 编码:      $([ "$has_nvenc" = "yes" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
echo ""

echo "优化建议:"
if [ "$has_trt" = "no" ]; then
    echo "  1. 安装 TensorRT: pip install tensorrt"
fi
if [ "$trt_configured" = "no" ]; then
    echo "  2. 检查配置文件是否使用 predict_type: \"trt\""
fi
if [ "$has_nvenc" = "no" ]; then
    echo "  3. 安装支持 NVENC 的 FFmpeg 版本"
fi
echo "  4. JoyVASA 当前使用 PyTorch，建议转换为 TensorRT"
echo "  5. 考虑减少 Diffusion 步数 (500 → 100) 加速"
echo ""