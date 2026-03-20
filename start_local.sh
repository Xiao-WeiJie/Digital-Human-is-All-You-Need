#!/bin/bash

# 数字人系统 - 本地服务启动脚本
# GPT-SoVITS 在远程服务器上运行，不在此启动

# 设置 CUDA 显存分配策略，避免碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  数字人系统 - 本地服务启动"
echo "  (GPT-SoVITS 使用远程服务)"
echo "============================================"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 日志目录（仅后台模式需要）
# ���查是否使用后台模式
BACKGROUND_MODE=false
if [ "$1" = "-d" ] || [ "$1" = "--daemon" ]; then
    BACKGROUND_MODE=true
    mkdir -p "$PROJECT_ROOT/logs"
fi

# 启动服务函数
start_service() {
    local name=$1
    local dir=$2
    local cmd=$3
    local log=$4

    echo "[$name] 启动中..."
    cd "$PROJECT_ROOT/$dir"
    if [ "$BACKGROUND_MODE" = true ]; then
        nohup $cmd > "$PROJECT_ROOT/logs/$log" 2>&1 &
        echo "[$name] PID: $! (日志: logs/$log)"
    else
        # 前台模式：日志直接输出到终端，添加服务名前缀
        $cmd 2>&1 | sed -u "s/^/[$name] /" &
        echo "[$name] PID: $!"
    fi
    cd "$PROJECT_ROOT"
}

# 1. SenseVoice ASR
start_service "SenseVoice" "SenseVoice" "python api.py" "sensevoice.log"
SENSEVOICE_PID=$!
sleep 3

# 2. Psychology_Rag
start_service "Psychology_Rag" "Psychology_Rag" "python api_server.py" "rag.log"
RAG_PID=$!
sleep 2

# 3. PerFRDiffRender
start_service "PerFRDiffRender" "PerFRDiffRender" "python inference_api.py --mode server" "render.log"
RENDER_PID=$!
sleep 3

# 4. 主后端
start_service "MainBackend" "digital_human_backend" "python main.py" "main.log"
MAIN_PID=$!

echo ""
echo "============================================"
echo "  所有本地服务已启动!"
echo "============================================"
echo ""
echo "  服务地址:"
echo "  - 主后端:     http://localhost:8000"
echo "  - SenseVoice: http://localhost:50000"
echo "  - RAG:        http://localhost:8002"
echo "  - Render:     http://localhost:8004"
echo "  - TTS(远程):  \${TTS_SERVICE_URL}"
echo ""
echo "  PID信息:"
echo "  - SenseVoice: $SENSEVOICE_PID"
echo "  - RAG:        $RAG_PID"
echo "  - Render:     $RENDER_PID"
echo "  - Main:       $MAIN_PID"
echo ""

if [ "$BACKGROUND_MODE" = true ]; then
    echo "  后台模式运行"
    echo "  查看日志: tail -f logs/<服务名>.log"
    echo "  停止服务: ./stop_all.sh"
    echo ""
else
    echo "  前台模式运行，日志直接输出到终端"
    echo "  按 Ctrl+C 停止所有服务..."
    echo ""
    # 捕获 Ctrl+C 信号，停止所有服务
    trap "echo ''; echo '正在停止所有服务...'; kill $SENSEVOICE_PID $RAG_PID $RENDER_PID $MAIN_PID 2>/dev/null; echo '所有服务已停止'; exit 0" SIGINT SIGTERM
    wait
fi
