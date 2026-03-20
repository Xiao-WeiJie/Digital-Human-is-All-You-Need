#!/bin/bash

echo "============================================"
echo "  停止所有数字人服务"
echo "============================================"

# 按端口杀进程
kill_port() {
    local port=$1
    local name=$2
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        kill $pid 2>/dev/null
        echo "[$name] 端口 $port 已停止 (PID: $pid)"
    else
        echo "[$name] 端口 $port 未运行"
    fi
}

kill_port 8000 "主后端"
kill_port 50000 "SenseVoice"
kill_port 8002 "RAG"
kill_port 8004 "Render"

echo ""
echo "所有服务已停止"
