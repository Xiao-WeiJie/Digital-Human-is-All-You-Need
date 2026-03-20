#!/bin/bash

# 数字人系统 - 启动前检查脚本
# 使用: ./check_before_start.sh

echo "============================================"
echo "  数字人系统 - 启动前检查"
echo "============================================"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

ERRORS=0
WARNINGS=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

# 1. 检查目录结构
echo -e "\n[1/6] 检查目录结构..."
[ -d "Human_Choice" ] && check_pass "Human_Choice 目录" || check_fail "Human_Choice 目录不存在"
[ -d "Human_Choice/Human_1" ] && check_pass "Human_1 目录" || check_fail "Human_1 目录不存在"
[ -d "Human_Choice/Human_2" ] && check_pass "Human_2 目录" || check_fail "Human_2 目录不存在"
[ -d "digital_human_backend" ] && check_pass "digital_human_backend 目录" || check_fail "digital_human_backend 目录不存在"

# 2. 检查必需资源文件
echo -e "\n[2/6] 检查数字人资源文件..."
[ -f "Human_Choice/Human_1/qys.png" ] && check_pass "Human_1 源图像 (qys.png)" || check_fail "缺少 Human_1/qys.png"
[ -f "Human_Choice/Human_1/qys.mp4" ] && check_pass "Human_1 待机视频 (qys.mp4)" || check_fail "缺少 Human_1/qys.mp4"
[ -f "Human_Choice/Human_2/wzx.png" ] && check_pass "Human_2 源图像 (wzx.png)" || check_fail "缺少 Human_2/wzx.png"
[ -f "Human_Choice/Human_2/wzx.mp4" ] && check_pass "Human_2 待机视频 (wzx.mp4)" || check_fail "缺少 Human_2/wzx.mp4"

# 3. 检查TTS参考音频（可选）
echo -e "\n[3/6] 检查TTS参考音频（可选）..."
if [ -d "/root/autodl-tmp/ref_audio/human_1" ]; then
    [ -f "/root/autodl-tmp/ref_audio/human_1/default.wav" ] && check_pass "Human_1 TTS参考音频" || check_warn "Human_1 无TTS参考音频，将使用默认音色"
else
    check_warn "Human_1 无ref_audio目录，将使用默认音色"
fi

if [ -d "/root/autodl-tmp/ref_audio/human_2" ]; then
    [ -f "/root/autodl-tmp/ref_audio/human_2/default.wav" ] && check_pass "Human_2 TTS参考音频" || check_warn "Human_2 无TTS参考音频，将使用默认音色"
else
    check_warn "Human_2 无ref_audio目录，将使用默认音色"
fi

# 4. 检查配置文件
echo -e "\n[4/6] 检查配置文件..."
[ -f "digital_human_backend/.env" ] && check_pass ".env 配置文件" || check_fail "缺少 digital_human_backend/.env"
[ -f "digital_human_backend/config.py" ] && check_pass "config.py" || check_fail "缺少 digital_human_backend/config.py"
[ -f "Human_Choice/avatar_config.json" ] && check_pass "avatar_config.json" || check_warn "缺少 avatar_config.json（使用默认配置）"

# 5. 检查端口占用
echo -e "\n[5/6] 检查端口..."
for port in 50000 8000 8002 8004; do
    if command -v lsof &> /dev/null; then
        if lsof -i:$port > /dev/null 2>&1; then
            check_warn "端口 $port 已被占用"
        else
            check_pass "端口 $port 可用"
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            check_warn "端口 $port 已被占用"
        else
            check_pass "端口 $port 可用"
        fi
    else
        check_warn "无法检查端口 $port（缺少 lsof/netstat）"
    fi
done

# 6. 检查Python依赖
echo -e "\n[6/6] 检查Python依赖..."
python3 -c "import fastapi" 2>/dev/null && check_pass "fastapi" || check_fail "缺少 fastapi"
python3 -c "import uvicorn" 2>/dev/null && check_pass "uvicorn" || check_fail "缺少 uvicorn"
python3 -c "import aiohttp" 2>/dev/null && check_pass "aiohttp" || check_fail "缺少 aiohttp"

# 检查子模块
echo -e "\n检查子模块..."
[ -f "SenseVoice/api.py" ] && check_pass "SenseVoice 模块" || check_fail "缺少 SenseVoice 模块"
[ -f "Psychology_Rag/api_server.py" ] && check_pass "Psychology_Rag 模块" || check_fail "缺少 Psychology_Rag 模块"
[ -f "/root/autodl-tmp/PerFRDiffRender/inference_api.py" ] && check_pass "PerFRDiffRender 模块" || check_fail "缺少 PerFRDiffRender 模块"

# 检查TTS配置
echo -e "\n检查TTS服务配置..."
if [ -f "digital_human_backend/.env" ]; then
    TTS_URL=$(grep "TTS_SERVICE_URL" digital_human_backend/.env | cut -d'=' -f2)
    if echo "$TTS_URL" | grep -q "localhost"; then
        check_warn "TTS服务配置为本地，请确认GPT-SoVITS是否在本地运行"
    else
        check_pass "TTS服务配置为远程: $TTS_URL"
    fi
fi

# 汇总
echo -e "\n============================================"
echo "  检查完成"
echo "============================================"
echo -e "  错误: ${RED}$ERRORS${NC} 个"
echo -e "  警告: ${YELLOW}$WARNINGS${NC} 个"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}存在错误，请先修复后再启动！${NC}"
    echo ""
    echo "快速修复："
    [ ! -f "digital_human_backend/.env" ] && echo "  - 创建配置: cp digital_human_backend/.env.example digital_human_backend/.env"
    [ ! -f "Human_Choice/Human_1/qys.png" ] && echo "  - 添加数字人源图像到 Human_Choice/Human_1/"
    echo "  - 安装依赖: pip install -r digital_human_backend/requirements.txt"
    exit 1
else
    echo -e "${GREEN}检查通过，可以启动！${NC}"
    echo ""
    echo "启动命令: ./start_local.sh"
    echo "停止命令: ./stop_all.sh"
    exit 0
fi
