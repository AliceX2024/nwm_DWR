#!/bin/bash
# Navigation World Model 快速设置脚本
# 使用方法: bash quick_setup.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Navigation World Model 快速设置脚本"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置变量（根据实际情况修改）
PROJECT_DIR="$HOME/nwm"
DATA_SOURCE="/DATA/DATANAS2/xiaoyj25/recon"
DATA_LINK="$PROJECT_DIR/data/recon"
ENV_NAME="nwm"
PYTHON_VERSION="3.10"

echo -e "${YELLOW}步骤 1: 检查项目目录${NC}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}错误: 项目目录不存在: $PROJECT_DIR${NC}"
    exit 1
fi
cd "$PROJECT_DIR"
echo -e "${GREEN}✓ 项目目录存在${NC}"

echo -e "\n${YELLOW}步骤 2: 检查数据集${NC}"
if [ ! -d "$DATA_SOURCE" ]; then
    echo -e "${RED}警告: 数据集目录不存在: $DATA_SOURCE${NC}"
    echo "请先上传并解压数据集"
    exit 1
fi
echo -e "${GREEN}✓ 数据集目录存在${NC}"

# 检查数据集结构
if [ ! -f "$DATA_SOURCE"/*/traj_data.pkl ] 2>/dev/null; then
    echo -e "${YELLOW}警告: 未找到 traj_data.pkl 文件，数据集可能需要预处理${NC}"
fi

echo -e "\n${YELLOW}步骤 3: 创建数据软链接${NC}"
mkdir -p "$PROJECT_DIR/data"
if [ -L "$DATA_LINK" ]; then
    echo -e "${YELLOW}软链接已存在，跳过${NC}"
elif [ -d "$DATA_LINK" ]; then
    echo -e "${YELLOW}目录已存在，跳过${NC}"
else
    ln -s "$DATA_SOURCE" "$DATA_LINK"
    echo -e "${GREEN}✓ 软链接已创建${NC}"
fi

echo -e "\n${YELLOW}步骤 4: 检查 conda/mamba${NC}"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ 找到 mamba${NC}"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "${GREEN}✓ 找到 conda${NC}"
else
    echo -e "${RED}错误: 未找到 conda 或 mamba${NC}"
    exit 1
fi

echo -e "\n${YELLOW}步骤 5: 创建虚拟环境${NC}"
if $CONDA_CMD env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}环境 ${ENV_NAME} 已存在${NC}"
    read -p "是否重新创建环境? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $CONDA_CMD env remove -n "$ENV_NAME" -y
        $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        echo -e "${GREEN}✓ 环境已重新创建${NC}"
    fi
else
    $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo -e "${GREEN}✓ 环境已创建${NC}"
fi

echo -e "\n${YELLOW}步骤 6: 激活环境并安装依赖${NC}"
echo "请手动执行以下命令安装依赖:"
echo ""
echo "  $CONDA_CMD activate $ENV_NAME"
echo "  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126"
echo "  $CONDA_CMD install ffmpeg -y"
echo "  pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets"
echo ""

echo -e "${GREEN}=========================================="
echo "设置脚本执行完成！"
echo "==========================================${NC}"
echo ""
echo "下一步:"
echo "1. 激活环境: $CONDA_CMD activate $ENV_NAME"
echo "2. 安装依赖（见上方命令）"
echo "3. 验证安装: python -c \"import torch; print(torch.cuda.is_available())\""
echo "4. 运行测试: python train.py --config config/nwm_cdit_xl.yaml --epochs 1 --torch-compile 0"
echo ""
echo "详细说明请查看: SETUP_GUIDE.md"

