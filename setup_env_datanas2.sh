#!/bin/bash
# 使用 conda 配置 NWM 环境的脚本（NAS 共享版）
# 使用方法: bash setup_env_datanas2.sh

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 环境配置
ENV_PATH="/DATA/DATANAS2/xiaoyj25/envs/nwm3"   # NAS 上环境路径
PYTHON_VERSION="3.10"
TSINGHUA_CONDA_CHANNELS=(
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2"
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
    "defaults"
)
TSINGHUA_PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

pip_install_with_mirror() {
    local packages=("$@")
    echo -e "${BLUE}使用清华 pip 镜像安装: ${packages[*]}${NC}"
    if pip3 install --cache-dir=/tmp/pip_cache -i "$TSINGHUA_PIP_INDEX" --timeout 60 --retries 3 "${packages[@]}"; then
        return 0
    fi
    echo -e "${YELLOW}镜像源安装失败，回退到官方 PyPI...${NC}"
    pip3 install --cache-dir=/tmp/pip_cache "${packages[@]}"
}

run_conda_cmd() {
    local subcmd="$1"
    shift
    local args=("$@")
    local channel_args=()
    for channel in "${TSINGHUA_CONDA_CHANNELS[@]}"; do
        channel_args+=(-c "$channel")
    done
    echo -e "${BLUE}使用清华 conda 镜像执行: conda $subcmd ${args[*]}${NC}"
    if conda "$subcmd" "${args[@]}" "${channel_args[@]}"; then
        return 0
    fi
    echo -e "${YELLOW}镜像源执行失败，回退到官方 conda 源...${NC}"
    conda "$subcmd" "${args[@]}"
}

echo -e "${BLUE}=========================================="
echo "NWM 环境配置脚本 (NAS 共享版)"
echo "==========================================${NC}"

# 检查 conda 是否可用
echo -e "\n${YELLOW}步骤 1: 检查 conda${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda 未找到${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 找到 conda: $(conda --version)${NC}"

# 初始化 conda shell
echo -e "\n${YELLOW}初始化 conda shell...${NC}"
eval "$(conda shell.bash hook)"

# 删除已有环境（如果存在）
if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}发现已存在的环境，将删除: $ENV_PATH${NC}"
    rm -rf "$ENV_PATH"
fi

# 创建环境
echo -e "\n${YELLOW}步骤 2: 创建虚拟环境到 NAS${NC}"
run_conda_cmd create -p "$ENV_PATH" python=$PYTHON_VERSION -y

# 激活环境
echo -e "\n${YELLOW}步骤 3: 激活环境${NC}"
conda activate "$ENV_PATH"
echo -e "${GREEN}✓ 环境已激活: $ENV_PATH${NC}"

# 安装 PyTorch (CUDA 12.6)
echo -e "\n${YELLOW}步骤 4: 安装 PyTorch (CUDA 12.6)${NC}"
pip3 install --cache-dir=/tmp/pip_cache --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 验证 PyTorch 安装
echo -e "\n${YELLOW}验证 PyTorch 安装...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 安装 ffmpeg
echo -e "\n${YELLOW}步骤 5: 安装 ffmpeg${NC}"
run_conda_cmd install ffmpeg -y

# 安装其他 Python 包
echo -e "\n${YELLOW}步骤 6: 安装其他依赖包${NC}"
PYTHON_PACKAGES=(decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets)
pip_install_with_mirror "${PYTHON_PACKAGES[@]}"

# 验证关键依赖
echo -e "\n${YELLOW}步骤 7: 验证关键依赖${NC}"
python -c "
import torch
import torchvision
import diffusers
import transformers
print('✓ torch:', torch.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ diffusers:', diffusers.__version__)
print('✓ transformers:', transformers.__version__)
print('')
print('所有关键依赖安装成功！')
"

echo -e "\n${GREEN}=========================================="
echo "✓ NAS 共享环境 nwm3 配置完成！"
echo "==========================================${NC}"
echo -e "${YELLOW}激活环境:${NC} conda activate $ENV_PATH"
