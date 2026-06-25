#!/bin/bash
# 使用 conda 配置 NWM 环境的脚本
# 使用方法: bash setup_env_conda.sh

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENV_NAME="nwm3"
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
    if pip3 install -i "$TSINGHUA_PIP_INDEX" --timeout 60 --retries 3 "${packages[@]}"; then
        return 0
    fi
    echo -e "${YELLOW}镜像源安装失败，回退到官方 PyPI...${NC}"
    pip3 install "${packages[@]}"
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
echo "NWM 环境配置脚本 (使用 Conda)"
echo "==========================================${NC}"

# 检查 conda 是否可用
echo -e "\n${YELLOW}步骤 1: 检查 conda${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda 未找到${NC}"
    echo "请先安装 miniconda 或 anaconda"
    echo "如果已安装，请运行: source ~/miniconda3/bin/activate"
    exit 1
fi

CONDA_VERSION=$(conda --version)
echo -e "${GREEN}✓ 找到 conda: $CONDA_VERSION${NC}"

# 初始化 conda shell（确保后续命令可用）
echo -e "\n${YELLOW}初始化 conda shell...${NC}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # 备用方案
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo -e "${RED}✗ 无法初始化 conda shell${NC}"
    echo "请手动运行: source ~/miniconda3/bin/activate"
    exit 1
fi

# 检查环境是否已存在
echo -e "\n${YELLOW}步骤 2: 检查环境是否已存在${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}环境 ${ENV_NAME} 已存在${NC}"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除现有环境...${NC}"
        conda env remove -n $ENV_NAME -y
    else
        echo -e "${YELLOW}使用现有环境${NC}"
        conda activate $ENV_NAME
        echo -e "${GREEN}✓ 环境已激活${NC}"
        exit 0
    fi
fi

# 创建虚拟环境
echo -e "\n${YELLOW}步骤 3: 创建虚拟环境${NC}"
echo -e "${BLUE}创建环境: $ENV_NAME (Python $PYTHON_VERSION)${NC}"
run_conda_cmd create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo -e "\n${YELLOW}步骤 4: 激活环境${NC}"
conda activate $ENV_NAME
echo -e "${GREEN}✓ 环境已激活${NC}"

# 安装 PyTorch (CUDA 12.6)
echo -e "\n${YELLOW}步骤 5: 安装 PyTorch (CUDA 12.6)${NC}"
echo -e "${BLUE}安装 PyTorch nightly 版本...${NC}"
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 验证 PyTorch 安装
echo -e "\n${YELLOW}验证 PyTorch 安装...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch 安装成功${NC}"
else
    echo -e "${RED}✗ PyTorch 安装验证失败${NC}"
    exit 1
fi

# 安装 ffmpeg
echo -e "\n${YELLOW}步骤 6: 安装 ffmpeg${NC}"
run_conda_cmd install ffmpeg -y
echo -e "${GREEN}✓ ffmpeg 安装完成${NC}"

# 安装其他 Python 包
echo -e "\n${YELLOW}步骤 7: 安装其他依赖包${NC}"
echo -e "${BLUE}安装: decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets${NC}"
PYTHON_PACKAGES=(decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets)
pip_install_with_mirror "${PYTHON_PACKAGES[@]}"

# 验证关键依赖
echo -e "\n${YELLOW}步骤 8: 验证关键依赖${NC}"
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

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=========================================="
    echo "✓ 环境配置完成！"
    echo "==========================================${NC}"
    echo ""
    echo -e "${YELLOW}下一步操作:${NC}"
    echo "1. 确保环境已激活: conda activate $ENV_NAME"
    echo "2. 等待数据集传输完成"
    echo "3. 解压数据集（如果上传的是 .tar.gz）"
    echo "4. 创建数据软链接: cd ~/nwm && ln -s /DATA/DATANAS1/xiaoyj25/recon_release data/recon"
    echo "5. 运行单 GPU 测试: python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 100 --eval-every 500 --bfloat16 1 --epochs 1 --torch-compile 0"
else
    echo -e "\n${RED}依赖验证失败，请检查错误信息${NC}"
    exit 1
fi

