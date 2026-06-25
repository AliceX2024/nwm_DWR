#!/bin/bash
# 下载 VAE 模型的脚本
# 使用方法: bash download_vae_model.sh

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_NAME="stabilityai/sd-vae-ft-ema"
CACHE_DIR="${HOME}/.cache/huggingface/hub"

echo -e "${BLUE}=========================================="
echo "下载 VAE 模型: $MODEL_NAME"
echo "==========================================${NC}"

# 方法 1: 尝试使用 Hugging Face 镜像
echo -e "\n${YELLOW}方法 1: 尝试使用 Hugging Face 镜像...${NC}"
export HF_ENDPOINT=https://hf-mirror.com

# 激活 conda 环境
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate nwm 2>/dev/null || echo "环境 nwm 未激活，请手动激活"
fi

# 使用 Python 下载
python3 << EOF
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL

try:
    print("尝试从镜像下载模型...")
    model_path = snapshot_download(
        repo_id="$MODEL_NAME",
        cache_dir="$CACHE_DIR",
        local_files_only=False
    )
    print(f"模型下载成功: {model_path}")
    
    # 验证模型可以加载
    print("验证模型...")
    vae = AutoencoderKL.from_pretrained("$MODEL_NAME")
    print("✓ 模型验证成功！")
except Exception as e:
    print(f"镜像下载失败: {e}")
    print("请尝试方法 2: 手动下载")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 模型下载成功！${NC}"
    exit 0
fi

echo -e "\n${YELLOW}方法 2: 手动下载说明${NC}"
echo "如果镜像也无法访问，请："
echo "1. 在有网络的机器上下载模型"
echo "2. 上传到服务器的 Hugging Face 缓存目录"
echo ""
echo "模型文件位置: $CACHE_DIR/models--stabilityai--sd-vae-ft-ema"
echo ""
echo "或者使用以下命令手动下载（需要网络）："
echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_NAME')\""


