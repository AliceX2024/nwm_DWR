#!/bin/bash
# 下载 VAE 模型到本地项目目录的脚本
# 使用方法: 在有网络的节点运行: bash download_vae_local.sh

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_NAME="stabilityai/sd-vae-ft-ema"
# 【关键修改】将模型保存到项目下的 models 文件夹，而不是系统缓存
SAVE_DIR="$(pwd)/models/sd-vae-ft-ema"

echo -e "${BLUE}=========================================="
echo "下载 VAE 模型: $MODEL_NAME"
echo "保存目标路径: $SAVE_DIR"
echo "==========================================${NC}"

# 检查是否有网络 (简单 ping 测试)
ping -c 1 hf-mirror.com > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}警告: 当前节点似乎无法访问互联网。${NC}"
    echo -e "${YELLOW}请务必在【登录节点】或【有外网权限的节点(如cluster44)】上运行此脚本！${NC}"
    echo -e "下载到共享存储后，Cluster41 即可直接读取。"
    # 这里不退出，因为有时候 ping 不通但能 curl
fi

# 尝试使用 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 激活 conda 环境 (自动判断 nwm2 或 nwm3)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    # 尝试激活当前正在使用的环境，如果没激活则默认尝试 nwm3
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        conda activate nwm3 2>/dev/null || conda activate nwm 2>/dev/null
    fi
    echo "当前环境: $CONDA_DEFAULT_ENV"
fi

# 使用 Python 下载并保存
python3 << EOF
import os
import shutil
# 强制使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import AutoencoderKL

save_directory = "$SAVE_DIR"

try:
    print(f"准备下载模型到: {save_directory}")
    
    # 1. 下载模型
    # 注意：这里我们不只是下载到缓存，而是加载后直接 save_pretrained 到指定文件夹
    # 这样得到的是标准的本地模型结构 (config.json, diffusion_pytorch_model.bin 等)
    print("正在连接镜像站下载...")
    vae = AutoencoderKL.from_pretrained("$MODEL_NAME")
    
    # 2. 保存到本地目录
    print("正在保存到本地目录...")
    os.makedirs(save_directory, exist_ok=True)
    vae.save_pretrained(save_directory)
    
    print(f"✓ 模型已成功保存到: {save_directory}")
    
    # 3. 验证本地加载
    print("验证本地加载...")
    test_load = AutoencoderKL.from_pretrained(save_directory, local_files_only=True)
    print("✓ 本地加载验证通过！")
    
except Exception as e:
    print(f"${RED}下载失败: {e}${NC}")
    print("请检查网络连接，或确认是否在有网节点运行。")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo " 操作完成！"
    echo " 现在你可以在 cluster41 上使用以下指令启动训练："
    echo ""
    echo " export VAE_MODEL_PATH=$SAVE_DIR"
    echo " torchrun ..."
    echo "==========================================${NC}"
    exit 0
else
    echo -e "${RED}脚本执行失败。${NC}"
    exit 1
fi