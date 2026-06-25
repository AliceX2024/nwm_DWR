#!/bin/bash
# 上传预训练权重到服务器的脚本
# 使用方法: bash upload_pretrained.sh

LOCAL_CHECKPOINT="/Users/xiaoyj/Desktop/pretrained/cdit_l_100000.pth.tar"
SERVER_USER="xiaoyj25"
SERVER_HOST="cluster44"
SERVER_DIR="/villa/xiaoyj25/nwm/logs/nwm_cdit_l/checkpoints"

echo "=========================================="
echo "上传预训练权重到服务器"
echo "=========================================="
echo "本地文件: $LOCAL_CHECKPOINT"
echo "目标位置: $SERVER_USER@$SERVER_HOST:$SERVER_DIR"
echo ""

# 检查本地文件是否存在
if [ ! -f "$LOCAL_CHECKPOINT" ]; then
    echo "错误: 本地文件不存在: $LOCAL_CHECKPOINT"
    exit 1
fi

# 创建服务器目录（如果不存在）
echo "在服务器上创建目录..."
ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p $SERVER_DIR"

# 上传文件
echo "开始上传..."
rsync -avzhP "$LOCAL_CHECKPOINT" ${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/0100000.pth.tar

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "上传成功！"
    echo "=========================================="
    echo "文件已保存到: $SERVER_DIR/0100000.pth.tar"
    echo ""
    echo "注意: 文件名已重命名为 0100000.pth.tar (标准格式)"
    echo "这是 CDiT-L 模型的 100k 步权重"
else
    echo "上传失败，请检查网络连接和权限"
    exit 1
fi

