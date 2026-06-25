#!/bin/bash
# 检查上传状态的脚本
# 在本地运行，自动检查服务器端状态

REMOTE_USER="xiaoyj25"
REMOTE_HOST="cluster44"
REMOTE_DEST="/DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz"
REMOTE_DIR="/DATA/DATANAS1/xiaoyj25"

echo "=========================================="
echo "检查上传状态"
echo "=========================================="
echo ""

# 检查目标目录是否存在
echo "1. 检查目标目录..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -ld $REMOTE_DIR 2>&1"
echo ""

# 检查目标目录中的所有文件
echo "2. 检查目标目录中的所有文件..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lh $REMOTE_DIR/ 2>&1"
echo ""

# 检查压缩文件是否存在
echo "3. 检查压缩文件..."
ssh "$REMOTE_USER@$REMOTE_HOST" "
    if [ -f '$REMOTE_DEST' ]; then
        echo '✓ 文件存在'
        ls -lh '$REMOTE_DEST'
        stat '$REMOTE_DEST' 2>/dev/null || echo '无法获取详细信息'
    else
        echo '✗ 文件不存在'
        echo '检查是否有部分文件（.tar.gz.part 等）...'
        ls -lh ${REMOTE_DEST}* 2>/dev/null || echo '没有找到部分文件'
    fi
"
echo ""

# 检查是否有 rsync 进程在运行
echo "4. 检查是否有上传进程在运行..."
ssh "$REMOTE_USER@$REMOTE_HOST" "ps aux | grep -E 'rsync|scp' | grep -v grep || echo '没有找到上传进程'"
echo ""

# 检查磁盘空间
echo "5. 检查磁盘空间..."
ssh "$REMOTE_USER@$REMOTE_HOST" "df -h $REMOTE_DIR"
echo ""

echo "=========================================="
echo "检查完成"
echo "=========================================="








