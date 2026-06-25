#!/bin/bash
# 简化版上传脚本 - 使用最基本的 rsync 选项
# 使用方法: bash upload_to_nas1_simple.sh

# 配置变量
LOCAL_SOURCE="/Volumes/T7/recon_release"
REMOTE_USER="xiaoyj25"
REMOTE_HOST="cluster44"
REMOTE_DEST="/DATA/DATANAS1/xiaoyj25/recon_release"

echo "=========================================="
echo "上传 recon_release 到 NAS1 (简化版)"
echo "=========================================="

# 检查本地源目录
if [ ! -d "$LOCAL_SOURCE" ]; then
    echo "错误: 本地目录不存在: $LOCAL_SOURCE"
    exit 1
fi

# 统计文件数量
FILE_COUNT=$(find "$LOCAL_SOURCE" -name "*.hdf5" | wc -l | tr -d ' ')
echo "找到 $FILE_COUNT 个 .hdf5 文件"

# 计算目录大小
if command -v du &> /dev/null; then
    DIR_SIZE=$(du -sh "$LOCAL_SOURCE" | cut -f1)
    echo "目录大小: $DIR_SIZE"
fi

# 确认上传
echo ""
echo "本地源: $LOCAL_SOURCE"
echo "远程目标: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST"
echo "文件数量: $FILE_COUNT 个 .hdf5 文件"
echo ""
read -p "确认开始上传? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消上传"
    exit 0
fi

# 创建远程目录
echo ""
echo "创建远程目录..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $(dirname $REMOTE_DEST)"
echo "远程目录已准备"

# 检查 rsync 命令
echo ""
echo "检查 rsync 命令..."
which rsync
rsync --version | head -1

# 使用 rsync 上传
echo ""
echo "开始传输..."
echo "提示: 对于大量文件，rsync 会先扫描文件列表，然后开始传输"
echo "提示: 传输过程中会显示每个文件的进度"
echo "提示: 可以使用 Ctrl+C 中断，下次运行会自动续传"
echo ""

# 使用详细模式和进度显示
# -v: 详细输出，显示正在传输的文件
# --progress: 显示每个文件的传输进度
# --partial: 支持断点续传
# -h: 人类可读的文件大小
# 注意: 对于大量文件，rsync 会先扫描，然后开始传输
# 传输过程中会显示每个文件的进度条

rsync -avzh --progress --partial \
    "$LOCAL_SOURCE/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST/"

# 保存 rsync 的退出状态
RSYNC_EXIT=$?

if [ $RSYNC_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 上传完成！"
    echo "=========================================="
    echo ""
    echo "下一步操作:"
    echo "1. SSH 登录服务器: ssh $REMOTE_USER@$REMOTE_HOST"
    echo "2. 检查上传的文件: ls -lh $REMOTE_DEST | head -20"
    echo "3. 验证文件数量: find $REMOTE_DEST -name '*.hdf5' | wc -l"
    echo "4. 创建软链接: cd ~/nwm && ln -s $REMOTE_DEST data/recon_release"
else
    echo ""
    echo "=========================================="
    echo "✗ 上传失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

