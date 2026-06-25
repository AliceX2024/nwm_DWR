#!/bin/bash
# 在服务器端检查上传进度的脚本
# 使用方法: 在服务器上运行此脚本，或直接运行命令

REMOTE_DEST="/DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz"
REMOTE_EXTRACT_DIR="/DATA/DATANAS1/xiaoyj25/recon_release"

echo "=========================================="
echo "检查上传进度"
echo "=========================================="
echo ""

# 检查压缩文件是否存在及大小
if [ -f "$REMOTE_DEST" ]; then
    echo "✓ 压缩文件存在: $REMOTE_DEST"
    FILE_SIZE=$(du -h "$REMOTE_DEST" | cut -f1)
    FILE_SIZE_BYTES=$(stat -c%s "$REMOTE_DEST" 2>/dev/null || stat -f%z "$REMOTE_DEST" 2>/dev/null)
    echo "  文件大小: $FILE_SIZE ($FILE_SIZE_BYTES 字节)"
    
    # 检查文件是否正在写入（通过比较两次读取的大小）
    echo ""
    echo "监控文件增长（每5秒更新一次，按Ctrl+C停止）..."
    PREV_SIZE=0
    while true; do
        CURRENT_SIZE=$(stat -c%s "$REMOTE_DEST" 2>/dev/null || stat -f%z "$REMOTE_DEST" 2>/dev/null)
        CURRENT_SIZE_HR=$(du -h "$REMOTE_DEST" | cut -f1)
        if [ "$CURRENT_SIZE" != "$PREV_SIZE" ]; then
            GROWTH=$((CURRENT_SIZE - PREV_SIZE))
            GROWTH_HR=$(numfmt --to=iec-i --suffix=B $GROWTH 2>/dev/null || echo "${GROWTH} bytes")
            echo "[$(date '+%H:%M:%S')] 文件大小: $CURRENT_SIZE_HR | 增长: $GROWTH_HR"
            PREV_SIZE=$CURRENT_SIZE
        else
            echo "[$(date '+%H:%M:%S')] 文件大小: $CURRENT_SIZE_HR | 无变化（可能传输完成或暂停）"
        fi
        sleep 5
    done
else
    echo "✗ 压缩文件不存在: $REMOTE_DEST"
    echo ""
    echo "检查解压后的目录..."
    
    if [ -d "$REMOTE_EXTRACT_DIR" ]; then
        echo "✓ 解压目录存在: $REMOTE_EXTRACT_DIR"
        DIR_SIZE=$(du -sh "$REMOTE_EXTRACT_DIR" 2>/dev/null | cut -f1)
        FILE_COUNT=$(find "$REMOTE_EXTRACT_DIR" -name "*.hdf5" 2>/dev/null | wc -l)
        echo "  目录大小: $DIR_SIZE"
        echo "  .hdf5 文件数量: $FILE_COUNT"
    else
        echo "✗ 解压目录不存在: $REMOTE_EXTRACT_DIR"
    fi
fi








