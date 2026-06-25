#!/bin/bash
# 上传 recon_dataset.tar.gz 压缩文件到 NAS1
# 使用方法: bash upload_tar_to_nas1.sh

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
LOCAL_TAR="/Volumes/T7/recon_dataset.tar.gz"
REMOTE_USER="xiaoyj25"
REMOTE_HOST="cluster44"
REMOTE_DEST="/DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz"
REMOTE_EXTRACT_DIR="/DATA/DATANAS1/xiaoyj25/recon_release"

echo -e "${BLUE}=========================================="
echo "上传 recon_dataset.tar.gz 到 NAS1"
echo "==========================================${NC}"

# 检查本地压缩文件
echo -e "\n${YELLOW}步骤 1: 检查本地压缩文件${NC}"
if [ ! -f "$LOCAL_TAR" ]; then
    echo -e "${RED}错误: 压缩文件不存在: $LOCAL_TAR${NC}"
    exit 1
fi

# 获取文件大小
if command -v stat &> /dev/null; then
    FILE_SIZE=$(stat -f%z "$LOCAL_TAR" 2>/dev/null || stat -c%s "$LOCAL_TAR" 2>/dev/null)
    FILE_SIZE_HR=$(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$(($FILE_SIZE / 1024 / 1024 / 1024))GB")
    echo -e "${GREEN}✓ 文件大小: $FILE_SIZE_HR${NC}"
else
    FILE_SIZE_HR=$(du -sh "$LOCAL_TAR" | cut -f1)
    echo -e "${GREEN}✓ 文件大小: $FILE_SIZE_HR${NC}"
fi

# 确认上传
echo -e "\n${YELLOW}步骤 2: 确认上传信息${NC}"
echo -e "本地文件: ${GREEN}$LOCAL_TAR${NC}"
echo -e "远程目标: ${GREEN}$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST${NC}"
echo -e "文件大小: ${GREEN}$FILE_SIZE_HR${NC}"
echo ""
read -p "确认开始上传? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消上传${NC}"
    exit 0
fi

# 创建远程目录
echo -e "\n${YELLOW}步骤 3: 创建远程目录${NC}"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $(dirname $REMOTE_DEST)"
echo -e "${GREEN}✓ 远程目录已准备${NC}"

# 使用 rsync 上传压缩文件
echo -e "\n${YELLOW}步骤 4: 开始上传压缩文件${NC}"
echo -e "${BLUE}提示: 上传单个大文件比上传大量小文件更稳定快速${NC}"
echo -e "${BLUE}提示: 可以使用 Ctrl+C 中断，下次运行会自动续传${NC}"
echo -e "${BLUE}提示: 上传完成后可以选择在服务器上解压${NC}"
echo ""

# 使用 rsync 上传，显示详细进度
# -P: 等同于 --progress --partial（显示进度并支持断点续传）
# -h: 人类可读的文件大小
rsync -avzhP "$LOCAL_TAR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST"

# 保存 rsync 的退出状态
RSYNC_EXIT=$?

if [ $RSYNC_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}=========================================="
    echo "✓ 上传完成！"
    echo "==========================================${NC}"
    
    # 询问是否在服务器上解压
    echo ""
    read -p "是否在服务器上解压文件? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n${YELLOW}步骤 5: 在服务器上解压文件${NC}"
        echo -e "${BLUE}解压目标目录: $REMOTE_EXTRACT_DIR${NC}"
        echo -e "${BLUE}这可能需要一些时间，请耐心等待...${NC}"
        echo ""
        
        ssh "$REMOTE_USER@$REMOTE_HOST" "
            mkdir -p $REMOTE_EXTRACT_DIR
            cd $(dirname $REMOTE_DEST)
            echo '开始解压...'
            tar -xzf recon_dataset.tar.gz -C $REMOTE_EXTRACT_DIR --strip-components=0
            if [ \$? -eq 0 ]; then
                echo '解压完成！'
                echo '解压后的文件数量:'
                find $REMOTE_EXTRACT_DIR -name '*.hdf5' 2>/dev/null | wc -l
                echo '解压后的目录大小:'
                du -sh $REMOTE_EXTRACT_DIR
            else
                echo '解压失败，请检查错误信息'
                exit 1
            fi
        "
        
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}=========================================="
            echo "✓ 解压完成！"
            echo "==========================================${NC}"
            echo -e "\n${YELLOW}下一步操作:${NC}"
            echo "1. SSH 登录服务器: ssh $REMOTE_USER@$REMOTE_HOST"
            echo "2. 检查解压后的文件: ls -lh $REMOTE_EXTRACT_DIR | head -20"
            echo "3. 验证文件数量: find $REMOTE_EXTRACT_DIR -name '*.hdf5' | wc -l"
            echo "4. 创建软链接: cd ~/nwm && ln -s $REMOTE_EXTRACT_DIR data/recon_release"
        else
            echo -e "\n${YELLOW}解压过程出现问题，但文件已成功上传${NC}"
            echo "您可以稍后手动解压:"
            echo "  ssh $REMOTE_USER@$REMOTE_HOST"
            echo "  cd $(dirname $REMOTE_DEST)"
            echo "  tar -xzf recon_dataset.tar.gz -C $REMOTE_EXTRACT_DIR"
        fi
    else
        echo -e "\n${YELLOW}文件已上传，但未解压${NC}"
        echo "您可以稍后手动解压:"
        echo "  ssh $REMOTE_USER@$REMOTE_HOST"
        echo "  cd $(dirname $REMOTE_DEST)"
        echo "  mkdir -p $REMOTE_EXTRACT_DIR"
        echo "  tar -xzf recon_dataset.tar.gz -C $REMOTE_EXTRACT_DIR"
    fi
else
    echo -e "\n${RED}=========================================="
    echo "✗ 上传失败，请检查错误信息"
    echo "==========================================${NC}"
    echo -e "\n${YELLOW}提示: 如果上传中断，可以重新运行脚本继续传输${NC}"
    exit 1
fi

