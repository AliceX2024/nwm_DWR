# 在服务器端检查上传进度的命令

## 快速检查命令

### 1. 检查压缩文件是否存在及大小

```bash
# SSH 登录服务器
ssh xiaoyj25@cluster44

# 检查文件是否存在
ls -lh /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz

# 或者使用 stat 查看详细信息
stat /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz
```

### 2. 实时监控文件增长（推荐）

```bash
# 方法 1: 使用 watch 命令（每2秒更新）
watch -n 2 'ls -lh /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz'

# 方法 2: 使用循环监控（每5秒更新）
while true; do
    clear
    echo "=== $(date) ==="
    ls -lh /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz
    echo ""
    du -h /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz
    sleep 5
done
```

### 3. 检查文件是否正在写入

```bash
# 检查文件是否被进程占用（如果被占用，说明正在写入）
lsof /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz

# 或者检查 rsync 进程
ps aux | grep rsync
```

### 4. 比较文件大小（本地 vs 远程）

```bash
# 在服务器上查看远程文件大小
du -h /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz

# 在本地查看本地文件大小（在另一个终端）
du -h /Volumes/T7/recon_dataset.tar.gz

# 如果大小相同，说明传输完成
```

### 5. 检查解压后的目录（如果已解压）

```bash
# 检查目录是否存在
ls -ld /DATA/DATANAS1/xiaoyj25/recon_release

# 检查目录大小
du -sh /DATA/DATANAS1/xiaoyj25/recon_release

# 统计 .hdf5 文件数量
find /DATA/DATANAS1/xiaoyj25/recon_release -name "*.hdf5" | wc -l

# 查看目录中的文件（前20个）
ls -lh /DATA/DATANAS1/xiaoyj25/recon_release | head -20
```

### 6. 检查磁盘空间使用情况

```bash
# 检查 NAS1 的磁盘使用情况
df -h /DATA/DATANAS1

# 检查用户目录的磁盘使用情况
du -sh /DATA/DATANAS1/xiaoyj25/*
```

## 一键检查脚本

在服务器上运行：

```bash
# 上传检查脚本到服务器（在本地执行）
scp ~/nwm/check_upload_progress.sh xiaoyj25@cluster44:~/

# 在服务器上运行
ssh xiaoyj25@cluster44
bash ~/check_upload_progress.sh
```

## 最常用的快速检查命令

```bash
# 一行命令：检查文件大小和最后修改时间
ssh xiaoyj25@cluster44 "ls -lh /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz && du -h /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz"

# 一行命令：实时监控（每5秒）
ssh xiaoyj25@cluster44 "watch -n 5 'ls -lh /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz && du -h /DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz'"
```








