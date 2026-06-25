# Go Stanford 数据集上传与处理指南

本指南说明如何将本地 `go_stanford (1).zip` 上传到服务器并进行处理。

---

## 步骤 1: 从本地上传 zip 文件到服务器

在**本地终端**执行：

```bash
# 上传 zip 文件到 cluster43
rsync -avzhP "/Users/xiaoyj/Downloads/go_stanford (1).zip" xiaoyj25@cluster43:/DATA/DATANAS2/xiaoyj25/
```

---

## 步骤 2: SSH 登录服务器并解压

在**本地终端**执行：

```bash
ssh xiaoyj25@cluster43
```

登录后执行：

```bash
cd /DATA/DATANAS2/xiaoyj25

# 解压 zip 文件
unzip "go_stanford (1).zip"

# 查看解压后的目录
ls -la
```

---

## 步骤 3: 创建软链接

在服务器上执行：

```bash
cd /DATA/DATANAS2/xiaoyj25/projects/nwm

# 创建 data 目录（如果不存在）
mkdir -p data

# 创建软链接 (根据实际解压的文件夹名调整)
# 假设解压后文件夹名为 go_stanford
ln -s /DATA/DATANAS2/xiaoyj25/go_stanford data/go_stanford_beforeprocess

# 验证软链接
ls -la data/
```

---

## 步骤 4: 运行 process_bags.py
这是读取.bag文件。
已经是处理过的图片，不需要再
根据你提供的 `process_bags.py`，需要先确认脚本位置。脚本应该在 `~/visualnav-transformer/vint_train/process_data/` 目录下。

首先确认 go_stanford 数据集对应的配置（检查 `~/visualnav-transformer/vint_train/process_data/process_bags_config.yaml`）：

```bash
cd /DATA/DATANAS2/xiaoyj25/projects/nwm

# 查看配置文件 (先检查 visualnav-transformer 是否存在)
ls ~/visualnav-transformer/vint_train/process_data/
cat ~/visualnav-transformer/vint_train/process_data/process_bags_config.yaml
```

然后运行处理脚本：

```bash
# 激活 conda 环境
source ~/miniconda3/bin/activate
conda activate nwm

# 运行 process_bags.py (使用完整路径)
python ~/visualnav-transformer/vint_train/process_data/process_bags.py \
    --dataset-name go_stanford \
    --input-dir /DATA/DATANAS2/xiaoyj25/go_stanford \
    --output-dir /DATA/DATANAS2/xiaoyj25/go_stanford_processed \
    --num-trajs -1 \
    --sample-rate 4.0
```

**注意**: 
- `--dataset-name` 必须是 `process_bags_config.yaml` 中已配置的 dataset name
- 如果 go_stanford 不在配置文件中，需要先添加配置
- 需要安装 rosbag 包: `pip install rosbag`

---

# 查看 data 目录下的所有内容（包括软链接）
ls -la /DATA/DATANAS2/xiaoyj25/projects/nwm/data/

# 查看软链接指向的目标
ls -l /DATA/DATANAS2/xiaoyj25/projects/nwm/data/stanford

# 或者直接 readlink
readlink /DATA/DATANAS2/xiaoyj25/projects/nwm/data/stanford


# 删除指定的软链接（不是目标文件）
rm /DATA/DATANAS2/xiaoyj25/projects/nwm/data/stanford

# 或者用 unlink
unlink /DATA/DATANAS2/xiaoyj25/projects/nwm/data/stanford

## 步骤 5: 运行 data_split.py

data_split.py 应该在 `~/visualnav-transformer/` 目录下。

```bash
cd /DATA/DATANAS2/xiaoyj25/projects/nwm

# 运行 data_split.py (使用完整路径)
python ~/visualnav-transformer/vint_train/data/data_split.py \
    --data-dir /DATA/DATANAS2/xiaoyj25/go_stanford_processed \
    --dataset-name go_stanford \
    --split 0.8 \
    --data-splits-dir data_splits
```

---

## 步骤 6: 创建最终软链接

```bash
cd /DATA/DATANAS2/xiaoyj25/projects/nwm

# 删除之前的软链接并重新创建
rm -f data/go_stanford
ln -s /DATA/DATANAS2/xiaoyj25/go_stanford_processed data/go_stanford

# 验证
ls -la data/go_stanford | head
```

---

## 注意事项

1. **process_bags.py 需要 rosbag**: 确保已安装 rosbag 包
2. **数据集配置**: 如果 go_stanford 不在 `process_bags_config.yaml` 中，需要添加对应配置
3. **路径根据实际情况调整**: 如果解压后的文件夹名不同，请相应修改命令
4. **vint_train 目录**: 脚本位于 `~/visualnav-transformer/vint_train/` 下，需要先克隆 visualnav-transformer 仓库（如果还没有）

---

## 快速检查清单

- [ ] zip 文件已上传到 `/DATA/DATANAS2/xiaoyj25/`
- [ ] zip 文件已解压
- [ ] 软链接 `data/go_stanford` 已创建
- [ ] `process_bags.py` 已运行，数据已处理
- [ ] `data_split.py` 已运行，数据已分割
- [ ] 最终软链接指向处理后的数据
