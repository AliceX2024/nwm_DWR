# Navigation World Model 复现指南 - Cluster 44

本指南提供在服务器 cluster 44 上从零开始复现 Navigation World Model 论文的详细步骤。

## 环境信息
- 服务器: cluster44 (A100-40G x 7)
- 代码位置: `~/nwm`
- 数据集: recon_dataset.tar.gz (本地路径: `/Volumes/T7/recon_dataset.tar.gz`)

---

## 步骤 1: 上传数据集到服务器

### 1.1 从本地传输数据集到服务器

#### 选项 A: 上传 .tar.gz 压缩文件到 NAS1（**强烈推荐**）

**推荐理由：**
- ✅ 传输更稳定：单个大文件比 11837 个小文件更可靠
- ✅ 传输更快：避免 rsync 扫描大量文件的开销
- ✅ 更易续传：中断后更容易恢复
- ✅ 服务器解压更快：在服务器上解压通常比传输大量小文件更快

**在本地终端执行：**

```bash
# 使用提供的上传脚本（推荐）
cd ~/nwm
bash upload_tar_to_nas1.sh
```

脚本会自动：
1. 上传 `/Volumes/T7/recon_dataset.tar.gz` 到 `/DATA/DATANAS1/xiaoyj25/`
2. 询问是否在服务器上自动解压
3. 解压到 `/DATA/DATANAS1/xiaoyj25/recon_release/`

**手动上传（如果不想使用脚本）：**

```bash
# 上传压缩文件
rsync -avzhP /Volumes/T7/recon_dataset.tar.gz \
    xiaoyj25@cluster44:/DATA/DATANAS1/xiaoyj25/

# SSH 登录服务器后解压
ssh xiaoyj25@cluster44
cd /DATA/DATANAS1/xiaoyj25/
mkdir -p recon_release
tar -xzf recon_dataset.tar.gz -C recon_release
```

**注意**: 
- 确保 `/DATA/DATANAS1/xiaoyj25/` 目录存在且有写权限
- 上传过程会显示实时进度
- 如果上传中断，可以重新运行命令，rsync 会自动续传

#### 选项 B: 上传解压后的 recon_release 文件夹到 NAS1（不推荐，仅当没有压缩文件时使用）

如果您只有解压后的 `recon_release` 文件夹（包含大量 .hdf5 文件），可以直接上传：

**在本地终端执行：**

```bash
# 使用提供的上传脚本
cd ~/nwm
bash upload_to_nas1_simple.sh

# 或手动使用 rsync 上传（支持断点续传）
rsync -avzh --progress --partial \
    /Volumes/T7/recon_release/ \
    xiaoyj25@cluster44:/DATA/DATANAS1/xiaoyj25/recon_release/
```

**注意**: 
- 对于大量文件（如 11837 个），rsync 会先扫描所有文件，这个过程可能较慢
- 传输过程中可能不会实时显示进度
- 如果上传中断，可以重新运行命令，rsync 会自动续传

#### 选项 C: 上传压缩包到 NAS2（如果 NAS2 有空间）

在**本地终端**执行：

```bash
# 使用 scp 上传（如果文件较大，建议使用 rsync 支持断点续传）
rsync -avz --progress /Volumes/T7/recon_dataset.tar.gz xiaoyj25@cluster44:/DATA/DATANAS2/xiaoyj25/

# 或者使用 scp（如果 rsync 不可用）
scp /Volumes/T7/recon_dataset.tar.gz xiaoyj25@cluster44:/DATA/DATANAS2/xiaoyj25/
```

**注意**: 确保 `/DATA/DATANAS2/xiaoyj25/` 目录存在且有写权限。如果不存在，需要先创建。

### 1.2 在服务器上解压数据集

SSH 登录到服务器后：

```bash
# 登录服务器
ssh xiaoyj25@cluster44

# 进入数据集目录
cd /DATA/DATANAS2/xiaoyj25/

# 解压数据集（如果数据集已经处理过，可能不需要解压，直接使用）
# 根据实际情况，可能需要先查看 tar.gz 内容
tar -tzf recon_dataset.tar.gz | head -20  # 查看压缩包内容

# 解压数据集
tar -xzf recon_dataset.tar.gz

# 查看解压后的目录结构
ls -la
```

### 1.3 数据预处理（如果需要）

**重要**: 根据 README，数据集需要按照 [NoMaD](https://github.com/robodhruv/visualnav-transformer) 的方式预处理。

如果您的 `recon_dataset.tar.gz` 是原始数据（ROS bag 文件等），需要先预处理：

```bash
# 1. 克隆 NoMaD 仓库（如果还没有）
cd ~
git clone https://github.com/robodhruv/visualnav-transformer.git


# 2. 修改预处理分辨率（从 160x120 改为 320x240）
# 编辑文件：train/vint_train/data/data_utils.py
# 找到第 13 行，将分辨率改为 (320, 240)

# 3. 运行预处理脚本
# 对于 recon 数据集，使用 process_recon.py
# 对于其他数据集（如 bag 文件），使用 process_bags.py

# 示例（根据实际路径调整）：
python process_recon.py \
    --input_dir /DATA/DATANAS2/xiaoyj25/recon_raw \
    --output_dir /DATA/DATANAS2/xiaoyj25/recon_processed \
    --resolution 320 240
```

**注意**: 
- 如果您的 `recon_dataset.tar.gz` 已经是预处理好的数据（包含 `traj_data.pkl` 和 `.jpg` 文件），可以跳过此步骤
- 预处理后的数据应该包含每个轨迹目录，每个目录下有编号的图片文件（0.jpg, 1.jpg, ...）和 `traj_data.pkl` 文件

### 1.4 创建软链接到项目目录

```bash
# 进入项目目录
cd ~/nwm

# 创建 data 目录（如果不存在）
mkdir -p data

# 根据数据存储位置创建软链接：

# 如果数据在 NAS1（recon_release 文件夹）
ln -s /DATA/DATANAS1/xiaoyj25/recon_release data/recon

重新建立软连接
ln -s /DATA/DATANAS2/xiaoyj25/recon_release data/recon

# 或者如果数据在 NAS2（已解压的 recon 目录）
# ln -s /DATA/DATANAS2/xiaoyj25/recon data/recon

# 或者如果预处理后的目录名不同：
# ln -s /DATA/DATANAS2/xiaoyj25/recon_processed data/recon
#实际上处理后的数据在
ln -s /villa/xiaoyj25/visualnav-transformer/datasets/recon_processed data/recon

# 验证软链接
ls -la data/
ls data/recon | head
```

**重要**: 确保软链接后的目录结构符合要求：
```
data/recon/
├── <traj_name1>/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   ├── T_1.jpg
│   └── traj_data.pkl
├── <traj_name2>/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   ├── T_2.jpg
│   └── traj_data.pkl
└── ...
```

---

## 步骤 2: 配置虚拟环境

### 2.1 检查 conda 是否安装

```bash
# 检查 conda 是否可用
which conda
conda --version
25.9.1
# 如果 conda 未找到，需要先安装 miniconda
# 下载地址: https://docs.conda.io/en/latest/miniconda.html
# 安装后运行: source ~/miniconda3/bin/activate
```

### 2.2 创建虚拟环境

**方法 1: 使用自动配置脚本（推荐）**

```bash
# 确保 conda 已激活
source ~/miniconda3/bin/activate

# 运行配置脚本
cd ~/nwm
bash setup_env_conda.sh
```

脚本会自动：
- 创建 nwm 环境
- 安装 PyTorch (CUDA 12.6)
- 安装所有依赖包
- 验证安装

**方法 2: 手动创建环境**

```bash
# 确保 conda 已激活
source ~/miniconda3/bin/activate

# 创建环境
conda create -n nwm python=3.10 -y

# 激活环境
conda activate nwm
```

### 2.3 安装 PyTorch（CUDA 12.6）

```bash
# 确保在 nwm 环境中
conda activate nwm

# 安装 PyTorch nightly 版本（CUDA 12.6）
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 验证安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### 2.4 安装其他依赖

```bash
# 确保在 nwm 环境中
conda activate nwm

# 安装 ffmpeg（使用 conda）
conda install ffmpeg -y

# 安装 Python 包
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets

# 可选：如果需要使用 Jupyter notebook
pip3 install jupyter ipykernel
python -m ipykernel install --user --name nwm --display-name "Python (nwm)"
```

### 2.5 验证环境

```bash
# 测试关键依赖
python -c "import torch; import torchvision; import diffusers; import transformers; print('All dependencies installed successfully!')"
```

---

## 步骤 3: 验证数据集结构

### 3.1 检查数据集路径和结构

```bash
cd ~/nwm

# 检查软链接
ls -la data/

# 检查数据集结构（应该看到轨迹目录）
ls data/recon/ | head -5

# 检查单个轨迹目录结构
ls data/recon/<某个轨迹名>/ | head -10

# 验证 traj_data.pkl 文件存在
find data/recon -name "traj_data.pkl" | head -3
```

### 3.2 检查配置文件

```bash
# 查看训练配置文件中的数据集路径
cat config/nwm_cdit_xl.yaml | grep -A 5 "recon:"

# 确认路径为 data/recon（这是相对路径，相对于项目根目录）
```

---

## 步骤 4: 测试运行（单 GPU 调试）

### 4.1 下载 VAE 模型（必需）

训练需要从 Hugging Face 下载 VAE 模型。如果服务器无法访问 Hugging Face，请先下载：

```bash
# 方法 1: 使用下载脚本（尝试镜像）
cd ~/nwm
bash download_vae_model.sh

bash download_vae_local.sh

# 方法 2: 手动下载（如果有网络）
conda activate nwm
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-vae-ft-ema')"

# 方法 3: 使用 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')"
```

**注意**: 如果服务器完全无法访问外网，需要：
1. 在有网络的机器上下载模型
2. 上传到服务器的 `~/.cache/huggingface/hub/` 目录
3. 或者设置环境变量 `VAE_MODEL_PATH` 指向本地模型路径

download_vae_model.sh 把 VAE 缓存到默认的 HF cache 目录 ${HOME}/.cache/huggingface/hub，模型名是 stabilityai/sd-vae-ft-ema，因此本地路径格式一般是：
/villa/xiaoyj25/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/<hash>/
确认缓存路径（默认 ~/.cache/huggingface/hub）
export HF_ENDPOINT=https://hf-mirror.com   # 可保留，也可不设
export VAE_MODEL_PATH=stabilityai/sd-vae-ft-ema  # 或者指向本地目录


# 可选：强制离线优先
export HF_HUB_OFFLINE=1
# 或如果仍需联网，设置镜像
export HF_ENDPOINT=https://hf-mirror.com# 可选：强制离线优先
export HF_HUB_OFFLINE=1

### 4.2 单 GPU 训练测试

```bash
cd ~/nwm
conda activate nwm
 Cmd+Shift+P，Python: Select Interpreter，选择~/miniconda3/envs/nwm/bin/python
# 单 GPU 训练测试（小批量，快速验证）
python train.py \
    --config config/nwm_recon_only.yaml \
    --ckpt-every 100 \
    --eval-every 500 \
    --bfloat16 1 \
    --epochs 1 \
    --torch-compile 0
```

**注意**: 
- 如果遇到 CUDA 内存不足，可以减小 batch_size（在配置文件中）
- 如果遇到数据集路径问题，检查软链接和目录结构

### 4.2 检查日志

```bash
# 查看训练日志
ls -la logs/nwm_cdit_xl/

# 查看最新的 checkpoint
ls -la logs/nwm_cdit_xl/checkpoints/
```

---

## 步骤 5: 多 GPU 训练（7 个 A100）

### 5.1 使用 torchrun（单节点多 GPU）

```bash
cd ~/nwm
conda activate nwm

# 使用 7 个 GPU（根据实际可用 GPU 数量调整）
torchrun \
  --nproc-per-node=7 \
  train.py \
  --config config/nwm_cdit_xl.yaml \
  --ckpt-every 2000 \
  --eval-every 10000 \
  --bfloat16 1 \
  --epochs 300 \
  --torch-compile 0
```

### 5.2 使用 SLURM（如果集群使用 SLURM）

```bash
# 查看是否有 submitit_train_cw.py
cat submitit_train_cw.py | head -20

# 如果使用 SLURM，可能需要修改脚本中的分区和 QoS 设置
# 然后运行：
python submitit_train_cw.py \
  --nodes 1 \
  --partition <your_partition> \
  --qos <your_qos> \
  --config config/nwm_cdit_xl.yaml \
  --ckpt-every 2000 \
  --eval-every 10000 \
  --bfloat16 1 \
  --epochs 300 \
  --torch-compile 0
```

---

## 步骤 6: 常见问题排查

### 6.1 数据集路径问题

```bash
# 如果遇到 "FileNotFoundError" 或路径错误：
# 1. 检查软链接
ls -la data/recon

# 2. 检查实际路径
readlink -f data/recon

# 3. 检查配置文件中的路径
grep -r "data_folder" config/nwm_cdit_xl.yaml
```

### 6.2 CUDA 相关问题

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 6.3 依赖包问题

```bash
# 如果某个包导入失败，重新安装
pip3 install <package_name> --upgrade

# 检查已安装的包
pip list | grep <package_name>
```

### 6.4 内存不足

```bash
# 在配置文件中减小 batch_size
# 编辑 config/nwm_cdit_xl.yaml，将 batch_size 从 16 改为更小的值（如 8 或 4）
```

---

## 步骤 7: 评估模型（可选）

### 7.1 下载预训练模型

```bash
# 从 Hugging Face 下载预训练模型
# 参考 README.md 中的说明
mkdir -p logs/nwm_cdit_xl/checkpoints
# 下载模型到该目录
```

### 7.2 运行评估

```bash
# 设置结果目录
export RESULTS_FOLDER=/path/to/results

# 准备 ground truth
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1

# 运行推理
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}

# 计算指标
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
    --eval_types time
```

---

## 快速检查清单

- [ ] 数据集已上传到 `/DATA/DATANAS1/xiaoyj25/recon_dataset.tar.gz` 或 `/DATA/DATANAS1/xiaoyj25/recon_release/` 或 `/DATA/DATANAS2/xiaoyj25/`
- [ ] 数据集已解压（如果上传的是压缩包）
- [ ] 数据集已预处理（如果原始数据需要预处理）
- [ ] 数据集结构正确（包含轨迹目录、编号图片文件和 traj_data.pkl 文件，或 .hdf5 文件）
- [ ] 软链接已创建：`~/nwm/data/recon -> /DATA/DATANAS1/xiaoyj25/recon_release` 或相应路径
- [ ] conda 环境已创建并激活
- [ ] PyTorch 已安装且 CUDA 可用
- [ ] 所有依赖包已安装
- [ ] 单 GPU 测试运行成功
- [ ] 配置文件路径正确

---

## 参考资源

- 项目 GitHub: https://github.com/facebookresearch/nwm
- 论文: https://arxiv.org/abs/2412.03572
- 预训练模型: https://huggingface.co/facebook/nwm
- NoMaD 数据预处理: https://github.com/robodhruv/visualnav-transformer

---

## 注意事项

1. **数据集预处理**: 根据 README，数据集需要按照 NoMaD 的方式预处理，分辨率为 (320, 240)。如果您的数据集还未预处理，需要先运行预处理脚本。

2. **存储空间**: 确保 `/DATA/DATANAS2/xiaoyj25/` 有足够的存储空间。

3. **GPU 内存**: A100-40G 应该足够，但如果遇到 OOM，减小 batch_size。

4. **训练时间**: 完整训练可能需要较长时间，建议使用 screen 或 tmux 保持会话。

5. **检查点**: 定期检查 checkpoints 是否正常保存。

修改：


train:

Dataset size = 350,699 images
Gradient Accumulation Steps = 16
Effective Batch = 256
world_size = 4

单卡batchsize 4，梯度累计16，等效batch size 64.四卡 256


每个epoch的step： 350699/256=1370
