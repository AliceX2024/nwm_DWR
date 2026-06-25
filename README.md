# Stable Navigation World Model (Stable-NWM)

基于 Diffusion 的导航世界模型，加入了 **spatiotemporal stable reweighting(SSR)** 机制和全局特征库，用于提升机器人视觉导航的轨迹预测和图像生成质量。

## 目录

- [项目简介](#项目简介)
- [核心创新](#核心创新)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [预训练模型](#预训练模型)
- [常见问题](#常见问题)

---

## 项目简介

本项目实现了 Stable Navigation World Model，在原有 CDiT 基础上，**创新性地引入了全局特征库 (Global Feature Bank) 和spatiotemporal stable reweighting 机制，提升了导航世界模型在分布偏移的情况下的预测准确性。
### 主要特性

- **CDiT 架构**: 支持 CDiT-L 和 CDiT-XL 两种模型规模
- **DWR 多样性加权重机制**: 根据样本特征的多样性动态调整训练权重
- **全局特征库 (Global Feature Bank)**: 维护历史特征队列，扩大批次样本量
- **多数据集支持**: recon, go_stanford, tartan_drive, scand, sacson
- **多种评估模式**: Time Prediction, Rollout, Planning

---

## 核心创新

### 全局特征库 (Global Feature Bank)

全局特征库存储历史特征，用于扩大计算样本权重的分母。

**核心功能**：
- 维护历史特征的移动平均队列
- 支持三种更新模式：`random`、`fifo`、`diversity`
- 与当前 Batch 特征拼接，形成更大的样本集合用于权重计算

```python
class GlobalFeatureBank:
    """维护全局特征库，支持与当前 Batch 拼接"""
    def __init__(self, feature_dim, bank_size=2048, device='cuda', mode='random')
    def update(self, batch_features)    # 更新特征库
    def get_combined_features(current_features)  # 获取拼接特征
```

### spatiotemporal stable reweighting(SSR)

SSR 通过优化样本权重，使加权后的特征协方差矩阵非对角线元素最小化，从而提升样本多样性。

**算法流程**：
1. 提取时空联合特征（静态 + 残差 + 时间）
2. 与全局特征库拼接
3. 使用 Log-Weight 参数化确保权重为正
4. 迭代优化使非对角线协方差最小
5. 数值保护 + 归一化 + Clipping

**时空联合特征维度**：

| 池化方式 | 静态维度 | 残差维度 | 时间维度 | 总维度 |
|---------|----------|----------|----------|--------|
| avg (4维) | 4 | 4 | 1 | 9 |
| adaptive_2x2 (16维) | 16 | 16 | 1 | 33 |
| adaptive_4x4 (64维) | 64 | 64 | 1 | 129 |
| raw (3136维) | 3136 | 3136 | 1 | 6273 |

### Reweight 配置参数详解

在 `config/4datasets_clip04_CDiTXL_temmeDWR.yaml` 中可以调节以下参数：

```yaml
reweight:
  enable: True              # 是否启用 reweight 机制
  use_spacetime: True       # 是否使用时空联合特征
  pool_type: adaptive_2x2   # 池化方式: 'avg', 'adaptive_2x2', 'raw'
  start_steps: 0            # Warmup 步数
  time_scale: 4.0           # 时间特征缩放倍数

  # Memory Bank 配置
  memory_mode: 'diversity'  # 更新模式: 'random', 'fifo', 'diversity'
  bank_size: 2048           # 特征库大小

  # DWR 算法参数
  dwr_lr: 0.01              # DWR 内部 Adam 学习率
  dwr_steps: 15             # 迭代优化步数

  # 权重控制
  clip_min: 0.6             # 权重下限
  clip_max: 1.4             # 权重上限
  alpha: 1.0                # 平滑系数

  # 轨迹相关
  group_by_traj: False       # 是否强制同轨迹权重相同
  debug_print_freq: 200     # 调试打印频率
```

**参数调优建议**：

| 参数 | 调优建议 |
|------|----------|
| `dwr_lr` | 学习率过大可能导致权重震荡，建议 0.005-0.02 |
| `dwr_steps` | 步数越多越精确，但训练变慢，建议 10-20 |
| `clip_min/max` | 范围越小约束越强，建议 0.5-2.0 |
| `alpha` | 设为 1.0 使用完整 DWR，设为 0.5 做平滑 |
| `time_scale` | 放大时间特征可增强时间关联性，建议 1.0-4.0 |
| `memory_mode` | `diversity` 模式优先替换相似样本，`random` 更通用 |

---

## 环境配置

### 服务器信息

- 服务器: cluster43/cluster44 (A100-40G x 7)
- 代码位置: `/DATA/DATANAS2/xiaoyj25/projects/nwm`

### 1. 配置虚拟环境

```bash
# 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh

# 运行自动配置脚本（推荐）
cd /DATA/DATANAS2/xiaoyj25/projects/nwm
bash setup_env_conda.sh

# 或手动创建环境
conda create -n nwm python=3.10 -y
conda activate nwm

# 安装 PyTorch (CUDA 12.6)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 安装其他依赖
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets opencv-python
```

### 2. 下载 VAE 模型

```bash
# 使用下载脚本
bash download_vae_model.sh

# 或使用 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
export VAE_MODEL_PATH=stabilityai/sd-vae-ft-ema
```

### 3. 验证环境

```bash
python -c "import torch; import diffusers; print('环境配置成功!')"
```

---

## 数据集准备

### 支持的数据集

| 数据集 | Planning 支持 | 说明 |
|--------|-------------|------|
| recon | ✅ | 主训练数据集 |
| go_stanford | ❌ | 仅支持 time/rollout 评估 |
| tartan_drive | ✅ | 自动驾驶数据集 |
| scand | ✅ | 室内扫描数据集 |
| sacson | ✅ | 室内导航数据集 |

### 数据集上传与配置

```bash
# 1. 本地上传压缩文件到服务器
rsync -avzhP "/path/to/recon_dataset.tar.gz" xiaoyj25@cluster44:/DATA/DATANAS2/xiaoyj25/

# 2. 服务器上解压
ssh xiaoyj25@cluster44
cd /DATA/DATANAS2/xiaoyj25/
tar -xzf recon_dataset.tar.gz

# 3. 创建软链接
cd /DATA/DATANAS2/xiaoyj25/projects/nwm
mkdir -p data
ln -s /DATA/DATANAS2/xiaoyj25/recon_release data/recon

# 4. 验证数据结构
ls data/recon/<轨迹名>/
# 应包含: 0.jpg, 1.jpg, ..., T.jpg, traj_data.pkl
```

### 数据集索引文件

评估前需要生成索引文件 (`time.pkl`, `rollout.pkl`, `navigation_eval.pkl`)，已存放在 `data_splits/` 目录下。

---

## 模型训练

### 基础训练

```bash
# 单 GPU 调试
python train.py \
    --config config/nwm_recon_only.yaml \
    --ckpt-every 100 \
    --eval-every 500 \
    --bfloat16 1 \
    --epochs 1 \
    --torch-compile 0
```

### 多 GPU 训练

```bash
# 7 GPU 训练（推荐配置）
torchrun --standalone --nproc-per-node=7 train.py \
    --config config/nwm_cdit_xl.yaml \
    --ckpt-every 2000 \
    --eval-every 10000 \
    --bfloat16 1 \
    --epochs 300 \
    --torch-compile 0
```

### 微调训练

```bash
# 使用 Reweight 策略微调
torchrun --standalone --nproc-per-node=4 train.py \
    --config config/nwm_recon_finetune.yaml \
    --ckpt-every 1000 \
    --eval-every 10000000 \
    --bfloat16 1 \
    --epochs 1 \
    --torch-compile 0
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | - |
| `--ckpt-every` | 保存 checkpoint 间隔 | 2000 |
| `--eval-every` | 评估间隔 | 10000 |
| `--bfloat16` | 使用 bfloat16 精度 | 1 |
| `--epochs` | 训练轮数 | 300 |
| `--torch-compile` | 使用 torch.compile | 0 |

---

## 模型评估

### 环境变量设置

```bash
export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /DATA/DATANAS2/xiaoyj25/envs/nwm3
```

### Time Prediction 评估（单步预测）

```bash
# 1. 准备 Ground Truth 图像
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1

# 2. 运行模型预测
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}

# 3. 计算评估指标
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_eval \
    --eval_types time
```

### Rollout 评估（轨迹预测）

```bash
# 1. 准备 GT Rollout
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4

# 2. 运行模型 Rollout
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --rollout_fps_values 1,4

# 3. 评估 Rollout
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_eval \
    --eval_types rollout
```

### Planning 评估（CEM 轨迹规划）

> **注意**: `go_stanford` 数据集不支持 Planning 评估。

```bash
# 单卡评测
torchrun --nproc-per-node=1 planning_eval.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 120 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp 0100000 \
    --opt_steps 1 \
    --num_repeat_eval 3

# 8 卡评测（加速）
torchrun --nproc-per-node=8 planning_eval.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 120 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp 0100000 \
    --opt_steps 1 \
    --num_repeat_eval 3
```

### Planning 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_samples` | CEM 采样轨迹数量 | 10 |
| `--topk` | 选择 top-k 最优轨迹 | 5 |
| `--opt_steps` | CEM 优化迭代次数 | 15 |
| `--rollout_stride` | Rollout 步长 | 1 |
| `--num_repeat_eval` | 重复评估次数（降方差） | 1 |

### 评估输出结构

```
${RESULTS_FOLDER}/
├── gt/                           # Ground Truth 图像
│   └── <dataset>/
│       └── time/ 或 rollout/
├── <exp_name>/                   # 实验结果
│   └── <dataset>/
│       └── time/ 或 rollout/ 或 CEM_*/
└── *.json                        # 评测指标文件
```

---

## 预训练模型

### 上传预训练权重

```bash
# 使用上传脚本（推荐）
cd ~/nwm
bash upload_pretrained.sh

# 或手动上传
rsync -avzhP /path/to/pretrained.pth.tar \
    xiaoyj25@cluster44:/villa/xiaoyj25/nwm/logs/nwm_cdit_l/checkpoints/0100000.pth.tar
```

### 可用模型配置

| 模型 | 配置文件 | Checkpoint |
|------|---------|------------|
| CDiT-L (预训练) | `nwm_cdit_l_eval.yaml` | `0100000` |
| CDiT-XL (预训练) | `nwm_cdit_xl_eval.yaml` | `0100000` |
| CDiT-XL (微调) | `nwm_recon_finetune.yaml` | `best` |
| 4数据集训练 | `4datasets_clip04_CDiTXL_temmeDWR_eval.yaml` | `0001600` |
| 消融实验 | `ab_clip04_*.yaml` | `best` |

### Checkpoint 路径格式

```
{results_dir}/{run_name}/checkpoints/{ckp}.pth.tar
# 例如: logs/nwm_cdit_l/checkpoints/0100000.pth.tar
```

---

## 常见问题

### Q: 模型加载失败，提示 key 不匹配

**A**: 检查：
1. 模型架构是否匹配（CDiT-L vs CDiT-XL）
2. checkpoint 格式是否正确（应包含 "ema" 键）
3. 配置文件中的 `model` 字段是否正确

### Q: 显存不足

**A**: 解决方案：
1. 减小 `batch_size`
2. 减小扩散步数（从 250 步减少到 50-100 步）
3. 使用更少的 `num_samples`

### Q: go_stanford 数据集无法使用 planning_eval.py

**A**: `go_stanford` 没有 `navigation_eval.pkl` 文件，请使用 `isolated_nwm_infer.py` 进行 time/rollout 评估。

### Q: 如何加速评测

**A**: 减少采样数量或使用多卡：
```bash
# 快速评测
--num_samples 60 --opt_steps 1

# 多卡加速
torchrun --nproc-per-node=8 isolated_nwm_infer.py ...
```

### Q: 数据集路径问题

**A**: 检查软链接：
```bash
ls -la data/recon
readlink -f data/recon
```

---

## 参考资源

- 项目 GitHub: https://github.com/facebookresearch/nwm
- 论文: https://arxiv.org/abs/2412.03572
- 预训练模型: https://huggingface.co/facebook/nwm
- NoMaD 数据预处理: https://github.com/robodhruv/visualnav-transformer

---

## 训练配置参考

### 训练数据集规模

| 数据集 | 图像数量 |
|--------|---------|
| recon | 350,699 |
| 4数据集混合 | ~800,000+ |

### 批量大小配置

| 配置 | 单卡 Batch | 梯度累积 | 等效 Batch | GPU 数量 |
|------|-----------|----------|-----------|----------|
| 默认 | 4 | 16 | 256 | 4 |
| 大 Batch | 8 | 32 | 256 | 1 |
| 小 Batch | 4 | 64 | 256 | 1 |

### 每 Epoch 步数计算

```
每 Epoch Step = 图像数量 / 等效 Batch Size
例: 350699 / 256 ≈ 1370 steps/epoch
```
