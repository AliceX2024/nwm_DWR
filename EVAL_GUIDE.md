# 预训练模型评估指南

本指南说明如何使用从 Hugging Face 下载的预训练权重进行推理和评估。

## 1. 上传预训练权重到服务器

### 方法 1: 使用上传脚本（推荐）



在**本地终端**执行：

```bash
cd ~/nwm
bash upload_pretrained.sh
```

脚本会自动：
- 检查本地文件是否存在
- 在服务器上创建必要的目录
- 上传并重命名为标准格式 `0100000.pth.tar`

### 方法 2: 手动上传

```bash
# 在服务器上创建目录
ssh xiaoyj25@cluster44 "mkdir -p /villa/xiaoyj25/nwm/logs/nwm_cdit_l/checkpoints"

# 上传文件
rsync -avzhP /Users/xiaoyj/Desktop/pretrained/cdit_l_100000.pth.tar \
    xiaoyj25@cluster44:/villa/xiaoyj25/nwm/logs/nwm_cdit_l/checkpoints/0100000.pth.tar
```

### 验证上传

```bash
ssh xiaoyj25@cluster44
cd /villa/xiaoyj25/nwm
ls -lh logs/nwm_cdit_l/checkpoints/
# 应该看到 0100000.pth.tar 文件
```

## 2. 权重文件位置说明

根据代码，checkpoint 的加载路径格式为：
```
{results_dir}/{run_name}/checkpoints/{ckp}.pth.tar
```

对于 CDiT-L 模型：
- `results_dir`: `logs` (在配置文件中定义)
- `run_name`: `nwm_cdit_l` (需要与配置文件中的 run_name 匹配)
- `ckp`: `0100000` (对应 100k 步，格式为 7 位数字，前面补零)

因此完整路径为：`logs/nwm_cdit_l/checkpoints/0100000.pth.tar`

cd /villa/xiaoyj25/nwm
conda activate nwm
python test_pretrained.py --ckp 0100000 验证前向传播是否正常。

## 3. 运行推理（单步预测）

### 3.1 准备 Ground Truth 图像（一次性）

```bash
cd /villa/xiaoyj25/nwm
conda activate nwm

设置结果目录
export RESULTS_FOLDER=/villa/xiaoyj25/nwm/eval_results

export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results

python isolated_nwm_infer.py \
    --exp config/nwm_cdit_l_eval.yaml \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1
```
python isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml    --datasets tartan_drive     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1     --rollout_fps_values 1,4

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml    --datasets tartan_drive     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1     --rollout_fps_values 1,4

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml    --datasets go_stanford,recon,tartan_drive,scand,sacson     --batch_size 64     --num_workers 12     --eval_type rollout     --output_dir ${RESULTS_FOLDER}     --gt 1     --rollout_fps_values 1,4

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml    --datasets go_stanford,recon,tartan_drive,scand,sacson     --batch_size 64     --num_workers 12     --eval_type rollout     --output_dir ${RESULTS_FOLDER}     --gt 1     --rollout_fps_values 1,4


 torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_onlymeDWR_eval.yaml    --ckp 0008800     --datasets recon,scand,tartan_drive,sacson,go_stanford     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}

cp /DATA/DATANAS2/xiaoyj25/projects/nwm/my_data_splits/tartan_drive/test/dataset_dist_*.pkl \
   /DATA/DATANAS2/xiaoyj25/projects/nwm/my_data_splits/tartan_drive/test/time.pkl

 在 --gt 1 时不会跑模型，只是把数据集中对应时间点的真值帧按评估时间点落盘，结构就是你看到的 id_0~id_499 目录。

### 3.2 使用预训练模型进行预测

source /DATA/DATANAS2/xiaoyj25/miniconda3/etc/profile.d/conda.sh
conda activate /DATA/DATANAS2/xiaoyj25/envs/nwm3

export HF_ENDPOINT=https://hf-mirror.com   # 可保留，也可不设
export VAE_MODEL_PATH=stabilityai/sd-vae-ft-ema  # 或者指向本地目录
# 可选：强制离线优先
export HF_HUB_OFFLINE=1

# 或如果仍需联网，设置镜像
export HF_ENDPOINT=https://hf-mirror.com# 可选：强制离线优先
export HF_HUB_OFFLINE=1
你的数据集（或 test/time.pkl 索引文件）里有 500 个样本，所以生成了 500 个 id 文件夹，每个文件夹下保存不同时间点（如 1,2,4,8,16 秒）的预测图片
```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_l_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
修改nwm_cdit_xl_eval，加入了新的数据集go_stanford
```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_finetune_eval.yaml \
    --ckp latest \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir results_finetune_latest
```
```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_eval.yaml \
    --ckp latest \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```

```bash
CUDA_VISIBLE_DEVICES=2 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_eval.yaml \
    --ckp latest \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_eval.yaml \
    --ckp latest \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_reweightfalse_eval.yaml \
    --ckp epoch_004_step_0004152 \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```
```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```
```bash
python isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_base_eval.yaml  \
    --ckp 0100000 \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```
```bash
python isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml  \
    --ckp 0001600 \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```
```bash
python isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```


```bash
python isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```
torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval.yaml    --ckp best     --datasets recon,go_stanford     --batch_size 64     --num_workers 4     --eval_type time     --output_dir ${RESULTS_FOLDER}

```bash
torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_eval.yaml     --ckp latest     --datasets recon     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}
```
export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results

python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_reweightfalse_eval.yaml     --ckp epoch_004_step_0004152     --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type rollout    --output_dir ${RESULTS_FOLDER}

python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch4_temporal_eval.yaml     --ckp epoch_004_step_0002076    --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time,rollout    --output_dir ${RESULTS_FOLDER}

python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch4_temporal_eval.yaml     --ckp epoch_001_step_0000519    --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time,rollout    --output_dir ${RESULTS_FOLDER}


torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp epoch_002_step_0002660     --datasets sacson,scand,tartan_drive,go_stanford      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp epoch_001_step_0001330     --datasets sacson,scand,tartan_drive,go_stanford      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp 0001200     --datasets sacson,scand,tartan_drive,go_stanford      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}

下一个测baseline：
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml      --ckp 0100000    --datasets sacson,scand,tartan_drive,go_stanford      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}



rollout模式：每个 id 文件夹下的图片数量等于 rollout 的步数（即预测的未来帧数）gt_image.shape[1] 是轨迹长度（如 64），所以每个 id 文件夹下有 64 张图片，文件名为 0.png, 1.png, ..., 63.png。

你的数据集（或 test/rollout.pkl 索引文件）里有 64 条轨迹，所以生成了 64 个 id 文件夹，每个文件夹下保存该轨迹的所有预测帧。
rollout_fps=1 表示每秒采样 1 帧。如果轨迹总长度是 64 帧，原始帧率是 4FPS（即每秒4帧），则每隔4帧采样1帧，总共采样 64/4 = 16 张图片。所以 rollout_1fps 文件夹下只有 16 张图。
rollout_fps=4 表示每秒采样 4 帧，与原始帧率一致。所以会保留全部 64 张图片（每一帧都保存）。因此，rollout_4fps 文件夹下应该有 64 张图片。
```bash
CUDA_VISIBLE_DEVICES=3 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```
多卡：
```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 \
torchrun --nproc-per-node=4 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```

```bash

torchrun --nproc-per-node=4 isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_eval.yaml  \
    --ckp latest \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```
torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_8epoch_2e6_batch8_lr001_eval.yaml     --ckp epoch_004_step_0002076     --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}

/DATA/DATANAS2/xiaoyj25/projects/nwm/logs/nwm_cdit_xl_finetune_reweight_8epoch_2e6_batch8_lr001_temporal/checkpoints/epoch_004_step_0002076.pth.tar

torchrun --standalone --nproc-per-node=2 train.py     --config config/nwm_recon_2epoch_2e6_batch4_temporal_test.yaml     --ckpt-every 800     --eval-every 10000000     --bfloat16 1     --epochs 8     --torch-compile 0
### 3.3 计算评估指标

记得pip install opencv-python，不要用conda，会产生版本冲突
```bash
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_l_eval \
    --eval_types time
```
```bash
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_eval \
    --eval_types time
```
```bash
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_eval \
    --eval_types time
```

```bash
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_eval_latest \
    --eval_types time
```
```bash
python isolated_nwm_eval.py \
    --datasets recon,go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_reweightfalse_eval_epoch_004_step_0004152 \
    --eval_types time
```
```bash
python isolated_nwm_eval.py \
    --datasets recon,go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch4_temporal_eval_epoch_004_step_0002076 \
    --eval_types time,rollout
```
```bash
python isolated_nwm_eval.py \
    --datasets recon,go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch4_temporal_eval_epoch_004_step_0002076 \
    --eval_types time
```
```bash
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_onlymeDWRTrue_eval_0001600 \
    --eval_types time
```
python isolated_nwm_eval.py     --datasets recon,go_stanford     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch4_temporal_test_eval_epoch_003_step_0001557    --eval_types time

python isolated_nwm_eval.py     --datasets sacson     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_0000800   --eval_types time

python isolated_nwm_eval.py     --datasets recon     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_0001200   --eval_types time

python isolated_nwm_eval.py     --datasets sacson,scand,tartan_drive     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_noDWR_eval --eval_types time
```bash
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_epoch_001_step_0001330 \
    --eval_types time
```

## 4. 调整模型参数

### 4.1 修改预测长度 (len_traj_pred)

编辑 `config/nwm_cdit_l_eval.yaml`:

```yaml
len_traj_pred: 32  # 改为 16, 32, 64 等
```

**注意**: 如果修改了 `len_traj_pred`，需要确保数据集索引文件存在，或者让程序重新生成。

### 4.2 修改历史上下文 (context_size)

编辑 `config/nwm_cdit_l_eval.yaml`:

```yaml
context_size: 2  # 改为 2, 4, 8 等
```

**注意**: 
- `context_size` 影响模型输入的历史帧数
- 预训练模型可能是在特定 `context_size` 下训练的（通常是 4）
- 如果改变 `context_size`，模型可能无法正常工作，因为位置编码维度会改变

### 4.3 修改扩散步数

在 `isolated_nwm_infer.py` 中，扩散步数由 `create_diffusion(str(250))` 控制。可以修改为：

```python
diffusion = create_diffusion(str(50))  # 更少的步数，更快但可能质量下降
```

## 5. 运行 Rollout 评估（轨迹预测）

### 5.1 准备 GT Rollout

```bash 对于xl：
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```

```bash 对于xl：
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```

```bash 对于xl：
python isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml\
    --datasets tartan_drive,scand,sacson \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```
go stanford：
```bash 
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval_go_stanford.yaml \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 4 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1
```
```bash 
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval_go_stanford.yaml \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 4 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp epoch_001_step_0001330     --datasets sacson,scand,tartan_drive,recon      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch32_temporal_2pool_time4_temmemory_lr001_step15_eval.yaml     --ckp epoch_002_step_0001038     --datasets recon,go_stanford     --batch_size 64     --num_workers 4     --eval_type rollout     --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1)  isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml     --ckp 0100000    --datasets recon,sacson,scand,tartan_drive,go_stanford      --batch_size 64     --num_workers 12    --eval_type time    --output_dir ${RESULTS_FOLDER}

(nwm3) xiaoyj25@cluster43:/DATA/DATANAS2/xiaoyj25/projects/nwm$ torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval.yaml    --ckp best     --datasets recon,go_stanford     --batch_size 64     --num_workers 4     --eval_type time     --output_dir ${RESULTS_FOLDER}


torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp 0001200     --datasets scand,sacson,tartan_drive      --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp 0001200     --datasets tartan_drive,recon      --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp 0000800     --datasets go_stanford      --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

### 5.2 模型 Rollout

```bash
CUDA_VISIBLE_DEVICES=3 \
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --rollout_fps_values 1,4
```

```bash
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_eval_go_stanford.yaml \
    --ckp latest \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 4 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
```bash
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_batch4_temporal_4pool_eval.yaml \
    --ckp epoch_008_step_0004152 \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 4 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```




python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch4_temporal_test_eval.yaml    --ckp epoch_003_step_0001557    --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

MASTER_PORT=$(shuf -i 20000-60000 -n 1) python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch4_temporal_test_eval.yaml    --ckp epoch_002_step_0001038    --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

MASTER_PORT=$(shuf -i 20000-60000 -n 1) torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval.yaml    --ckp best     --datasets recon,go_stanford     --batch_size 64     --num_workers 4     --eval_type time     --output_dir ${RESULTS_FOLDER}

MASTER_PORT=$(shuf -i 20000-60000 -n 1)  torchrun --nproc-per-node=1 isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval.yaml    --ckp best     --datasets recon,go_stanford     --batch_size 64     --num_workers 4     --eval_type time     --output_dir ${RESULTS_FOLDER}


python isolated_nwm_infer.py     --exp config/nwm_recon_2epoch_2e6_batch4_temporal_2pool_clip03_lr05_eval.yaml    --ckp 0001400    --datasets recon,go_stanford     --batch_size 64     --num_workers 12     --eval_type time    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval.yaml \
    --ckp best \
    --datasets recon,go_stanford \
    --batch_size 64 \
    --num_workers 12\
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml \
    --ckp 0001600 \
    --datasets recon,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
### 5.3 评估 Rollout

```bash
CUDA_VISIBLE_DEVICES=6 \
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_0001600 \
    --eval_types rollout
```
```bash
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_eval \
    --eval_types time
```

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \                           --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml \                                                             --ckp 0100000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml      --ckp 0100000     --datasets recon     --batch_size 64     --num_workers 12     --eval_type rollout    --rollout_fps_values 1,4    --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml      --ckp 0001600     --datasets recon     --batch_size 64     --num_workers 12     --eval_type rollout    --rollout_fps_values 1,4    --output_dir ${RESULTS_FOLDER}

/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/nwm_recon_2epoch_2e6_batch4_temporal_4pool_eval_epoch_002_step_0001038

python isolated_nwm_eval.py     --datasets recon,go_stanford     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch4_temporal_4pool_eval_epoch_002_step_0001038    --eval_types time

python isolated_nwm_eval.py     --datasets recon,go_stanford     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip04_lr05_eval_best --eval_types time

python isolated_nwm_eval.py     --datasets recon,go_stanford     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch4_temporal_2pool_clip03_lr05_eval_0001400 --eval_types time

python isolated_nwm_eval.py     --datasets recon,go_stanford     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/nwm_recon_2epoch_2e6_batch8_temporal_2pool_clip03_lr05_eval_epoch_002_step_0001038 --eval_types time

python isolated_nwm_eval.py     --datasets sacson,scand,tartan_drive,recon     --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_0000800 --eval_types time
source /DATA/DATANAS2/xiaoyj25/miniconda3/etc/profile.d/conda.sh
conda activate /DATA/DATANAS2/xiaoyj25/envs/nwm3

## 6. 分析 Failure Cases

### 6.1 可视化预测结果

预测的图像会保存在：
```
${RESULTS_FOLDER}/nwm_cdit_l/recon/time/
```

可以：
1. 对比 GT 和预测图像
2. 找出预测质量差的样本
3. 分析失败模式（模糊、颜色偏移、结构错误等）

### 6.2 使用 Jupyter Notebook 交互式分析

```bash
cd /villa/xiaoyj25/nwm
conda activate nwm
jupyter notebook interactive_model.ipynb
```

在 notebook 中：
- 修改 `MODEL_PATH` 指向你的 checkpoint
- 修改 `EXP_NAME` 为 `nwm_cdit_l_eval`
- 可以交互式地输入图像和动作，查看预测结果

### 6.3 批量分析特定样本

创建一个分析脚本：

```python
# analyze_failures.py
import torch
import numpy as np
from PIL import Image
import os

# 加载预测和GT图像
pred_dir = "eval_results/nwm_cdit_l/recon/time"
gt_dir = "eval_results/gt/recon/time"

# 计算每张图像的LPIPS或DreamSim分数
# 找出分数最低的样本（失败案例）
# 可视化这些样本
```

## 7. 性能调优建议

### 7.1 推理速度优化

1. **减少扩散步数**: 从 250 步减少到 50-100 步
2. **使用 torch.compile**: 在推理时启用（但注意兼容性）
3. **减小 batch_size**: 如果显存不足
4. **使用 bfloat16**: 已经在代码中启用

### 7.2 质量优化

1. **增加扩散步数**: 从 250 增加到 500（更慢但质量更好）
2. **调整采样策略**: 修改 `diffusion.p_sample_loop` 的参数
3. **后处理**: 对预测图像进行去噪或增强

### 7.3 参数敏感性分析

创建多个配置文件，测试不同参数组合：

```bash
# 测试不同的 len_traj_pred
for len_traj in 16 32 64; do
    # 修改配置文件
    # 运行推理
    # 收集结果
done
```

## 8. 常见问题

### Q: 模型加载失败，提示 key 不匹配

A: 检查：
1. 模型架构是否匹配（CDiT-L vs CDiT-XL）
2. checkpoint 格式是否正确（应包含 "ema" 键）
3. 配置文件中的 `model` 字段是否正确

### Q: 预测结果质量很差

A: 可能原因：
1. `context_size` 与训练时不匹配
2. 数据预处理方式不同
3. 需要检查 VAE 是否正确加载

### Q: 显存不足

A: 解决方案：
1. 减小 `batch_size`
2. 使用梯度检查点（如果训练）
3. 使用更少的扩散步数

## 9. 下一步

1. **运行基础评估**: 先跑通单步预测，确保模型正常工作
2. **调整参数**: 系统性地测试不同 `len_traj_pred` 和 `context_size`
3. **分析失败案例**: 找出模型在哪些场景下表现不佳
4. **可视化分析**: 使用 notebook 进行交互式探索
5. **性能基准**: 记录不同配置下的指标（LPIPS, DreamSim, FID）


微调：
1. 新建 reweight_utils.py
这个文件实现了 StableNet 中的 Global Memory Bank 思想。它维护一个先进先出（FIFO）的队列，存储历史特征，用于扩大计算样本权重的分母。

2. 修改 train.py
在 train.py 中修改导入模块、初始化 Memory Bank、在 Loss 计算前插入逻辑。

3. 新的训练配置nwm_recon_finetune.yaml

4.训练指令：
torchrun --standalone --nproc-per-node=4 train.py \
    --config config/nwm_recon_finetune.yaml \
    --ckpt-every 1000 \
    --eval-every 10000000 \
    --bfloat16 1 \
    --epochs 1 \
    --torch-compile 0

5.评测
新建一个文件 config/nwm_cdit_xl_finetune_eval.yaml


上传新数据集

1.本地
# 上传 zip 文件到 cluster43
rsync -avzhP "/Users/xiaoyj/Downloads/go_stanford (1).zip" xiaoyj25@cluster43:/DATA/DATANAS2/xiaoyj25/

2. unzip "go_stanford (1).zip"
# 查看解压后的目录

3.创建软连接
cd /DATA/DATANAS2/xiaoyj25/projects/nwm
# 创建 data 目录（如果不存在）
mkdir -p data
# 创建软链接 (根据实际解压的文件夹名调整)
# 假设解压后文件夹名为 go_stanford
ln -s /DATA/DATANAS2/xiaoyj25/go_stanford data/go_stanford_beforeprocess
# 验证软链接
ls -la data/


python analyze_logs.py   --logs "logs/nwm_cdit_xl_finetune_reweight_2epoch_2e6/log.txt" "logs/nwm_cdit_xl_finetune_reweight_4epoch_2e6_temporal_test/log.txt" "logs/nwm_cdit_xl_finetune_reweight_4epoch_2e6_temporal_4pool/log.txt" "logs/nwm_cdit_xl_finetune_reweight_8epoch_2e6_temporal_2pool_clip03_lr05/log.txt" --output ./log_analysis


torchrun --standalone --nproc-per-node=4 train.py     --config config/nwm_recon_2epoch_2e6_batch4_temporal_2pool_clip03_lr05.yaml     --ckpt-every 400     --eval-every 10000000     --bfloat16 1     --epochs 6     --torch-compile 0




torchrun --standalone --nproc-per-node=4 train.py     --config config/nwm_recon_2epoch_2e6_batch4_temporal_2pool_clip03_lr05.yaml  --ckpt-every 200     --eval-every 10000000     --bfloat16 1     --epochs 6     --torch-compile 0

python analyze_logs.py   --logs "logs/nwm_cdit_xl_finetune_reweight_2epoch_2e6/log.txt" "logs/nwm_cdit_xl_finetune_reweight_4epoch_2e6_temporal_test/log.txt" "logs/nwm_cdit_xl_finetune_reweight_8epoch_2e6_temporal_2pool_clip03_lr05/log.txt" "logs/nwm_cdit_xl_finetune_reweight_8epoch_2e6_temporal_2pool_clip04_lr05/log.txt" --output ./log_analysis/clip

python analyze_logs.py   --logs "logs/nwm_cdit_xl_finetune_reweight_2epoch_2e6/log.txt" "logs/nwm_recon_2epoch_2e6_batch32_temporal_2pool_time4_temmemory/log.txt" "logs/nwm_recon_2epoch_2e6_batch32_temporal_2pool_time4_temmemory_lr001_step15/log.txt" "logs/nwm_recon_2epoch_2e6_batch32_temporal_4pool_time4_temmemory_lr001_step15/log.txt" --output ./log_analysis/temmemory

Export CUDA_VISIBLE_DEVICES=1,2
torchrun --standalone --nproc-per-node=2 train.py     --config config/ab_clip04_noDWR.yaml     --ckpt-every 200     --eval-every 10000000     --bfloat16 1     --epochs 4     --torch-compile 0


torchrun --standalone --nproc-per-node=2 train.py     --config config/ab_clip04_temDWR.yaml     --ckpt-every 200     --eval-every 10000000     --bfloat16 1     --epochs 4     --torch-compile 0

## 10. Trajectory Evaluation - Planning（轨迹规划评估）

### 10.1 概述

Trajectory Evaluation - Planning 使用 **1步 Cross Entropy Method (CEM)** 规划算法评估模型的轨迹预测能力。该方法通过采样多个候选动作序列，选择 LPIPS 损失最低的 top-k 样本更新动作分布，迭代优化找到最优轨迹。

### 10.2 支持的数据集

Planning 评估需要 `navigation_eval.pkl` 文件。

#### 针对 recon 和 go_stanford

| 数据集 | Planning 支持 | 说明 |
|--------|--------------|------|
| `recon` | ✅ 支持 | 有 `navigation_eval.pkl` |
| `go_stanford` | ❌ 不支持 | 没有 `navigation_eval.pkl`，只有 `time.pkl` 和 `rollout.pkl` |

#### 所有支持的数据集

- `recon`
- `tartan_drive`
- `scand`
- `sacson`

**注意**：`go_stanford` 数据集只有 `time.pkl` 和 `rollout.pkl`，不支持 planning 评估。如需评测 go_stanford，请使用 `isolated_nwm_infer.py` 进行 time/rollout 评估。

### 10.3 核心参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp` | 配置文件路径 | - |
| `--datasets` | 数据集名称（逗号分隔） | - |
| `--ckp` | Checkpoint 名称或 `best` | `0100000` |
| `--num_samples` | CEM 采样的轨迹数量 | 10 |
| `--topk` | 选择 top-k 个最优轨迹更新分布 | 5 |
| `--opt_steps` | CEM 优化迭代次数 | 15 |
| `--rollout_stride` | Rollout 步长 | 1 |
| `--num_repeat_eval` | 每个动作序列的重复评估次数（降低方差） | 1 |
| `--batch_size` | 批处理大小 | 16 |
| `--num_workers` | DataLoader worker 数量 | 8 |
| `--output_dir` | 结果输出目录 | - |
| `--save_preds` | 是否保存预测结果 | False |
| `--plot` | 是否生成可视化图表 | False |

### 10.4 调用示例

> **说明**：以下示例针对 `recon` 和 `go_stanford` 数据集。注意 `go_stanford` 不支持 planning 评估。

#### 10.4.1 基础调用（单步 CEM，120 轨迹）- Recon 数据集

```bash
# 设置结果目录
export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results

# 使用 8 卡运行，采样 120 条轨迹，topk=5，1 步优化
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

#### 10.4.2 评测 ab_clip04_noDWR 模型

```bash
# 模型配置：ab_clip04_noDWR.yaml
# Checkpoint: ab_clip04_noDWR/checkpoints/best

export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results

# Recon 数据集 - Planning 评估
torchrun --nproc-per-node=1 planning_eval.py \
    --exp config/ab_clip04_noDWR.yaml \
    --datasets recon \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 120 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp best \
    --opt_steps 1 \
    --num_repeat_eval 3

# go_stanford 数据集 - Planning 不支持，使用 Rollout 评估
# 注意：go_stanford 没有 navigation_eval.pkl，需使用 isolated_nwm_infer.py
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/ab_clip04_noDWR_eval.yaml \
    --ckp best \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```

#### 10.4.3 评测 nwm_recon_2epoch_2e6 模型（带 Reweight）

```bash
# 模型配置：nwm_recon_2epoch_2e6.yaml
# 注意：该配置的 run_name 为 nwm_cdit_xl_finetune_reweight_2epoch_2e6
# Checkpoint: nwm_cdit_xl_finetune_reweight_2epoch_2e6/checkpoints/best

export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results

# Recon 数据集 - Planning 评估
torchrun --nproc-per-node=8 planning_eval.py \
    --exp config/nwm_recon_2epoch_2e6.yaml \
    --datasets recon \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 120 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp best \
    --opt_steps 1 \
    --num_repeat_eval 3


torchrun --nproc-per-node=1 planning_eval.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml \
    --datasets recon,tartan_drive,sacson,scand \
    --rollout_stride 1 \
    --batch_size 1 \
    --num_samples 30 \
    --topk 5 \
    --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} \
    --save_preds \
    --ckp 0000800 \
    --opt_steps 1 \
    --num_repeat_eval 1

一个一个数据集放：
torchrun --nproc-per-node=1 planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets recon     --rollout_stride 1     --batch_size 1     --num_samples 30     --topk 5     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0000800     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets go_stanford     --rollout_stride 1     --batch_size 1     --num_samples 30     --topk 5     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0000800     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets scand     --rollout_stride 1     --batch_size 1     --num_samples 30     --topk 5     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0000800     --opt_steps 1     --num_repeat_eval 1

提速：
torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets sacson     --rollout_stride 4     --batch_size 8     --num_samples 10     --topk 3     --num_
workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0000800
     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWRTrue_eval.yaml     --ckp 0001600     --datasets recon,scand,tartan_drive,sacson,go_stanford     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}

终极提速：
torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets sacson     --rollout_stride 8     --batch_size 16     --num_samples 10     --topk 3     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0001600     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets sacson     --rollout_stride 8     --batch_size 16     --num_samples 10     --topk 3     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0001600     --opt_steps 1     --num_repeat_eval 1

export RESULTS_FOLDER=/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results
torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml     --datasets tartan_drive     --rollout_stride 8     --batch_size 16     --num_samples 10     --topk 3     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0001600     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_base_eval.yaml    --datasets scand    --rollout_stride 8     --batch_size 16     --num_samples 10     --topk 3     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0100000     --opt_steps 1     --num_repeat_eval 1

torchrun --nproc-per-node=1  --master_port=$(shuf -i 20000-60000 -n 1) planning_eval.py     --exp config/4datasets_clip04_CDiTXL_traDWR_eval.yaml   --datasets recon  --rollout_stride 8     --batch_size 16     --num_samples 10     --topk 3     --num_workers 12     --output_dir ${RESULTS_FOLDER}     --save_preds     --ckp 0001600     --opt_steps 1     --num_repeat_eval 1

python isolated_nwm_eval.py     --datasets go_stanford,recon,tartan_drive,scand,sacson    --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_traDWR_eval_0001600  --eval_types time

python isolated_nwm_eval.py     --datasets tartan_drive,scand,sacson    --gt_dir ${RESULTS_FOLDER}/gt     --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_eval_0001600  --eval_types time

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTL_onlymeDWR_eval.yaml     --ckp 0001600     --datasets recon,go_stanford,scand,tartan_drive,sacson     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_noDWR_eval.yaml    --ckp 0001600     --datasets recon,go_stanford,scand,tartan_drive,sacson     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}


# go_stanford 数据集 - Rollout 评估
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/nwm_recon_2epoch_2e6_eval.yaml \
    --ckp best \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```

### 10.5 评测结果解读

#### 10.5.1 输出文件结构

```
${RESULTS_FOLDER}/<exp_name>/<dataset_name>/CEM_N{num_samples}_K{topk}_RS{rollout_stride}_rep{num_repeat_eval}_OPT{opt_steps}/
```

其中包含：
- `plots/`: 可视化图片目录（当 `--plot` 启用时）
- 预测和 GT 数据

#### 10.5.2 评估指标

运行完成后会生成 `${dataset_name}_CEM_*.json` 文件，包含以下指标：

| 指标 | 说明 |
|------|------|
| `ate` | Absolute Trajectory Error（绝对轨迹误差） |
| `rpe_trans` | Relative Pose Error - Translation（相对位姿误差 - 平移） |
| `pos_diff_norm` | 终点位置与目标位置的欧氏距离 |
| `yaw_diff_norm` | 终点偏航角与目标偏航角的差值 |

#### 10.5.3 查看评测结果

```bash
# 查看评测指标
cat ${RESULTS_FOLDER}/tartan_drive_CEM_N120_K5_RS1_rep3_OPT1.json

# 可视化预测轨迹（需要 --plot 参数）
ls ${RESULTS_FOLDER}/<exp_name>/tartan_drive/CEM_*/plots/
```

### 10.6 常见问题

#### Q: go_stanford 数据集无法使用 planning_eval.py？

A: `go_stanford` 数据集没有 `navigation_eval.pkl` 文件，不支持 planning 评估。请使用 `isolated_nwm_infer.py` 进行 time/rollout 评估：

```bash
# go_stanford 只能使用 rollout 评估
torchrun --nproc-per-node=1 isolated_nwm_infer.py \
    --exp config/ab_clip04_noDWR_eval.yaml \
    --ckp best \
    --datasets go_stanford \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}
```

#### Q: 显存不足怎么办？

A: 减小 `num_samples` 或 `num_repeat_eval`，或减小 batch_size：

```bash
# 减小采样数量
--num_samples 60 --num_repeat_eval 2

# 或减小 batch_size
--batch_size 1
```

#### Q: 如何加速评测？

A: 减少采样数量或优化步数：

```bash
# 快速评测（牺牲精度）
--num_samples 60 --opt_steps 1
```

#### Q: checkpoint 路径问题

A: 确保配置文件中的 `results_dir` 和 `run_name` 与实际路径一致：

```yaml
# ab_clip04_noDWR.yaml
results_dir: logs
run_name: ab_clip04_noDWR

# 完整路径: logs/ab_clip04_noDWR/checkpoints/{ckp}.pth.tar
```
torchrun --standalone --nproc-per-node=2 train.py --config config/4datasets_clip04_CDiTXL_temmeDWR_ts2.yaml --ckpt-every 200 --eval-every 10000000 --bfloat16 1 --epochs 40 --torch-compile 0 

torchrun --standalone --nproc-per-node=2 train.py --config config/4datasets_clip04_CDiTXL_temmeDWR_clip02.yaml --ckpt-every 200 --eval-every 10000000 --bfloat16 1 --epochs 40 --torch-compile 0 

0408
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval.yaml     --ckp 0001600     --datasets go_stanford,sacson,scand,tartan_drive,recon     --batch_size 64     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py     --exp config/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval.yaml  --datasets go_stanford,recon,tartan_drive,scand,sacson     --batch_size 64     --num_workers 12     --eval_type rollout     --output_dir ${RESULTS_FOLDER}         --rollout_fps_values 1,4

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval.yaml  \
    --ckp 0001600 \
    --datasets go_stanford,sacson,scand,tartan_drive,recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval.yaml  \
    --ckp 0000800 \
    --datasets go_stanford\
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_clip02_eval.yaml  \
    --ckp 0001600 \
    --datasets go_stanford,recon\
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

0409
```bash
CUDA_VISIBLE_DEVICES=6 \
python isolated_nwm_eval.py \
    --datasets go_stanford \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval_0002400 \
    --eval_types rollout
```



```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_clip02_eval.yaml  \
    --ckp 0004000 \
    --datasets go_stanford,recon\
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_clip02_eval.yaml  \
    --ckp 0003200 \
    --datasets go_stanford,recon\
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}



    0525
```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/nwm_recon_scratch_2gpu_batch32_eval.yaml  \
    --ckp 0011200 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/nwm_recon_scratch_2gpu_batch32_eval.yaml  \
    --ckp 0011200 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```

0529
```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/nwm_recon_scratch_2gpu_eval.yaml  \
    --ckp 0020400 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/nwm_recon_scratch_2gpu_eval.yaml  \
    --ckp 0020400 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```


0531
```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank1024_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```
```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank4096_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```
0531晚

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank1024_eval.yaml   \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank4096_eval.yaml   \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval.yaml   \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout\
    --rollout_fps_values 1,4\
    --output_dir ${RESULTS_FOLDER}
```

```bash
torchrun --nproc-per-node=1 --master_port=$(shuf -i 20000-60000 -n 1) isolated_nwm_infer.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval.yaml  \
    --ckp 0001600 \
    --datasets recon,go_stanford,sacson,scand,tartan_drive \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time\
    --output_dir ${RESULTS_FOLDER}
```

0601
```bash
python isolated_nwm_eval.py \
    --datasets recon,go_stanford,scand,sacson,tartan_drive \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval_0001600 \
    --eval_types time
```

python multi_run_eval.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_eval.yaml \
    --ckp 0001600 \
    --datasets go_stanford \
    --eval_type time \
    --gt_dir eval_results/gt \
    --base_output_dir eval_results/multi_run_0001600 \
    --n_runs 5 \
    --base_seed 0 \
    --ngpus 1 \
    --batch_size 64 \
    --num_workers 12

for i in 0 1 2 3 4; do
  python isolated_nwm_eval.py \
    --gt_dir eval_results/gt \
    --exp_dir eval_results/multi_run_0001600/run_00$i/4datasets_clip04_CDiTXL_temmeDWR_bank512_eval_0001600 \
    --datasets go_stanford \
    --eval_types time
done

python multi_run_eval.py \
    --exp config/4datasets_clip04_CDiTXL_noDWRTrue_eval.yaml \
    --ckp 0001600 \
    --datasets go_stanford \
    --eval_type time \
    --gt_dir eval_results/gt \
    --base_output_dir eval_results/multi_run_0001600 \
    --n_runs 5 \
    --base_seed 0 \
    --ngpus 1 \
    --batch_size 64 \
    --num_workers 12



python multi_run_eval.py \
    --exp config/4datasets_clip04_CDiTXL_temmeDWR_bank1024_eval.yaml \
    --ckp 0001600 \
    --datasets go_stanford,recon,sacson,scand,tartan_drive \
    --eval_type time \
    --gt_dir eval_results/gt \
    --base_output_dir eval_results/multi_run_4datasets_clip04_CDiTXL_temmeDWR_bank1024_eval_0001600 \
    --n_runs 5 \
    --base_seed 0 \
    --ngpus 1 \
    --batch_size 64 \
    --num_workers 12