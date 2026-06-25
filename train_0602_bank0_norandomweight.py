# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------





# === 导入我们编写的重加权工具 ===
from reweight_utils import GlobalFeatureBank, compute_dwr_weights_online, apply_group_weighting, compute_dwr_weights_spacetime

from isolated_nwm_infer import model_forward_wrapper

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import glob
import re

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟，根据需要调整
import matplotlib.pyplot as plt 
import yaml


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

from distributed import init_distributed
from models import CDiT_models
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def _ddp_gather_scalars(t: torch.Tensor):
    """
    Gather a 1D tensor of scalars (on CUDA) from all ranks.
    Returns list[torch.Tensor] with length world_size.
    """
    world_size = dist.get_world_size()
    out = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(out, t)
    return out


def _ddp_assert_all_ranks_equal(name: str, t: torch.Tensor, atol: float = 0.0, rtol: float = 0.0):
    """
    Assert a scalar tensor is equal (within tol) across ranks.
    """
    gathered = _ddp_gather_scalars(t.detach().flatten().to(dtype=torch.float32))
    if dist.get_rank() == 0:
        vals = torch.stack([g.cpu() for g in gathered], dim=0)
        ref = vals[0]
        max_diff = (vals - ref).abs().max().item()
        if max_diff > (atol + rtol * ref.abs().max().item()):
            raise RuntimeError(f"DDP desync detected for {name}: gathered={vals.squeeze(-1).tolist()}")


def _model_fingerprint(model: torch.nn.Module, device: torch.device, n_keys: int = 6, n_elems: int = 1024):
    """
    Lightweight deterministic fingerprint of model parameters.
    Uses first `n_elems` values from a few state_dict keys.
    Returns tensor([mean, std, l2, max_abs]) on CUDA.
    """
    sd = model.state_dict()
    keys = sorted(sd.keys())[:n_keys]
    chunks = []
    for k in keys:
        v = sd[k]
        if not torch.is_tensor(v):
            continue
        flat = v.detach().to(device=device, dtype=torch.float32).flatten()
        if flat.numel() == 0:
            continue
        chunks.append(flat[: min(n_elems, flat.numel())])
    if not chunks:
        return torch.zeros(4, device=device, dtype=torch.float32)
    x = torch.cat(chunks, dim=0)
    mean = x.mean()
    std = x.std(unbiased=False)
    l2 = torch.sqrt((x * x).mean())
    max_abs = x.abs().max()
    return torch.stack([mean, std, l2, max_abs], dim=0)


@torch.no_grad()
#EMA模型（Exponential Moving Average Model）是训练过程中维护的模型副本，其参数是当前训练模型参数的滑动平均。滑动平均是一种加权平均，最近的参数权重更高，公式为：
#ema_param = decay * ema_param + (1 - decay) * current_param
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP: 分布式训练初始化
    _, rank, device, _ = init_distributed()
    
    # 设置随机种子，保证实验可复现
    # seed = args.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # 注意：所有 rank 使用相同的种子进行模型初始化，以确保 DDP 同步
    master_seed = args.global_seed
    torch.manual_seed(master_seed)
    torch.cuda.manual_seed(master_seed)
    print(f"Starting rank={rank}, master_seed={master_seed}, world_size={dist.get_world_size()}.")
    
    # === 1. 配置加载 (Config Loading) ===
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # === [新增] 读取 Reweighting 实验配置 ===
    # 如果 config 中没有 reweight 字段，默认关闭，防止报错
    rw_cfg = config.get('reweight', {})
    enable_reweight = rw_cfg.get('enable', False)
    
    # Setup folders: 创建实验目录
    os.makedirs(config['results_dir'], exist_ok=True) 
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  
    checkpoint_dir = f"{experiment_dir}/checkpoints"  
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # [Debug] 打印当前实验配置状态
        if enable_reweight:
            logger.info(f"🚀 Reweighting Module: ENABLED")
            logger.info(f"   Config: {rw_cfg}")
        else:
            logger.info("⚠️ Reweighting Module: DISABLED")
    else:
        logger = create_logger(None) # 非 rank0 进程创建空 logger

    # IMPORTANT: ensure filesystem side-effects (dirs/log file) are visible to all ranks
    # before any rank tries to read checkpoints from `checkpoint_dir`.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Create model: 加载 VAE 和 Tokenizer
    vae_model_name = os.getenv("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-ema")
    try:
        tokenizer = AutoencoderKL.from_pretrained(vae_model_name, local_files_only=True).to(device)
    except Exception as e:
        if "local_files_only" in str(e) or "not found" in str(e).lower():
            print(f"Local model not found, trying to download from Hugging Face...")
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                tokenizer = AutoencoderKL.from_pretrained(vae_model_name).to(device)
            except Exception as e2:
                print(f"Failed to download model: {e2}")
                raise
        else:
            raise
            
    latent_size = config['image_size'] // 8
    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']

    # 初始化 CDiT 模型
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    model.use_checkpoint = True   

  
    # 初始化 EMA 模型 (测试时用)
    ema = deepcopy(model).to(device) 
    requires_grad(ema, False)
    
    # 优化器设置
    lr = float(config.get('lr', 1e-4))
    # [建议] weight_decay 设为 0.01 可缓解过拟合
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0))

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # 加载预训练权重 / 断点续训
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    # NOTE: checkpoint selection MUST be identical across ranks; otherwise barriers/logging
    # conditions can diverge and deadlock (NCCL allreduce timeout).
    resume_path = ""
    if rank == 0:
        has_latest = os.path.isfile(latest_path)
        has_user_ckpt = bool(config.get('from_checkpoint', 0))
        prefer_from_ckpt = bool(config.get("prefer_from_checkpoint", False))

        # 自动查找最新的 epoch checkpoint（优先级最高）
        def find_latest_epoch_checkpoint():
            epoch_ckpts = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth.tar"))
            if not epoch_ckpts:
                return None
            # 按文件名中的 step 数字排序，取最大的
            epoch_ckpts.sort(key=lambda x: int(re.search(r'step_(\d+)\.pth\.tar$', x).group(1)), reverse=True)
            return epoch_ckpts[0]

        latest_epoch_path = find_latest_epoch_checkpoint()
        if latest_epoch_path:
            print(f"Found latest epoch checkpoint: {latest_epoch_path}")

        # 优先级：from_checkpoint (prefer_from_checkpoint=true) > latest epoch > latest > user ckpt without preference
        if prefer_from_ckpt and has_user_ckpt:
            resume_path = str(config.get('from_checkpoint', 0))
        elif latest_epoch_path:
            # 优先使用最新的 epoch checkpoint（更完整，包含 epoch 信息）
            resume_path = latest_epoch_path
        elif has_latest and has_user_ckpt:
            if rank == 0:
                try:
                    logger.warning(
                        "Both `latest.pth.tar` exists and `from_checkpoint` is set in config. "
                        "Defaulting to resume from latest. "
                        "Set `prefer_from_checkpoint: true` in config to override."
                    )
                except Exception:
                    pass
            resume_path = latest_path
        elif has_user_ckpt:
            resume_path = str(config.get('from_checkpoint', 0))
        elif has_latest:
            resume_path = latest_path

    if dist.is_available() and dist.is_initialized():
        obj_list = [resume_path]
        dist.broadcast_object_list(obj_list, src=0)
        resume_path = obj_list[0]

    if resume_path:
        print("Loading model from ", resume_path)
        # Guard: make sure all ranks can see the same checkpoint path
        if dist.is_available() and dist.is_initialized():
            exists = 1 if os.path.isfile(resume_path) else 0
            exists_t = torch.tensor([exists], device=device, dtype=torch.int32)
            dist.all_reduce(exists_t, op=dist.ReduceOp.SUM)
            if int(exists_t.item()) != dist.get_world_size():
                raise FileNotFoundError(
                    f"Checkpoint path not visible on all ranks: {resume_path}"
                )

        # [DEBUG] Load checkpoint identically on all ranks: load on rank0, then broadcast to others.
        # This avoids subtle differences when map_location maps to different GPUs.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if rank == 0:
                print("Loading checkpoint (broadcast to all ranks)...")
                latest_checkpoint_cpu = torch.load(resume_path, map_location='cpu', weights_only=False)
            else:
                latest_checkpoint_cpu = None
            import pickle
            from torch.distributed import broadcast_object_list
            # Serialize and broadcast
            if rank == 0:
                data = pickle.dumps(latest_checkpoint_cpu)
            else:
                data = None
            data_list = [data]
            broadcast_object_list(data_list, src=0)
            latest_checkpoint = pickle.loads(data_list[0])
            print(f"[rank{rank}] Loaded checkpoint, keys: {list(latest_checkpoint.keys())}")
        else:
            latest_checkpoint = torch.load(resume_path, map_location=f'cuda:{device}', weights_only=False)

        # Handle checkpoint loading: could have both 'model' and 'ema', or only 'ema'.
        # Fine-tuning from a pretrained checkpoint often only saves 'ema'.
        if "model" in latest_checkpoint:
            # Standard case: both 'model' and 'ema' are present
            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        elif "ema" in latest_checkpoint:
            # Fine-tuning case: only 'ema' exists (pretrained checkpoint).
            # Use EMA weights to initialize both model and EMA.
            ema_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = model.load_state_dict(ema_ckp, strict=True)
            print("Loading model from EMA (fine-tune start)", res)
            res = ema.load_state_dict(ema_ckp, strict=True)
            print("Loading EMA from EMA (fine-tune start)", res)
        else:
            # No checkpoint data, start from scratch
            update_ema(ema, model, decay=0)

        # 区分：断点续训 vs 微调
        # 判断标准：resume_path 是否在当前实验目录下
        # - 在当前目录 → 断点续训，加载 train_steps/epoch/opt
        # - 外部路径 → 微调，重置为 0
        is_resume = (resume_path and 
                     os.path.isfile(resume_path) and 
                     checkpoint_dir in os.path.abspath(resume_path))
        
        if is_resume:
            # 断点续训：加载优化器状态和训练进度
            print(f"[Resume] Detected resume from training checkpoint: {resume_path}")
            if "opt" in latest_checkpoint:
                opt.load_state_dict(latest_checkpoint["opt"])
            if "train_steps" in latest_checkpoint:
                train_steps = latest_checkpoint["train_steps"]
            if "epoch" in latest_checkpoint:
                start_epoch = latest_checkpoint["epoch"]
        else:
            # 微调：从 0 开始，不加载训练进度
            print(f"[Fine-tune] Detected fine-tune from pretrained model: {resume_path}")
            train_steps = 0
            start_epoch = 0

        if "scaler" in latest_checkpoint and bfloat_enable:
            scaler.load_state_dict(latest_checkpoint["scaler"])

    # -------------------- 同步 epoch 和 train_steps 到所有 rank --------------------
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # 所有 rank 都要调用 broadcast
        epoch_tensor = torch.tensor([float(start_epoch)], device=device)
        train_steps_tensor = torch.tensor([float(train_steps)], device=device)
        dist.broadcast(epoch_tensor, src=0)
        dist.broadcast(train_steps_tensor, src=0)
        start_epoch = int(epoch_tensor.item())
        train_steps = int(train_steps_tensor.item())
        if rank == 0:
            print(f"[Sync] Broadcasting start_epoch={start_epoch}, train_steps={train_steps} to all ranks")
    else:
        # 单 GPU 或非分布式模式下，直接使用加载的值
        pass

    # -------------------- Step-0 DDP alignment checks --------------------
    # These checks catch "some ranks didn't load the same checkpoint" early.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # train_steps must be identical across ranks (controls barriers/logging/ckpt schedule).
        _ddp_assert_all_ranks_equal("train_steps", torch.tensor([float(train_steps)], device=device))

        # Model / EMA fingerprints should match across ranks.
        fp_model = _model_fingerprint(model, device=device)
        fp_ema = _model_fingerprint(ema, device=device)
        for i, n in enumerate(["mean", "std", "l2", "max_abs"]):
            _ddp_assert_all_ranks_equal(f"model_fp[{n}]", fp_model[i : i + 1], atol=1e-6, rtol=1e-6)
            _ddp_assert_all_ranks_equal(f"ema_fp[{n}]", fp_ema[i : i + 1], atol=1e-6, rtol=1e-6)

        dist.barrier()
        
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device])
    
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # === [关键新增] 初始化全局特征记忆库 ===
    # mode='random'：使用随机替换策略，避免特征只包含最近几分钟的数据，防止灾难性遗忘
    # feature_dim:
    #   pool_type='avg': base_dim=4, 时空维度=9
    #   pool_type='adaptive_2x2': base_dim=16, 时空维度=33
    #   pool_type='adaptive_4x4': base_dim=64, 时空维度=129
    #   pool_type='raw': base_dim=3136, 时空维度=6273
    pool_type = rw_cfg.get('pool_type', 'avg')
    if pool_type == 'raw':
        base_dim = 3136
    elif pool_type == 'adaptive_4x4':
        base_dim = 64  # 4 * 4 * 4
    elif pool_type == 'adaptive_2x2':
        base_dim = 16  # 4 * 2 * 2
    else:  # 'avg'
        base_dim = 4
    
    if rw_cfg.get('use_spacetime', False):
        # 时空版本: 静态(base_dim) + 残差(base_dim) + 时间(1) = base_dim*2 + 1
        feature_dim = base_dim * 2 + 1
    else:
        feature_dim = base_dim
    reweight_bank = GlobalFeatureBank(
        feature_dim=feature_dim, 
        bank_size=rw_cfg.get('bank_size', 2048), 
        device=device, 
        mode=rw_cfg.get('memory_mode', 'random')
    )
    use_reweight_bank = getattr(reweight_bank, 'use_bank', True)
    
    if use_reweight_bank:
        # 如果是从 checkpoint 恢复，恢复 reweight_bank 状态（确保各 rank 一致）
        # 优先从单独的 reweight_bank checkpoint 恢复
        reweight_ckpt_path = f"{checkpoint_dir}/reweight_bank_rank{rank}.pt"
        if os.path.isfile(reweight_ckpt_path):
            reweight_state = torch.load(reweight_ckpt_path, map_location=f'cuda:{device}')
            reweight_bank.load_state_dict(reweight_state)
            print(f"[rank{rank}] Restored reweight_bank from {reweight_ckpt_path}: is_full={reweight_bank.is_full}, ptr={reweight_bank.ptr}")
        elif resume_path and "reweight_bank" in latest_checkpoint:
            # 兼容旧格式：从主 checkpoint 恢复
            reweight_bank.load_state_dict(latest_checkpoint["reweight_bank"])
            if rank == 0:
                print(f"Restored reweight_bank from main checkpoint: is_full={reweight_bank.is_full}, ptr={reweight_bank.ptr}")
        
        # === [防死锁] 验证各 rank 的 reweight_bank 状态一致性 ===
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # 验证 is_full 和 ptr 是否一致
            is_full_tensor = torch.tensor([float(reweight_bank.is_full)], device=device)
            ptr_tensor = torch.tensor([float(reweight_bank.ptr)], device=device)
            _ddp_assert_all_ranks_equal("reweight_bank.is_full", is_full_tensor)
            _ddp_assert_all_ranks_equal("reweight_bank.ptr", ptr_tensor)
            dist.barrier()
        # ================================
    
    # Dataset Preparation (保持原有逻辑)
    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]
                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)
    
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=False  # 禁用以避免 worker 状态不一致
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Step-0 input sanity checks (shapes + action-condition stats).
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        sampler.set_epoch(start_epoch)
        x0, y0, rel_t0 = next(iter(loader))
        # Shapes must match across ranks.
        shape_t = torch.tensor(
            [x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4],
             y0.shape[0], y0.shape[1], y0.shape[2],
             rel_t0.shape[0], rel_t0.shape[1]],
            device=device,
            dtype=torch.int32,
        )
        gathered_shapes = _ddp_gather_scalars(shape_t.to(torch.float32))
        if rank == 0:
            shapes = torch.stack([g.to(torch.int32).cpu() for g in gathered_shapes], dim=0)
            if not torch.all(shapes == shapes[0]):
                raise RuntimeError(f"DDP input shape mismatch across ranks: {shapes.tolist()}")

        # Action condition (y) and rel_t statistics should be broadly similar.
        y_stats = torch.tensor(
            [y0.float().mean(), y0.float().std(unbiased=False), y0.float().abs().max()],
            device=device,
            dtype=torch.float32,
        )
        rel_stats = torch.tensor(
            [rel_t0.float().mean(), rel_t0.float().std(unbiased=False), rel_t0.float().abs().max()],
            device=device,
            dtype=torch.float32,
        )
        for i, n in enumerate(["mean", "std", "max_abs"]):
            # Not asserting tight equality (different samples per rank); just ensure finite.
            if torch.isnan(y_stats).any() or torch.isinf(y_stats).any():
                raise RuntimeError("Found NaN/Inf in y (action condition) at step0.")
            if torch.isnan(rel_stats).any() or torch.isinf(rel_stats).any():
                raise RuntimeError("Found NaN/Inf in rel_t at step0.")

        dist.barrier()

    model.train() 
    ema.eval() 

    log_steps = 0
    running_loss = 0
    start_time = time()
    
    # === [新增] 追踪最低 loss ===
    best_loss = float('inf')
    
    # === [关键新增] 梯度累积步数设置 (防崩溃核心) ===
    # 单卡 Batch=4，4卡 Batch=16。对于 Diffusion 这太小了，容易梯度震荡。
    # 如果累积 16 次，等效 Batch = 16 * 16 = 256，可以稳定训练。
    accum_iter = config.get('gradient_accumulation_steps', 4) 
    logger.info(f"Gradient Accumulation Steps: {accum_iter} (Effective Batch: {config['batch_size'] * dist.get_world_size() * accum_iter})")
    
    logger.info(f"Training for {args.epochs} epochs...")
    
    # ======================= Training Loop =======================
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        # enumerate 这里的 data_iter_step 用于判断何时该 step 优化器
        for data_iter_step, (x, y, rel_t) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            
            # VAE Encoding: 提取潜空间特征
            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    B, T = x.shape[:2]
                    x = x.flatten(0,1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                    x = x.unflatten(0, (B, T))
                
                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1) # (N, 4, 28, 28)
                
                x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)

                # =========================================================
                #              Reweighting Logic (核心实验模块)
                # =========================================================
                batch_weights = torch.ones(x_start.shape[0], device=device) # 默认全1权重
                
                # 1. Warmup 判断: 前几百步不加权，防止初始不稳定特征污染记忆库
                is_warmup = train_steps <= rw_cfg.get('start_steps', 200)
                
                # 获取时间跨度 rel_t (用于时空联合特征)
                # rel_t 已经在前面 flatten 了，shape: [B * num_goals]
                
                # [新增] 选择池化方式: 'avg' (全局平均池化 4维), 'adaptive_2x2' (2x2自适应池化 16维), 'raw' (原始 3136维)
                pool_type = rw_cfg.get('pool_type', 'avg')
                
                # 定义池化函数
                def extract_features(x, pool_type):
                    """
                    提取特征:
                    - 'avg': 全局平均池化 -> [B, 4]
                    - 'adaptive_2x2': 2x2 自适应池化 -> [B, 4*2*2 = 16]
                    - 'adaptive_4x4': 4x4 自适应池化 -> [B, 4*4*4 = 64]
                    - 'raw': 原始特征 -> [B, 4*28*28 = 3136]
                    """
                    if pool_type == 'raw':
                        return x.flatten(start_dim=1)  # [B, 3136]
                    elif pool_type == 'adaptive_4x4':
                        # 4x4 自适应池化: 输出固定为 4x4
                        # x shape: [B, C, H, W] = [B, 4, 28, 28]
                        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).flatten(start_dim=1)  # [B, 4*4*4 = 64]
                    elif pool_type == 'adaptive_2x2':
                        # 2x2 自适应池化: 输出固定为 2x2
                        # x shape: [B, C, H, W] = [B, 4, 28, 28]
                        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(start_dim=1)  # [B, 4*2*2 = 16]
                    else:  # 'avg' 默认
                        return torch.mean(x, dim=[2, 3])  # [B, 4]
                
                if enable_reweight and not is_warmup:
                    # 特征提取
                    curr_features = extract_features(x_start, pool_type)  # [B*goals, base_dim]
                    
                    # === [新增] 时空联合特征 DWR ===
                    if rw_cfg.get('use_spacetime', False):
                        # 提取观测特征: 历史帧最后时刻的特征
                        # x_cond shape: [B*num_goals, num_cond, 4, 28, 28]
                        obs_features = extract_features(x_cond[:, -1, :, :, :], pool_type)  # [B*goals, base_dim]
                        
                        # 调用时空联合权重函数
                        # 传入 logger 用于调试输出
                        batch_weights = compute_dwr_weights_spacetime(
                            curr_features=curr_features,
                            obs_features=obs_features,
                            rel_t=rel_t,
                            reweight_bank=reweight_bank,
                            current_batch_size=curr_features.shape[0],
                            lr=rw_cfg.get('dwr_lr', 0.005),
                            num_steps=rw_cfg.get('dwr_steps', 10),
                            clip_min=rw_cfg.get('clip_min', 0.8),
                            clip_max=rw_cfg.get('clip_max', 1.2),
                            alpha=rw_cfg.get('alpha', 1.0),
                            time_scale=rw_cfg.get('time_scale', 4.0),
                            logger=logger if (rank == 0 and train_steps % rw_cfg.get('debug_print_freq', 100) == 0) else None,
                            debug_step=train_steps if (rank == 0 and train_steps % rw_cfg.get('debug_print_freq', 100) == 0) else 0
                        )
                    else:
                        # === 原有 DWR 逻辑 ===
                        # 3. 记忆库拼接: 获取 (Current + History) 扩大统计样本量
                        combined_features = reweight_bank.get_combined_features(curr_features)
                        
                        # 4. 计算权重 (带 Alpha 平滑与 Clipping)
                        #    dwr_lr 和 steps 控制优化力度，clip 防止权重爆炸
                        batch_weights = compute_dwr_weights_online(
                            combined_features.float(), 
                            current_batch_size=curr_features.shape[0],
                            lr=rw_cfg.get('dwr_lr', 0.005),       
                            num_steps=rw_cfg.get('dwr_steps', 10),
                            clip_min=rw_cfg.get('clip_min', 0.8), # 保守下限
                            clip_max=rw_cfg.get('clip_max', 1.2), # 保守上限
                            alpha=rw_cfg.get('alpha', 1.0)        # 平滑系数
                        )
                        
                        # 7. 更新记忆库 (时空版本在函数内部更新)
                        reweight_bank.update(curr_features)
                    
                    # 5. [实验设想2] 轨迹约束
                    #    如果配置开启 group_by_traj，则强制同一条轨迹的4帧使用相同权重
                    #    这有助于我们探索 "轨迹级" 的 OOD 特性
                    goals_per_obs = int(config["datasets"]["recon"]["goals_per_obs"])
                    if rw_cfg.get('group_by_traj', False):
                        batch_weights = apply_group_weighting(batch_weights, goals_per_obs)

                    # 6. [关键调试] 打印权重与轨迹的关系 (详细仪表盘)
                    #    每隔 debug_print_freq 步，并在 rank 0 打印
                    if rank == 0 and train_steps % rw_cfg.get('debug_print_freq', 100) == 0:
                         with torch.no_grad():
                            w_reshaped = batch_weights.view(-1, goals_per_obs) # (Batch, Goals)
                            
                            # 计算方差统计量
                            intra_std = w_reshaped.std(dim=1).mean().item() # 轨迹内波动 (Intra)
                            inter_std = w_reshaped.mean(dim=1).std().item() # 轨迹间差异 (Inter)
                            
                            # print(f"\n[Reweight Debug Step {train_steps}]")
                            # print(f"  Stats: Mean={batch_weights.mean():.3f}, Std={batch_weights.std():.3f}")
                            # print(f"  Relation: Intra-Traj Std={intra_std:.4f} (内), Inter-Traj Std={inter_std:.4f} (间)")
                            # print(f"  First Traj Weights: {w_reshaped[0].cpu().numpy().round(3)}")
                            # print("-" * 40)

                            logger.info(f"\n[Reweight Debug Step {train_steps}]")
                            logger.info(f"  Stats: Mean={batch_weights.mean():.3f}, Std={batch_weights.std():.3f}")
                            logger.info(
                                f"  Relation: Intra-Traj Std={intra_std:.4f} (内), "
                                f"Inter-Traj Std={inter_std:.4f} (间)"
                            )
                            logger.info(f"  First Traj Weights: {w_reshaped[0].cpu().numpy().round(3)}")
                            logger.info("-" * 40)
                # =========================================================

                # 计算原始 Loss (Vector if reduction='none')
                model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
                loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                raw_loss = loss_dict["loss"]
                
                # 安全检查
                if log_steps == 0 and rank == 0 and raw_loss.dim() == 0:
                     raise ValueError("严重错误：diffusion 返回的是标量 Loss，请检查 reduction='none'")

                # 应用权重
                if raw_loss.dim() > 0:
                    # 将权重应用到 Loss 上
                    loss = (raw_loss * batch_weights).mean()
                else:
                    loss = raw_loss.mean()

                # === [防崩溃核心] Loss 缩放 ===
                # 因为梯度是累加的，所以 Loss 必须除以累积次数，否则梯度会大 accum_iter 倍
                loss = loss / accum_iter

            # 反向传播
            if not bfloat_enable:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            
            # === [防崩溃核心] 参数更新 ===
            # 只有当积攒了 accum_iter 步的梯度，才真正 step 一次优化器
            if (data_iter_step + 1) % accum_iter == 0:
                if not bfloat_enable:
                    if config.get('grad_clip_val', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                    opt.step()
                    opt.zero_grad() 
                else:
                    if config.get('grad_clip_val', 0) > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad() 
                
                # 同步更新 EMA
                update_ema(ema, model.module)
                train_steps += 1
                
                # === [防死锁] 每隔 100 步强制同步，防止各 rank 累积误差导致 NCCL 超时 ===
                if train_steps % 100 == 0:
                    torch.cuda.synchronize()
                    dist.barrier()
                    # 每 100 步打印进度，帮助定位卡死位置
                    if rank == 0:
                        logger.info(f"[Debug-100] train_steps={train_steps}")
                
                # 记录还原后的真实 Loss (用于日志显示)
                running_loss += loss.item() * accum_iter
                log_steps += 1

                # === 日志打印 ===
                if train_steps % args.log_every == 0:
                    # === [关键修复] 确保所有 rank 都到达这里再执行 all_reduce ===
                    torch.cuda.synchronize()
                    if rank == 0:
                        logger.info(f"[Debug-log] Before barrier, train_steps={train_steps}")
                    dist.barrier()
                    
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Samples/sec 需要算上 accum_iter
                    samples_per_sec = dist.get_world_size() * x_cond.shape[0] * steps_per_sec * accum_iter
                    
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    
                    if rank == 0:
                        logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                    
                    running_loss = 0
                    log_steps = 0
                    start_time = time()
                    
                    torch.cuda.synchronize()
                    if rank == 0:
                        logger.info(f"[Debug-log] After allreduce, train_steps={train_steps}")
                    dist.barrier()

                # === 保存 Checkpoint ===
                
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    # === [关键修复] 确保所有 rank 都完成当前迭代 ===
                    torch.cuda.synchronize()
                    if rank == 0:
                        logger.info(f"[Debug-ckpt] Before barrier, train_steps={train_steps}")
                    dist.barrier()
                    if rank == 0:
                        logger.info(f"[Debug-ckpt] After barrier, building checkpoint...")

                    # 构建 checkpoint（所有rank都构建，但只有rank 0保存）
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    
                    # === [关键修复] 保存 latest checkpoint ===
                    # if rank == 0:
                    #     checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    #     torch.save(checkpoint, checkpoint_path)
                    #     logger.info(f"Saved latest checkpoint to {checkpoint_path}")
                    #     #torch.cuda.synchronize()
                    
                    # # === [关键修复] latest 保存后对齐 ===
                    # torch.cuda.synchronize()
                    # dist.barrier()
                    
                    # # === [关键修复] 保存带数字的 checkpoint ===
                    if train_steps % (2*args.ckpt_every) == 0:
                        if rank == 0:
                            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved numbered checkpoint to {checkpoint_path}")
                            #torch.cuda.synchronize()
                        
                        # === [关键修复] numbered checkpoint 保存后对齐 ===
                        torch.cuda.synchronize()
                        dist.barrier()
                    
                    # # === [关键修复] 保存 loss 最低的 best checkpoint ===
                    # # 只有 rank 0 判断并保存，避免多rank同时进入导致不同步
                    # if rank == 0 and 'avg_loss' in dir() and avg_loss < best_loss:
                    #     best_loss = avg_loss
                    #     best_checkpoint = {
                    #         "model": model.module.state_dict(),
                    #         "ema": ema.state_dict(),
                    #         "opt": opt.state_dict(),
                    #         "args": args,
                    #         "epoch": epoch,
                    #         "train_steps": train_steps,
                    #     }
                    #     if bfloat_enable:
                    #         best_checkpoint.update({"scaler": scaler.state_dict()})
                    #     best_ckpt_path = f"{checkpoint_dir}/best.pth.tar"
                    #     torch.save(best_checkpoint, best_ckpt_path)
                    #     logger.info(f"Saved best checkpoint with loss {best_loss:.4f} to {best_ckpt_path}")

                    # # === [关键修复] best checkpoint 保存后对齐（所有 rank 都要等待）===
                    # torch.cuda.synchronize()
                    # dist.barrier()

                # === 评估 (建议设大 eval_every 避免 OOM) ===
                if train_steps % args.eval_every == 0 and train_steps > 0:
                    eval_start_time = time()
                    save_dir = os.path.join(experiment_dir, str(train_steps))
                    sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond)
                    dist.barrier()
                    logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}")

        # === [关键修复] epoch末checkpoint保存需要先同步 ===
        # 所有rank必须同时到达这里才能确保epoch边界一致
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # 保存 epoch checkpoint
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "epoch": epoch,  # 修复: 直接使用epoch，不要+1
                "train_steps": train_steps
            }
            if bfloat_enable:
                checkpoint.update({"scaler": scaler.state_dict()})
            
            # 保存 epoch checkpoint
            epoch_checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1:03d}_step_{train_steps:07d}.pth.tar"
            torch.save(checkpoint, epoch_checkpoint_path)
            logger.info(f"Saved epoch checkpoint to {epoch_checkpoint_path}")
            torch.cuda.synchronize()
        
        # === [关键修复] epoch checkpoint 保存后对齐 ===
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # 更新 latest
        if rank == 0:
            latest_checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
            torch.save(checkpoint, latest_checkpoint_path)
            logger.info(f"Updated latest checkpoint at end of epoch {epoch}")
            torch.cuda.synchronize()
        
        # === [关键修复] latest checkpoint 保存后对齐 ===
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # 每个 rank 都保存自己的 reweight_bank 状态
        # [关键修复] 先确保所有rank到达这里再各自保存
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        reweight_state = reweight_bank.get_state_dict()
        reweight_ckpt_path = f"{checkpoint_dir}/reweight_bank_rank{rank}.pt"
        torch.save(reweight_state, reweight_ckpt_path)
        
        # === [关键修复] reweight_bank 保存后对齐 ===
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        logger.info(f"[rank{rank}] Saved reweight_bank state to {reweight_ckpt_path}")
        
        dist.barrier()
        logger.info(f"Completed epoch {epoch}, total steps: {train_steps}")
    
 
    # === [关键修复] final checkpoint保存需要同步 ===
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # 保存 final checkpoint
    if rank == 0:
        final_checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "epoch": args.epochs,
            "train_steps": train_steps
        }
        if bfloat_enable:
            final_checkpoint.update({"scaler": scaler.state_dict()})
        
        # 保存最终 checkpoint
        final_checkpoint_path = f"{checkpoint_dir}/final_step_{train_steps:07d}.pth.tar"
        torch.save(final_checkpoint, final_checkpoint_path)
        logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        torch.cuda.synchronize()
    
    # === [关键修复] final checkpoint 保存后对齐 ===
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # 更新 latest
    if rank == 0:
        latest_checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
        torch.save(final_checkpoint, latest_checkpoint_path)
        logger.info(f"Updated latest checkpoint at training end")
        torch.cuda.synchronize()
    
    # === [关键修复] 确保所有保存完成后再继续 ===
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    model.eval()  
    logger.info("Done!")
    cleanup()




@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for x, y, rel_t in loader:
        #x的形状是(B, T, C, H, W)，y的形状是(B, T, 3)，rel_t的形状是(B, T)，B是batch size，T是历史长度+未来真值长度，C是通道数，H是高度，W是宽度。
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            #sample是模型预测的未来帧，输入历史帧（长了会自动截取到num_cond长度）、目标位姿、时间差，输出预测的latent空间图像。这里一次性预测的帧数是B*num_goals.
            samples = model_forward_wrapper((model, diffusion, vae), x, y, num_timesteps=None, latent_size=latent_size, device=device, num_cond=num_cond, num_goals=num_goals, rel_t=rel_t)
            #在第二维度即T上进行拆分，并flatten，x_start_pixels 是目标帧真值，x_cond_pixels 是历史帧真值。
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            #具体来说unsqueeze(1)在第二维度插入一个维度，expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4])在第二维度扩展num_goals倍，即对于每一个要预测的未来帧都把完整的历史帧传进去，输出(B * num_goals, num_cond, C, H, W)，给定相同历史片段，预测未来单帧
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            #将图像数据的数值范围从 [-1, 1] 映射回 [0, 1]
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            #长度为B*num_goals的预测帧和B*num_goals的真值帧，计算相似度dreamsim
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break
    # === [关键修复] 确保所有 rank 都完成循环再执行 all_reduce ===
    dist.barrier()
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            _, ax = plt.subplots(1,3,dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[2].imshow((samples[i].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
