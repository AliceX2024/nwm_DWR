# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from distributed import init_distributed
import torch
import torch.distributed as torch_dist
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import misc
import distributed as dist
from models import CDiT_models
from datasets import EvalDataset
from PIL import Image


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)
        
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)
    
    
def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]    
    if predefined_index:
        #原逻辑if下只有一行，即
        # predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
        # Use the test path from config instead of hardcoded path
        test_folder = data_config["test"].rstrip('/')
        eval_type_name = eval_type if eval_type else "time"
        pkl_path = f"{test_folder}/{eval_type_name}.pkl"
        # Check if the pkl file exists, if not, rebuild it
        if os.path.exists(pkl_path):
            predefined_index = pkl_path
        else:
            print(f"****** PKL file not found: {pkl_path}, will rebuild index... ******")
            predefined_index = None
    else:
        predefined_index=None

    #每个样本返回 (idx, obs_image, pred_image, delta)，obs_image：context_size 张上下文帧（你配置里是 4）。pred_image：后续 len_traj_pred 张未来帧（你配置里 64），按原序列顺序，delta：对应动作增量
    dataset = EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
                max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
                len_traj_pred=config["eval_len_traj_pred"],
                traj_stride=config["traj_stride"], 
                context_size=config["eval_context_size"],
                normalize=config["normalize"],
                transform=misc.transform,
                goals_per_obs=4,
                predefined_index=predefined_index,
                traj_names='traj_names.txt'
            )
    
    return dataset

@torch.no_grad()
#model_forward_wrapper 是模型推理函数，输入历史帧、目标位姿、时间差，输出预测的latent空间图像。这里一次性预测的帧数是num_goals，即预测目标个数。rel_t是目标帧和当前帧之间的时间差。
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, num_goals=1, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    #输入的x是观测，全是真值，用于生成x_cond
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B)* (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0,1)
        #将x编码为latent空间图像，没有加入噪声，得到latent空间图像x，维度为(B * T, 4, latent_size, latent_size)，4是指通道数，latent_size是latent空间图像的宽高。
        #VAE的思想是把一张确定的图片，编码成了一个“概率分布”，然后从这个分布里“随机抽”了一个样本出来
        #vae.encode(x)提取特征，.latent_dist获取分布信息，包括图像特征的中心位置和不确定性范围（均值和方差）。.sample()采样，也叫重参数化，基于预测出的均值和方差，加入了一点点随机扰动，生成了最终的 Latent 向量
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        #x_cond是切片得到的历史帧，维度为(B * num_goals, num_cond, 4, latent_size, latent_size)，num_cond是历史帧数，4是通道数，latent_size是latent空间图像的宽高。
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        #z是随机的高斯噪声，这代表了扩散过程的 T 时刻（全噪状态）。模型接下来的任务就是把这个 z 变成清晰的未来图像
        z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)      
        #去噪过程，模型看着 x_cond（历史帧），一点点去除 z 里的噪声，最终得到预测的未来帧。sample就是预测的未来第k帧
        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )
        #去噪得到的samples是latent空间图像，需要解码为像素空间图像
        samples = vae.decode(samples / 0.18215).sample
        #clip裁剪，确保输出在[-1,1]范围内
        return torch.clip(samples, -1., 1.)

def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta, num_cond, device):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)
    
    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i+1].to(device)
        if args.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, rollout_stride, args.latent_size, num_cond=num_cond, num_goals=1, device=device)

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1) # append current prediction
        curr_obs = curr_obs[:, 1:] # remove first observation
        visualize_preds(output_dir, idxs, i, x_pred_pixels)

#generate_time 会取秒数序列 secs = [1,2,4,8,16]（因为 num_sec_eval=5，secs=2**i），再乘以 input_fps=4 得到时间步 timestep=4,8,16,32,64。
def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device):
    eval_timesteps = [sec*args.input_fps for sec in secs]#对于每个 sec，计算对应的 timestep=sec*input_fps
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)#如果 --gt 1，直接从 pred_image 里取第 timestep-1 帧作为真值图
        if args.gt:
            x_pred_pixels = gt_output[:, timestep-1].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(all_models, obs_image, curr_delta, timestep, args.latent_size, num_cond=num_cond, num_goals=1, device=device)
        #将该帧保存为 output_dir/dataset/id_<样本索引>/<sec>.png
        visualize_preds(output_dir, idxs, sec, x_pred_pixels)

def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.squeeze()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[batch_idx], True)

@torch.no_grad
def main(args):
    _, _, device, _ = init_distributed()

    # 0601设置推理随机种子（控制 torch.randn / diffusion sampling 随机性）
    if hasattr(args, 'seed') and args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print(args)
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)
    
    if  args.ckp != '0100000':
        args.save_output_dir = args.save_output_dir + "_%s"%(args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    if user_config is None:
        raise ValueError(f"Config file '{exp_eval}' is empty or contains only comments. Please ensure it has valid YAML content.")
    config.update(user_config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)
        ckp = torch.load(f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()
        model.to(device)
        model = torch.compile(model)
        diffusion = create_diffusion(str(250))
        # 保持与训练脚本一致：优先使用环境变量 VAE_MODEL_PATH（如本地缓存路径）
        vae_model_name = os.getenv("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-ema")
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, local_files_only=True).to(device)
        except Exception as e:
            print(f"[VAE] local_files_only load failed ({vae_model_name}): {e}")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        print(f"[VAE] using: {vae_model_name}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_lst = (model, diffusion, vae)

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(config, dataset_name, args.eval_type, predefined_index=True)

        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        curr_data_loader = torch.utils.data.DataLoader(
                            dataset_val, sampler=sampler_val,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False
                        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = dist.MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        
        for data_iter_step, (idxs, obs_image, gt_image, delta) in enumerate(metric_logger.log_every(curr_data_loader, print_freq, header)):
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)
                num_cond = config["context_size"]
                if args.eval_type == 'rollout':
                    for rollout_fps in args.rollout_fps_values:
                        curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)
                        generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs, model_lst, obs_image, gt_image, delta, num_cond, device)
                elif args.eval_type == 'time':
                    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(args, curr_time_output_dir, idxs, model_lst, obs_image, gt_image, delta, secs, num_cond, device)
    # clean up distributed to avoid NCCL warning on exit
    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="eval_results", help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--eval_type", type=str, default=None, help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    
    #0601 === 新增：推理随机种子（控制扩散采样 / torch.randn 随机性）===
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for inference. Each run should use a distinct seed "
                             "to produce different diffusion samples.")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    main(args)