#!/usr/bin/env python3
"""
快速测试预训练模型是否加载成功
使用方法: python test_pretrained.py --ckp 0100000
"""

import torch
import yaml
import argparse
import os
from models import CDiT_models
from diffusers.models import AutoencoderKL

def test_model_loading(ckp_path, model_name="CDiT-L/2", context_size=4, image_size=224):
    """测试模型加载"""
    print(f"测试加载模型: {model_name}")
    print(f"Checkpoint 路径: {ckp_path}")
    print(f"Context size: {context_size}")
    print("-" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(ckp_path):
        print(f"❌ 错误: Checkpoint 文件不存在: {ckp_path}")
        return False
    
    try:
        # 加载 checkpoint
        print("1. 加载 checkpoint...")
        ckp = torch.load(ckp_path, map_location='cpu', weights_only=False)
        print(f"   ✅ Checkpoint 加载成功")
        print(f"   Checkpoint keys: {list(ckp.keys())}")
        
        # 检查是否有 'ema' 键
        if 'ema' not in ckp:
            print("   ⚠️  警告: Checkpoint 中没有 'ema' 键")
            if 'model' in ckp:
                print("   ℹ️  找到 'model' 键，将使用它")
                ckp['ema'] = ckp['model']
            else:
                print("   ❌ 错误: Checkpoint 中没有 'ema' 或 'model' 键")
                return False
        
        # 创建模型
        print("2. 创建模型...")
        latent_size = image_size // 8
        model = CDiT_models[model_name](
            context_size=context_size,
            input_size=latent_size,
            in_channels=4
        )
        print(f"   ✅ 模型创建成功")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 加载权重
        print("3. 加载模型权重...")
        missing_keys, unexpected_keys = model.load_state_dict(ckp["ema"], strict=False)
        
        if len(missing_keys) > 0:
            print(f"   ⚠️  缺失的键 ({len(missing_keys)} 个):")
            for key in missing_keys[:10]:  # 只显示前10个
                print(f"      - {key}")
            if len(missing_keys) > 10:
                print(f"      ... 还有 {len(missing_keys) - 10} 个")
        
        if len(unexpected_keys) > 0:
            print(f"   ⚠️  意外的键 ({len(unexpected_keys)} 个):")
            for key in unexpected_keys[:10]:
                print(f"      - {key}")
            if len(unexpected_keys) > 10:
                print(f"      ... 还有 {len(unexpected_keys) - 10} 个")
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("   ✅ 权重加载成功，完全匹配")
        elif len(missing_keys) == 0:
            print("   ✅ 权重加载成功（有一些意外的键，可能是正常的）")
        else:
            print("   ⚠️  权重加载完成，但有缺失的键（可能不兼容）")
        
        # 测试前向传播
        print("4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            # 创建虚拟输入
            batch_size = 2
            num_goals = 1  # 1 个目标即可验证前向传播
            latent = torch.randn(batch_size * num_goals, 4, latent_size, latent_size)#预测帧，(B*num_goals, 4, H, W)来自 VAE latent 空间，通道是 4，H=W=latent_size（224/8=28）
            t = torch.randint(0, 1000, (batch_size * num_goals,))#整型时间步（扩散步），与 latent 一一对应。
            y = torch.randn(batch_size * num_goals, 3)  # action: x, y, angle条件动作向量 (dx, dy, dθ)；在测试里是随机填充
            # 保持 (B, T, C, H, W) 形状，forward 内部会自己处理
            x_cond = torch.randn(batch_size * num_goals, context_size, 4, latent_size, latent_size)#历史上下文帧的 VAE latent 序列，T=context_size
            rel_t = torch.ones(batch_size * num_goals) * 0.5#相对时间/归一化步长，测试中是常数 0.5
            
            try:
                output = model(
                    latent,
                    t,
                    y=y,
                    x_cond=x_cond,
                    rel_t=rel_t
                )
                print(f"   ✅ 前向传播成功")
                print(f"   输出形状: {output.shape}")
            except Exception as e:
                print(f"   ❌ 前向传播失败: {e}")
                return False
        
        print("-" * 50)
        print("✅ 所有测试通过！模型可以正常使用。")
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="测试预训练模型加载")
    parser.add_argument("--ckp", type=str, default="0100000", help="Checkpoint 名称（如 0100000）")
    parser.add_argument("--config", type=str, default="config/nwm_cdit_l_eval.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型名称（覆盖配置文件）")
    parser.add_argument("--run_name", type=str, default=None, help="Run name（覆盖配置文件）")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 确定模型名称
    model_name = args.model or config.get('model', 'CDiT-L/2')
    run_name = args.run_name or config.get('run_name', 'nwm_cdit_l')
    results_dir = config.get('results_dir', 'logs')
    context_size = config.get('context_size', 4)
    image_size = config.get('image_size', 224)
    
    # 构建 checkpoint 路径
    ckp_path = f"{results_dir}/{run_name}/checkpoints/{args.ckp}.pth.tar"
    
    print("=" * 50)
    print("预训练模型加载测试")
    print("=" * 50)
    
    success = test_model_loading(ckp_path, model_name, context_size, image_size)
    
    if success:
        print("\n💡 下一步:")
        print("   1. 运行推理: python isolated_nwm_infer.py --exp config/nwm_cdit_l_eval.yaml --ckp 0100000 ...")
        print("   2. 查看评估指南: cat EVAL_GUIDE.md")
    else:
        print("\n❌ 测试失败，请检查:")
        print("   1. Checkpoint 文件是否存在")
        print("   2. 模型名称是否匹配（CDiT-L/2 vs CDiT-XL/2）")
        print("   3. context_size 是否与训练时一致")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

