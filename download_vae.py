import os
import sys

# 强制设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 定义保存路径 (绝对路径)
save_directory = "/DATA/DATANAS2/xiaoyj25/projects/nwm/models/sd-vae-ft-ema"

print(f"Python 进程已启动，准备导入库...", flush=True)

try:
    from diffusers import AutoencoderKL
    print("库导入成功，准备下载...", flush=True)
    
    # 确保目录存在
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"正在下载模型到: {save_directory}")
    print("这可能需要几分钟，请留意下方进度条...", flush=True)
    
    # 下载并保存
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.save_pretrained(save_directory)
    
    print(f"\n[成功] 模型已保存到: {save_directory}")

except Exception as e:
    print(f"\n[失败] 发生错误: {e}")