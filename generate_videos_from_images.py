"""
将rollout_4fps目录下的图片序列生成为视频文件
使用 imageio 生成 MP4 视频
"""
import os
import imageio
from PIL import Image
import numpy as np
from tqdm import tqdm

def images_to_video(image_dir, output_path, fps=8):
    """
    将指定目录下的图片按序号排序后生成视频

    Args:
        image_dir: 包含图片的目录路径
        output_path: 输出视频路径
        fps: 帧率
    """
    # 获取所有png图片并按序号排序
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: int(x.replace('.png', '')))

    if len(image_files) == 0:
        print(f"警告: {image_dir} 中没有找到PNG图片")
        return False

    # 使用 imageio 生成视频
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')

    for image_file in tqdm(image_files, desc=f"处理 {os.path.basename(image_dir)}"):
        img_path = os.path.join(image_dir, image_file)
        
        # 使用 PIL 读取图片，确保 RGB 格式
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        writer.append_data(img_array)

    writer.close()
    print(f"已生成视频: {output_path}")
    return True

def generate_all_videos(base_dir, output_dir, fps=8):
    """
    遍历base_dir下所有id文件夹，为每个生成视频

    Args:
        base_dir: rollout_4fps目录路径
        output_dir: 视频输出目录
        fps: 帧率
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有id文件夹
    id_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('id_')]
    id_dirs = sorted(id_dirs, key=lambda x: int(x.replace('id_', '')))

    print(f"找到 {len(id_dirs)} 个视频序列")

    for id_dir in tqdm(id_dirs, desc="生成所有视频"):
        full_path = os.path.join(base_dir, id_dir)
        video_name = f"{id_dir}.mp4"
        output_path = os.path.join(output_dir, video_name)

        if os.path.exists(output_path):
            print(f"跳过已存在的视频: {video_name}")
            continue

        images_to_video(full_path, output_path, fps)

if __name__ == "__main__":
    # 配置路径 - 修改这里来生成不同目录的视频
   #base_dir = "/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/4datasets_clip04_CDiTXL_temmeDWR_eval_0001600/go_stanford/rollout_4fps"
    #output_dir = os.path.join(os.path.dirname(base_dir), "videos")
    
    # base_dir = "/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/4datasets_clip04_CDiTXL_base_eval/go_stanford/rollout_4fps"
    # output_dir = os.path.join(os.path.dirname(base_dir), "videos")
    base_dir = "/DATA/DATANAS2/xiaoyj25/projects/nwm/eval_results/4datasets_clip04_CDiTXL_temmeDWR_ts2_eval_0002400/go_stanford/rollout_4fps"
    output_dir = os.path.join(os.path.dirname(base_dir), "videos")

    # 生成视频，帧率8fps
    generate_all_videos(base_dir, output_dir, fps=8)

    print(f"\n所有视频已保存到: {output_dir}")

