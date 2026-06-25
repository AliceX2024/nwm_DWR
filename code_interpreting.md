入口脚本 train.py：负责分布式初始化、读取配置、构建数据集、创建模型/扩散过程、训练循环、评估与 checkpoint 管理。
数据层 datasets.py：封装视觉导航数据（多轨迹 .jpg + traj_data.pkl）的索引、抽样、归一化和张量化逻辑，提供训练 / 评估 Dataset。
模型层 models.py：实现 CDiT 结构（Diffusion Transformer with action conditioning），包含时间/动作嵌入、DiT block、最终头等模块。
扩散层 diffusion/*：提供噪声日程、timestep 采样等 DDPM 组件，供训练时 create_diffusion() 使用。
辅助模块 misc.py, distributed.py, isolated_nwm_infer.py 等：负责规范化函数、分布式包装、评估/推理封装。
