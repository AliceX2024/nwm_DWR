"""
维护一个全局特征库（Global Features $Z_G$）和全局权重库（Global Weights $w_G$）。
在每一轮训练中，它把当前 Batch 的特征（Local $Z_L$）和全局特征库拼接起来（Concat），形成一个很大的 $Z_O$ 。
计算权重时：利用这个拼接后的大 Batch 计算权重，既利用了当前信息，又利用了历史全局信息，解决了 Batch Size 小的问题。
更新库：训练完一步后，用移动平均（Moving Average）的方式把当前 Batch 的信息融合进全局库。
"""


import torch
import torch.nn.functional as F

class GlobalFeatureBank:
    def __init__(self, feature_dim, bank_size=2048, device='cuda', mode='random'):
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.device = device
        self.mode = mode
        self.use_bank = bank_size > 0
        self.ptr = 0
        self.is_full = False
        self._rng_state = None
        self.features = torch.randn(bank_size, feature_dim, device=device).detach() if self.use_bank else None

    def get_state_dict(self):
        """获取用于保存的状态字典"""
        return {
            'features': self.features.cpu() if self.features is not None else None,
            'ptr': self.ptr,
            'is_full': self.is_full,
            '_rng_state': self._rng_state,
        }

    def load_state_dict(self, state_dict):
        """从状态字典恢复"""
        if state_dict.get('features', None) is None:
            self.use_bank = False
            self.features = None
            self.ptr = 0
            self.is_full = False
            self._rng_state = state_dict.get('_rng_state', None)
            return
        self.features = state_dict['features'].to(self.device)
        self.ptr = state_dict['ptr']
        self.is_full = state_dict['is_full']
        self._rng_state = state_dict.get('_rng_state', None)

    def update(self, batch_features):
        if not self.use_bank:
            return
        batch_features = batch_features.detach()
        n = batch_features.shape[0]

        if self.mode == 'fifo':
            if self.ptr + n <= self.bank_size:
                self.features[self.ptr:self.ptr+n] = batch_features
                self.ptr = (self.ptr + n) % self.bank_size
            else:
                tail = self.bank_size - self.ptr
                self.features[self.ptr:] = batch_features[:tail]
                self.features[:n-tail] = batch_features[tail:]
                self.ptr = n - tail
                self.is_full = True
            if self.ptr == 0: self.is_full = True

        elif self.mode == 'random':
            if not self.is_full and self.ptr + n <= self.bank_size:
                self.features[self.ptr:self.ptr+n] = batch_features
                self.ptr += n
                if self.ptr >= self.bank_size: self.is_full = True
            else:
                self.is_full = True
                random_indices = torch.randperm(self.bank_size, device=self.device)[:n]
                self.features[random_indices] = batch_features

        elif self.mode == 'diversity':
            if not self.is_full and self.ptr + n <= self.bank_size:
                self.features[self.ptr:self.ptr+n] = batch_features
                self.ptr += n
                if self.ptr >= self.bank_size:
                    self.is_full = True
            else:
                self.is_full = True
                batch_norm = F.normalize(batch_features, p=2, dim=1)
                bank_norm = F.normalize(self.features, p=2, dim=1)
                similarity = torch.mm(batch_norm, bank_norm.T)
                accum_scores = similarity.sum(dim=0)
                _, replace_indices = torch.topk(accum_scores, k=n, sorted=True)
                self.features[replace_indices] = batch_features

    def get_combined_features(self, current_features):
        if not self.use_bank:
            return current_features
        if not self.is_full: 
             return current_features
        return torch.cat([current_features, self.features], dim=0)

# === [修改] 增强版 DWR 计算函数 ===
def compute_dwr_weights_online(X, current_batch_size, lr=0.005, num_steps=10, 
                             clip_min=0.5, clip_max=2.0, alpha=1.0):
    """
    增加了 lr, clip, alpha 等控制参数
    """
    # IMPORTANT:
    # - 权重必须为正，否则 sqrt(weight) 会产生 NaN，进而污染训练（模型会“训坏”，推理变色块）
    # - 该函数在训练时常处于 autocast(bfloat16) 作用域内，强制转为 float32 提升稳定性
    X = X.detach().float()
    n, p = X.shape
    
    # 标准化
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    
    # 用 log_weight 参数化，保证 weight = exp(log_weight) 始终为正
    log_weight = torch.zeros(n, 1, device=X.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([log_weight], lr=lr)  # 使用传入的 lr
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        weight = torch.exp(log_weight)  # (n, 1), strictly positive
        weight_norm = weight / (weight.sum() + 1e-12) * n
        X_w = X * torch.sqrt(weight_norm + 1e-6)
        cov = torch.mm(X_w.T, X_w) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        loss = torch.sum(off_diag ** 2)
        loss.backward()
        optimizer.step()
        
    final_weights = torch.exp(log_weight.detach())

    # 数值保护：出现 NaN/Inf 则回退为全 1（宁可不加权，也不要污染训练）
    if torch.isnan(final_weights).any() or torch.isinf(final_weights).any():
        final_weights = torch.ones_like(final_weights)
    
    # 1. 归一化均值为 1（原始权重）
    final_weights = final_weights / final_weights.mean()
    
    # 2. [设想1] 平滑处理 (Alpha Blending)
    # alpha=1.0 -> 原始DWR权重; alpha=0.0 -> 全1权重
    final_weights = alpha * final_weights + (1 - alpha) * 1.0
    
    # 3. 先 clip 去除极端值，再归一化确保均值为 1
    # clip 去除异常值（双重保险），归一化保持训练稳定
    # 注意：归一化后少量样本可能轻微超出 clip 边界，但范围有限
    final_weights = torch.clamp(final_weights, min=clip_min, max=clip_max)
    final_weights = final_weights / final_weights.mean()
    
    return final_weights[:current_batch_size].squeeze()

# === [新增] 轨迹聚合函数 [设想2] ===
def apply_group_weighting(weights, goals_per_obs):
    """
    强制让同一条轨迹内的权重相同。
    weights: [B * goals_per_obs]
    """
    N = weights.shape[0]
    batch_size = N // goals_per_obs
    
    # Reshape 成 (Batch, Goals)
    weights_reshaped = weights.view(batch_size, goals_per_obs)
    
    # 计算每条轨迹的平均权重
    traj_weights = weights_reshaped.mean(dim=1, keepdim=True) # (Batch, 1)
    
    # 广播回 (Batch, Goals) 并展平
    final_weights = traj_weights.expand(batch_size, goals_per_obs).flatten()
    
    return final_weights


# === [新增] 时空联合特征 DWR (维度: base_dim*2 + 1) ===
def compute_dwr_weights_spacetime(curr_features, obs_features, rel_t, 
                                  reweight_bank, current_batch_size,
                                  lr=0.005, num_steps=10,
                                  clip_min=0.5, clip_max=2.0, alpha=1.0,
                                  time_scale=4.0,
                                  logger=None, debug_step=0):
    """
    时空联合特征 DWR: 让模型自主学习静态特征、残差特征、时间跨度之间的关系
    
    输入:
        curr_features: 目标帧特征 [B*goals, base_dim]
            - base_dim=4: 全局平均池化
            - base_dim=16: 2x2 自适应池化 (4*2*2)
            - base_dim=64: 4x4 自适应池化 (4*4*4)
            - base_dim=3136: 原始高维特征 (4*28*28)
        obs_features: 历史帧最后时刻特征 [B*goals, base_dim]
        rel_t: 归一化时间跨度 [B*goals] (范围 -0.5 到 0.5)
        reweight_bank: GlobalFeatureBank 实例
        current_batch_size: 当前 batch 大小
        time_scale: 时间特征放大倍数 (默认4.0, 设置为1.0则不放大)
    
    输出:
        时空联合权重 [B*goals]
    
    时空特征维度: base_dim (静态) + base_dim (残差) + 1 (时间) = base_dim*2 + 1
    """
    # [调试] 打印输入特征统计
    has_debug = logger is not None and debug_step > 0
    
    if has_debug:
        logger.info(f"\n[DEBUG spacetime Step {debug_step}]")
        logger.info(f"  curr_features: mean={curr_features.mean():.6f}, std={curr_features.std():.6f}")
        logger.info(f"  obs_features: mean={obs_features.mean():.6f}, std={obs_features.std():.6f}")
        logger.info(f"  rel_t: min={rel_t.min():.6f}, max={rel_t.max():.6f}, mean={rel_t.mean():.6f}")
    
    # 1. 计算残差特征 (4维)
    residual_features = curr_features - obs_features  # [B*goals, 4]
    
    # [调试] 打印残差特征
    if has_debug:
        logger.info(f"  residual (curr - obs): mean={residual_features.mean():.6f}, std={residual_features.std():.6f}")
    
    # 2. 提取时间跨度 (1维), 可选择放大时间信号以增强时空关联
    # 原始 rel_t 范围约 [-0.5, 0.5]
    # time_scale=4.0: 放大到约 [-2, 2]
    # time_scale=1.0: 不放大
    time_features = (rel_t * time_scale).unsqueeze(-1)  # [B*goals, 1]
    
    # 3. 拼接为时空联合特征
    # 维度: 静态(base_dim) + 残差(base_dim) + 时间(1)
    #    - base_dim=4 (avg池化): 4 + 4 + 1 = 9
    #    - base_dim=16 (2x2池化): 16 + 16 + 1 = 33
    #    - base_dim=64 (4x4池化): 64 + 64 + 1 = 129
    #    - base_dim=3136 (原始): 3136 + 3136 + 1 = 6273
    spacetime_features = torch.cat([
        curr_features,        # 静态特征 (base_dim维)
        residual_features,   # 残差特征 (base_dim维)
        time_features        # 时间跨度 (1维, 放大后)
    ], dim=-1)
    
    # [调试] 打印拼接后的时空特征
    if has_debug:
        base_dim = curr_features.shape[1]
        logger.info(f"  spacetime_features shape: {spacetime_features.shape}")
        logger.info(f"    - {base_dim} dims (curr/static): mean={spacetime_features[:, :base_dim].mean():.6f}, std={spacetime_features[:, :base_dim].std():.6f}")
        logger.info(f"    - {base_dim} dims (residual): mean={spacetime_features[:, base_dim:2*base_dim].mean():.6f}, std={spacetime_features[:, base_dim:2*base_dim].std():.6f}")
        logger.info(f"    - 1 dim (time): mean={spacetime_features[:, -1].mean():.6f}, std={spacetime_features[:, -1].std():.6f}")
    
    # 4. 从 memory bank 获取历史时空特征
    combined_features = reweight_bank.get_combined_features(spacetime_features)
    
    # [调试] 打印拼接后的特征 (当前batch + bank)
    if has_debug:
        logger.info(f"  combined_features shape: {combined_features.shape}")
        logger.info(f"  combined (curr+bank): mean={combined_features.mean():.6f}, std={combined_features.std():.6f}")
    
    # 5. 使用原有的 DWR 逻辑计算权重
    combined_features = combined_features.detach().float()
    n, p = combined_features.shape
    
    # 标准化 (对9维特征分别标准化)
    combined_features = (combined_features - combined_features.mean(dim=0)) / (combined_features.std(dim=0) + 1e-6)
    
    # 用 log_weight 参数化
    log_weight = torch.zeros(n, 1, device=combined_features.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([log_weight], lr=lr)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        weight = torch.exp(log_weight)
        weight_norm = weight / (weight.sum() + 1e-12) * n
        X_w = combined_features * torch.sqrt(weight_norm + 1e-6)
        cov = torch.mm(X_w.T, X_w) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        loss = torch.sum(off_diag ** 2)
        loss.backward()
        optimizer.step()
        
    final_weights = torch.exp(log_weight.detach())
    
    # 数值保护
    if torch.isnan(final_weights).any() or torch.isinf(final_weights).any():
        final_weights = torch.ones_like(final_weights)
    
    # 1. 归一化均值为 1（原始权重）
    final_weights = final_weights / final_weights.mean()
    
    # 2. 平滑处理 (Alpha Blending)
    # alpha=1.0 -> 原始DWR权重; alpha=0.0 -> 全1权重
    final_weights = alpha * final_weights + (1 - alpha) * 1.0
    
    # 3. 先 clip 去除极端值，再归一化确保均值为 1
    # clip 去除异常值（双重保险），归一化保持训练稳定
    # 注意：归一化后少量样本可能轻微超出 clip 边界，但范围有限
    final_weights = torch.clamp(final_weights, min=clip_min, max=clip_max)
    final_weights = final_weights / final_weights.mean()
    
    # 更新 memory bank
    reweight_bank.update(spacetime_features)
    
    return final_weights[:current_batch_size].squeeze()


# version1
# import torch
# import torch.nn.functional as F

# class GlobalFeatureBank:
#     """
#     # 实现 StableNet 中的 Saving and Reloading 机制。
#     # 维护一个全局特征队列，用于在 Batch Size 较小时估计全局分布。
#     """
#     def __init__(self, feature_dim, bank_size=2048, device='cuda'):
#         self.bank_size = bank_size
#         self.feature_dim = feature_dim
#         self.device = device
        
#         # 初始化 buffer，使用随机值或零初始化
#         self.features = torch.randn(bank_size, feature_dim, device=device).detach()
#         self.ptr = 0 # 指针，指示当前写入位置
#         self.is_full = False

#     def update(self, batch_features):
#         """
#         #更新记忆库：将当前 batch 的特征写入 buffer
#         """
#         batch_features = batch_features.detach() # 必须切断梯度，只存数值
#         n = batch_features.shape[0]
        
#         # 如果当前 batch 大于剩余空间，分段写入（简化逻辑，通常 bank_size >> batch_size）
#         assert n <= self.bank_size, "Batch size is larger than bank size!"

#         if self.ptr + n <= self.bank_size:
#             self.features[self.ptr:self.ptr+n] = batch_features
#             self.ptr = (self.ptr + n) % self.bank_size
#         else:
#             # 循环写入
#             tail = self.bank_size - self.ptr
#             self.features[self.ptr:] = batch_features[:tail]
#             self.features[:n-tail] = batch_features[tail:]
#             self.ptr = n - tail
#             self.is_full = True
            
#         if self.ptr == 0:
#             self.is_full = True

#     def get_combined_features(self, current_features):
#         """
#         #返回 [Current_Batch + Memory_Bank] 的拼接特征
#         #用于计算更稳健的统计量 (如协方差)
#         """
#         # 如果 buffer 还没满，只返回当前特征，避免引入初始化的随机噪声
#         if not self.is_full and self.ptr < 100: 
#              return current_features
             
#         return torch.cat([current_features, self.features], dim=0)

# # ---  DWR 算法 (经过优化以适应 float16 和速度) ---
# def compute_dwr_weights_online(X, current_batch_size, order=2, num_steps=10, lr=0.01):
#     """
#     #计算权重。
#     #X: 拼接后的特征 (Current + Bank)
#     #current_batch_size: 当前 batch 的大小，用于最后切片返回
#     """
#     X = X.detach()
#     n, p = X.shape
    
#     # 简单的归一化，防止数值爆炸
#     X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    
#     weight = torch.ones(n, 1, device=X.device)
#     weight.requires_grad = True
#     optimizer = torch.optim.Adam([weight], lr=lr)
    
#     # 简化的去相关损失计算
#     for _ in range(num_steps):
#         optimizer.zero_grad()
#         # 计算加权协方差
#         weight_norm = weight / weight.sum() * n # 保持和为 n
#         X_w = X * torch.sqrt(weight_norm + 1e-6)
#         cov = torch.mm(X_w.T, X_w) / (n - 1)
        
#         # Loss: 让非对角线元素趋于 0 (去相关)
#         off_diag = cov - torch.diag(torch.diag(cov))
#         loss = torch.sum(off_diag ** 2)
        
#         loss.backward()
#         optimizer.step()
        
#     # 后处理权重
#     final_weights = weight.detach()
#     final_weights = torch.clamp(final_weights, min=0.1, max=10.0) # 截断极端权重
#     final_weights = final_weights / final_weights.mean() # 归一化均值为 1
    
#     # 【关键】只返回属于当前 Batch 的权重 (前 N 个)
#     return final_weights[:current_batch_size].squeeze()
