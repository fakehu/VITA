import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_flow_matching(model, train_loader, val_loader=None, epochs=100, lr=1e-4, device='cuda'):
    """
    训练 Flow Matching 模型
    """

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in train_pbar:
            # 获取数据
            trajectories = batch['trajectory'].to(device)  # [batch_size, seq_len, 3]
            conditions = batch['condition'].to(device)     # [batch_size, condition_dim]
            
            batch_size = trajectories.shape[0]
            
            # 为 flow matching 准备数据
            # x0: 噪声轨迹，x1: 真实轨迹
            x0 = torch.randn_like(trajectories)  # 标准正态噪声
            x1 = trajectories  # 真实轨迹
            
            # 随机时间步 t ~ U[0,1]
            t = torch.rand(batch_size, device=device)
            
            # 梯度置零
            optimizer.zero_grad()
            
            # 计算损失
            loss = flow_matching_loss(model, x0, x1, conditions, t)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        

        # 学习率调度
        scheduler.step()
    
    return train_losses, val_losses



# Flow Matching 训练函数示例
def flow_matching_loss(model, x0, x1, condition, t):
    """
    x0: 初始样本 [batch_size, seq_len, dim]
    x1: 目标样本 [batch_size, seq_len, dim] 
    condition: 条件信息 [batch_size, condition_dim]
    t: 随机时间 [batch_size]
    """
    # 线性插值
    xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
    
    # 真实向量场: 从xt指向x1的方向
    true_vector_field = x1 - x0
    
    # 预测向量场
    pred_vector_field = model(xt, condition, t)
    
    # MSE损失
    loss = torch.mean((pred_vector_field - true_vector_field) ** 2)
    return loss








