import torch
import torch.nn as nn



class VITA(nn.Module):
    def __init__(self, trajectory_dim=3, condition_dim=1606, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        # 条件编码
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim)

        )

        # 时间步编码
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        # 特征融合编码
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),  
            nn.Linear(hidden_dim, hidden_dim) 
        )

        # 轨迹编码
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        # 输出层 - 预测向量场
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, trajectory_dim)
        )

    def forward(self, noisy_trajectory, condition, timesteps):
        """
        noisy_trajectory: [batch_size, seq_len, trajectory_dim] - 在flow matching中这是插值样本
        condition: [batch_size, condition_dim]
        timesteps: [batch_size] - 在flow matching中是时间t ∈ [0,1]
        """
        batch_size, seq_len, _ = noisy_trajectory.shape
        
        # 1. 编码轨迹
        traj_embed = self.trajectory_encoder(noisy_trajectory)  # [batch_size, seq_len, hidden_dim]
        
        # 2. 编码条件信息
        cond_embed = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        cond_embed = cond_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 3. 编码时间步 (flow matching中t ∈ [0,1])
        time_embed = self.time_embedding(timesteps.unsqueeze(1).float())  # [batch_size, hidden_dim]
        time_embed = time_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 4. 合并所有特征
        combined = torch.cat([traj_embed, cond_embed, time_embed], dim=-1)  # [batch_size, seq_len, hidden_dim * 3]
        combined = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_dim]

        # # 5. Transformer处理
        # encoded = self.transformer(combined)  # [batch_size, seq_len, hidden_dim]

        # 6. 输出预测的向量场 (在flow matching中预测的是目标方向)
        predicted_vector_field = self.output_layer(combined)  # [batch_size, seq_len, trajectory_dim]
        
        return predicted_vector_field
        

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














































