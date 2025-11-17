import torch
import torch.nn as nn
import math 


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
        
class VITA_Large(nn.Module):
    def __init__(self, trajectory_dim=3, condition_dim=1606, hidden_dim=1024, num_timesteps=1000, 
                 num_layers=8, num_heads=16, dropout=0.1):
        super().__init__()
        
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 条件编码 - 更深的网络
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

        # 正弦位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=1000)

        # 时间步编码 - 更丰富的表示
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 轨迹编码 - 更深的编码器
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 特征融合编码 - 更宽更深的融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 2*hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer编码器 - 多层Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层 - 更深的解码器
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.GELU(),
            nn.Linear(hidden_dim//4, trajectory_dim)
        )

        # 残差连接和层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, noisy_trajectory, condition, timesteps):
        """
        noisy_trajectory: [batch_size, seq_len, trajectory_dim]
        condition: [batch_size, condition_dim]
        timesteps: [batch_size] - 在flow matching中是时间t ∈ [0,1]
        """
        batch_size, seq_len, _ = noisy_trajectory.shape
        
        # 1. 编码轨迹
        traj_embed = self.trajectory_encoder(noisy_trajectory)  # [batch_size, seq_len, hidden_dim]
        
        # 2. 编码条件信息
        cond_embed = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        cond_embed = cond_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 3. 编码时间步
        time_embed = self.time_embedding(timesteps.unsqueeze(1).float())  # [batch_size, hidden_dim]
        time_embed = time_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 4. 合并所有特征
        combined = torch.cat([traj_embed, cond_embed, time_embed], dim=-1)  # [batch_size, seq_len, hidden_dim * 3]
        fused_features = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_dim]

        # 5. 添加位置编码
        fused_features = self.positional_encoding(fused_features)

        # 6. Transformer处理 (使用残差连接)
        transformer_input = self.layer_norm1(fused_features)
        encoded = self.transformer(transformer_input)  # [batch_size, seq_len, hidden_dim]
        encoded = self.layer_norm2(encoded + fused_features)  # 残差连接

        # 7. 输出预测的向量场
        predicted_vector_field = self.output_layer(encoded)  # [batch_size, seq_len, trajectory_dim]
        
        return predicted_vector_field


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)
    
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




# 参数量对比函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试参数量
if __name__ == "__main__":
    # 原始模型
    original_model = VITA(trajectory_dim=3, condition_dim=1606, hidden_dim=256)
    
    # 大型模型
    large_model = VITA_Large(
        trajectory_dim=3, 
        condition_dim=1606, 
        hidden_dim=512,  # 4倍隐藏维度
        num_layers=6,     # 8层Transformer
        num_heads=8      # 16头注意力
    )
    
    original_params = count_parameters(original_model)
    large_params = count_parameters(large_model)
    
    print(f"原始模型参数量: {original_params:,}")
    print(f"大型模型参数量: {large_params:,}")
    print(f"参数量增加倍数: {large_params/original_params:.2f}x")




































