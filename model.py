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


class VITA_Large_VAE(nn.Module):
    def __init__(self, trajectory_dim=3, condition_dim=1606, hidden_dim=1024, num_timesteps=1000, 
                 num_layers=8, num_heads=16, dropout=0.1):
        super().__init__()
        
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps

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
            nn.Linear(hidden_dim * 2, hidden_dim),
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


        # condtion编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
        )

        #flow_matching
        self.flow_matching = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trajectory_dim),

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

    def forward(self, condition, timesteps, ground_truth_trajectory, inference_mode=False):
        """
        condition: [batch_size, condition_dim]
        timesteps: [batch_size] - 在flow matching中是时间t ∈ [0,1]
        """
        if inference_mode:
            seq_len = 32
        else:
            batch_size, seq_len, _ = ground_truth_trajectory.shape

        # 1.condition处理
        # # 1.1 编码轨迹
        # traj_embed = self.trajectory_encoder(noisy_trajectory)  # [batch_size, seq_len, hidden_dim]
        
        # 1.1 编码条件信息
        cond_embed = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        cond_embed = cond_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 1.2 编码时间步
        time_embed = self.time_embedding(timesteps.unsqueeze(1).float())  # [batch_size, hidden_dim]
        time_embed = time_embed.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 1.3 合并所有特征
        combined = torch.cat([cond_embed, time_embed], dim=-1)  # [batch_size, seq_len, hidden_dim * 2]
        fused_features = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_dim]

        # 1.4 添加位置编码
        fused_features = self.positional_encoding(fused_features)

        # 1.5 Transformer处理 (使用残差连接)
        transformer_input = self.layer_norm1(fused_features)
        encoded_condition = self.transformer(transformer_input)  # [batch_size, seq_len, hidden_dim]
        encoded_condition = self.layer_norm2(encoded_condition + fused_features)  # 残差连接

        # 2. flow_matching 处理 - 完整迭代过程
        current_state = encoded_condition.clone() # [batch_size, seq_len, hidden_dim]
        # 我还是需要预测的向量来计算 L_fm 对吧，呜呜呜
        predicted_vector_field = self.flow_matching(current_state)

        # 为每个样本计算需要迭代的步数
        start_steps = (timesteps * self.num_timesteps).long()  # [batch_size]
        dt = 1.0 / self.num_timesteps

        # 创建迭代掩码
        max_steps = start_steps.max()
        steps_range = torch.arange(max_steps, device=condition.device)  # [max_steps]

        # 为每个样本创建步数掩码 [batch_size, max_steps]
        step_mask = steps_range.unsqueeze(0) >= start_steps.unsqueeze(1)  # [batch_size, max_steps]


        # 批量处理所有迭代步
        for step_idx in range(max_steps):
            # 获取需要当前步的样本掩码
            current_mask = step_mask[:, step_idx]  # [batch_size]
            
            if current_mask.any():
                t = 1.0 - step_idx * dt
                
                # 预测速度场
                vector_field = self.flow_matching(current_state)
                
                # 只更新需要当前步的样本
                update_mask = current_mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                current_state = torch.where(update_mask, 
                                        current_state - vector_field * dt,
                                        current_state)
        



        # 生成 encoded_action_head 也就是动作编码的估计值
        encoded_action_head = current_state


        # 解码为动作
        predicted_action_from_encoded_action_head = self.action_decoder(encoded_action_head)

        # 3. action_VAE 处理
        if not inference_mode:
            encoded_action = None
            true_vector_field = None
            predicted_action_from_encoded_action = None
        else:
            encoded_action = self.action_encoder(ground_truth_trajectory)
            true_vector_field = encoded_action - current_state
            predicted_action_from_encoded_action = self.action_decoder(encoded_action)
        
        return  predicted_vector_field, \
                true_vector_field, \
                encoded_action_head, \
                encoded_action, \
                predicted_action_from_encoded_action, \
                predicted_action_from_encoded_action_head


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




































