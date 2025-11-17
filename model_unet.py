import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=None, cond_embed_dim=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # 时间步和条件信息的融合
        self.has_condition = time_embed_dim is not None or cond_embed_dim is not None
        if self.has_condition:
            total_embed_dim = 0
            if time_embed_dim is not None:
                total_embed_dim += time_embed_dim
            if cond_embed_dim is not None:
                total_embed_dim += cond_embed_dim
                
            self.condition_proj = nn.Linear(total_embed_dim, out_channels)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, time_embed=None, cond_embed=None):
        # 第一个卷积+归一化+激活
        x = self.conv1(x)
        if self.has_condition and (time_embed is not None or cond_embed is not None):
            # 融合条件信息
            if time_embed is not None and cond_embed is not None:
                combined_embed = torch.cat([time_embed, cond_embed], dim=-1)
            elif time_embed is not None:
                combined_embed = time_embed
            else:
                combined_embed = cond_embed
                
            # 将条件信息投影并加到特征上
            condition_bias = self.condition_proj(combined_embed).unsqueeze(-1)
            x = x + condition_bias
            
        x = self.norm1(x)
        x = self.activation(x)
        
        # 第二个卷积+归一化+激活
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, trajectory_dim=3, condition_dim=1606, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

        # 时间步编码器
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        # U-Net 编码器 (下采样)
        self.enc1 = UNetBlock(trajectory_dim, hidden_dim//2, hidden_dim, hidden_dim)
        self.enc2 = UNetBlock(hidden_dim//2, hidden_dim, hidden_dim, hidden_dim)
        self.enc3 = UNetBlock(hidden_dim, 2*hidden_dim, hidden_dim, hidden_dim)
        self.enc4 = UNetBlock(2*hidden_dim, 4*hidden_dim, hidden_dim, hidden_dim)
        
        # 下采样 - 使用池化而不是卷积以获得更好的尺寸控制
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)
        
        # 瓶颈层
        self.bottleneck = UNetBlock(4*hidden_dim, 4*hidden_dim, hidden_dim, hidden_dim)
        
        # 上采样 - 使用转置卷积
        self.up1 = nn.ConvTranspose1d(4*hidden_dim, 2*hidden_dim, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose1d(2*hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=2, stride=2)
        
        # U-Net 解码器 (上采样)
        self.dec1 = UNetBlock(4*hidden_dim, 2*hidden_dim, hidden_dim, hidden_dim)  # 4*hidden_dim = 2*hidden_dim(上采样) + 2*hidden_dim(跳跃连接)
        self.dec2 = UNetBlock(2*hidden_dim, hidden_dim, hidden_dim, hidden_dim)    # 2*hidden_dim = hidden_dim(上采样) + hidden_dim(跳跃连接)
        self.dec3 = UNetBlock(hidden_dim, hidden_dim//2, hidden_dim, hidden_dim)   # hidden_dim = hidden_dim//2(上采样) + hidden_dim//2(跳跃连接)
        
        # 输出层
        self.output_conv = nn.Conv1d(hidden_dim//2, trajectory_dim, 3, padding=1)

    def forward(self, noisy_trajectory, condition, timesteps):
        """
        noisy_trajectory: [batch_size, seq_len, trajectory_dim]
        condition: [batch_size, condition_dim]
        timesteps: [batch_size]
        """
        batch_size, seq_len, _ = noisy_trajectory.shape
        
        # 转置输入以适应1D卷积 [batch_size, trajectory_dim, seq_len]
        x = noisy_trajectory.transpose(1, 2)
        
        # 编码条件信息
        cond_embed = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        
        # 编码时间步
        time_embed = self.time_embedding(timesteps.unsqueeze(1).float())  # [batch_size, hidden_dim]
        
        # 编码器路径
        e1 = self.enc1(x, time_embed, cond_embed)  # [batch_size, hidden_dim//2, seq_len]
        e2 = self.enc2(self.pool1(e1), time_embed, cond_embed)  # [batch_size, hidden_dim, seq_len//2]
        e3 = self.enc3(self.pool2(e2), time_embed, cond_embed)  # [batch_size, 2*hidden_dim, seq_len//4]
        e4 = self.enc4(self.pool3(e3), time_embed, cond_embed)  # [batch_size, 4*hidden_dim, seq_len//8]
        
        # 瓶颈层
        bottleneck = self.bottleneck(e4, time_embed, cond_embed)  # [batch_size, 4*hidden_dim, seq_len//8]
        
        # 解码器路径 (带跳跃连接)
        # 上采样并调整尺寸以匹配跳跃连接
        d1_up = self.up1(bottleneck)
        # 确保尺寸匹配
        if d1_up.shape[2] != e3.shape[2]:
            d1_up = F.interpolate(d1_up, size=e3.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1_up, e3], dim=1), time_embed, cond_embed)  # [batch_size, 2*hidden_dim, seq_len//4]
        
        d2_up = self.up2(d1)
        if d2_up.shape[2] != e2.shape[2]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1), time_embed, cond_embed)  # [batch_size, hidden_dim, seq_len//2]
        
        d3_up = self.up3(d2)
        if d3_up.shape[2] != e1.shape[2]:
            d3_up = F.interpolate(d3_up, size=e1.shape[2], mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([d3_up, e1], dim=1), time_embed, cond_embed)  # [batch_size, hidden_dim//2, seq_len]
        
        # 输出
        output = self.output_conv(d3)  # [batch_size, trajectory_dim, seq_len]
        output = output.transpose(1, 2)  # [batch_size, seq_len, trajectory_dim]
        
        return output

# 使用U-Net的VITA模型
class VITA_UNet(nn.Module):
    def __init__(self, trajectory_dim=3, condition_dim=1606, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        self.unet = UNet(trajectory_dim, condition_dim, hidden_dim, num_timesteps)
    
    def forward(self, noisy_trajectory, condition, timesteps):
        return self.unet(noisy_trajectory, condition, timesteps)

# Flow Matching 训练函数保持不变
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

# 使用示例
if __name__ == "__main__":
    # 测试模型
    batch_size, seq_len, trajectory_dim = 32, 100, 3
    condition_dim = 1606
    
    model = VITA_UNet(trajectory_dim, condition_dim)
    
    # 模拟输入
    noisy_trajectory = torch.randn(batch_size, seq_len, trajectory_dim)
    condition = torch.randn(batch_size, condition_dim)
    timesteps = torch.rand(batch_size)
    
    # 前向传播
    output = model(noisy_trajectory, condition, timesteps)
    print(f"Input shape: {noisy_trajectory.shape}")
    print(f"Output shape: {output.shape}")