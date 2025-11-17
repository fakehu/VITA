import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import  create_data_loader, plot_trajectory_data_ori
from model import VITA, VITA_Large
from model_unet import VITA_UNet
from train import train_flow_matching
from inference import sample_flow_matching, plot_trajectory_data
import sys
sys.path.append(r".\utils")
from utils import load_jsonl
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#0_dataload
data_path = r"./data.jsonl"
ori_data = load_jsonl(data_path)

train_loader = create_data_loader(ori_data, batch_size=32, obs_len=1600)

#1_model_init

# model = VITA(
#     trajectory_dim=3,
#     condition_dim=1606,  # start(3) + target(3) + obstacles(1600)
#     hidden_dim=256
#     )

# model = VITA_UNet(trajectory_dim=3, condition_dim=1606, hidden_dim=256, num_timesteps=1000)

model = VITA_Large(
        trajectory_dim=3, 
        condition_dim=1606, 
        hidden_dim=512,  # 4倍隐藏维度
        num_layers=6,     # 8层Transformer
        num_heads=8      # 16头注意力
    )
#1_1模型参数量
def count_parameters(model):
    """计算模型总参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
input("按回车键继续...")

#2_model_train


train_losses, val_losses = train_flow_matching(model=model, 
                                               train_loader=train_loader, 
                                               epochs=1000, 
                                               device='cuda' if torch.cuda.is_available() else 'cpu')

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
if val_losses:
    plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show(block=False)

#3_inference
# 3.1生成轨迹
print("Sampling trajectories...")
batch = next(iter(train_loader))
sample_condition = batch['condition'][:1]
sample_trajectory = batch['trajectory'][:1]


sampled_trajectories = sample_flow_matching(
    model=model,
    condition=sample_condition,
    num_samples=1,
    seq_len=32,
    trajectory_dim=3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    steps=1000
)

print(f"Sampled trajectories shape: {sampled_trajectories.shape}")

# 3.2可视化结果
plot_trajectory_data_ori(train_loader)

plot_trajectory_data(trajectories=sample_trajectory, conditions=sample_condition)

plot_trajectory_data(trajectories=sampled_trajectories, conditions=sample_condition)

input("按回车键继续...")























