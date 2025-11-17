import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import  create_data_loader, plot_trajectory_data_ori
from model import VITA, VITA_Large, VITA_Large_VAE
from model_unet import VITA_UNet
from train import train_flow_matching, train_VITA_Large_VAE
from inference import sample_flow_matching, plot_trajectory_data
import sys
sys.path.append(r".\utils")
from utils import load_jsonl
import matplotlib.pyplot as plt
import matplotlib.patches as patches



#0_dataload
train_data_path = r"./train_data.jsonl"
val_data_path = r"./val_data.jsonl"

train_data = load_jsonl(train_data_path)
val_data = load_jsonl(val_data_path)
 
train_loader = create_data_loader(train_data, batch_size=32, obs_len=1600)
val_loader = create_data_loader(val_data, batch_size=32, obs_len=1600)

#1_model_init

model = VITA_Large_VAE(
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

train_VITA_Large_VAE(model=model,
                     train_loader=train_data,
                     val_loader=val_data,
                     epochs=1000,
                     lr=1e-4,
                     device='cuda',
                     save_dir='results'
                     )
















