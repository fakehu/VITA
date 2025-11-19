import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import  create_data_loader, plot_trajectory_data_ori
from model import VITA, VITA_Large, VITA_Large_VAE
from model_unet import VITA_UNet
from train import train_flow_matching, train_VITA_Large_VAE
from inference import sample_flow_matching, plot_trajectory_data, plot_trajectory_comparison
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
 
train_loader = create_data_loader(train_data, batch_size=16, obs_len=1600)
val_loader = create_data_loader(val_data, batch_size=16, obs_len=1600)

#1_model_init

model = VITA_Large_VAE(
        trajectory_dim=3, 
        condition_dim=1606, 
        hidden_dim=512,  # 4倍隐藏维度
        num_layers=6,     # 8层Transformer
        num_heads=8      # 16头注意力
)
model = model.to('cuda')
# #1_1模型参数量
# def count_parameters(model):
#     """计算模型总参数量"""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     return total_params, trainable_params

# total_params, trainable_params = count_parameters(model)
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")
# input("按回车键继续...")

# #2_model_train

# train_VITA_Large_VAE(model=model,
#                      train_loader=train_loader,
#                      val_loader=val_loader,
#                      epochs=1000,
#                      lr=1e-4,
#                      device='cuda',
#                      save_dir='results'
#                      )




#3_inference

checkpoint = torch.load('./results/checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])  # 加载权重
model.eval()

batch = next(iter(val_loader))
# sample_condition = batch['condition'][:4].cuda()
# sample_trajectory = batch['trajectory'][:4].cuda()

sample_condition = batch['condition'][4:8].cuda()
sample_trajectory = batch['trajectory'][4:8].cuda()

batch_size = sample_trajectory.size(0)
t = torch.rand(batch_size, device='cuda')

predicted_vector_field, \
true_vector_field, \
encoded_action_head, \
encoded_action, \
predicted_action_from_encoded_action, \
predicted_action_from_encoded_action_head = model(sample_condition, t, sample_trajectory, inference_mode=True)  

# plot_trajectory_data(predicted_action_from_encoded_action_head, sample_condition, batch_size)
# plot_trajectory_data(sample_trajectory, sample_condition, batch_size)
plot_trajectory_comparison(sample_trajectory, predicted_action_from_encoded_action_head, sample_condition, batch_size)
input('按任意键继续......')












