import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 采样函数示例
def sample_flow_matching(model, condition, num_samples, seq_len, trajectory_dim, device, steps=100):
    """
    使用欧拉方法从flow matching模型采样
    """
    model.eval()
    
    # 从标准正态分布初始化
    x = torch.randn(num_samples, seq_len, trajectory_dim).to(device)

    condition = condition.to(device)
    
    # 离散时间步
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([i * dt] * num_samples).to(device)
            
            # 预测向量场
            v = model(x, condition, t)
            
            # 欧拉步
            x = x + dt * v
    
    return x



# #可视化
# def plot_trajectory_data(trajectories, conditions, num_samples=5):
    
#     # # 获取一个批次的数据
#     # batch = next(iter(loader))
#     # trajectories = batch['trajectory']  # [batch_size, seq_len, 3]
#     # conditions = batch['condition']     # [batch_size, condition_dim]



#     # 限制样本数量
#     num_samples = min(num_samples, trajectories.shape[0])
    
#     # 创建子图
#     fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
#     if num_samples == 1:
#         axes = [axes]
    
#     for i in range(num_samples):
#         ax = axes[i]
#         trajectory = trajectories[i].numpy()  # [seq_len, 3]
#         condition = conditions[i].numpy()     # [condition_dim]
        
#         # 从条件中提取起点、目标点和障碍物
#         start = condition[:3]      # [x, y, angle]
#         target = condition[3:6]    # [x, y, angle]
#         obstacles_flat = condition[6:]  # 展平的障碍物
        
#         # 绘制轨迹
#         ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
#         ax.scatter(trajectory[:, 0], trajectory[:, 1], c=range(len(trajectory)), 
#                   cmap='viridis', s=30, alpha=0.6)
        
#         # 绘制起点和目标点
#         ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
#         ax.plot(target[0], target[1], 'ro', markersize=10, label='Target')
        
#         # 绘制起点和目标点的方向
#         arrow_length = 0.5
#         start_dx = arrow_length * np.cos(start[2] * np.pi)
#         start_dy = arrow_length * np.sin(start[2] * np.pi)
#         target_dx = arrow_length * np.cos(target[2] * np.pi)
#         target_dy = arrow_length * np.sin(target[2] * np.pi)
        
#         ax.arrow(start[0], start[1], start_dx, start_dy, head_width=0.1, 
#                 head_length=0.1, fc='g', ec='g')
#         ax.arrow(target[0], target[1], target_dx, target_dy, head_width=0.1, 
#                 head_length=0.1, fc='r', ec='r')
        
#         # 绘制障碍物
#         obstacles = obstacles_flat.reshape(-1, 4)  # 重塑为 [n_obs, 4]
#         for obs in obstacles:
#             if np.any(obs != 0):  # 只绘制非零障碍物
#                 x1, y1, x2, y2 = obs
#                 width = x2 - x1
#                 height = y2 - y1
#                 rect = patches.Rectangle((x1, y1), width, height, 
#                                        linewidth=1, edgecolor='black', 
#                                        facecolor='gray', alpha=0.5)
#                 ax.add_patch(rect)
        
#         # 设置图形属性
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_title(f'Sample {i+1}')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')
#         ax.legend()
    
#     plt.tight_layout()
#     plt.show(block=False)



def plot_trajectory_data(trajectories, conditions, num_samples=5):
    """
    可视化轨迹数据 - 修复GPU张量转换问题
    """
    # 限制样本数量
    num_samples = min(num_samples, trajectories.shape[0])
    
    # 将张量从GPU移动到CPU并转换为numpy
    if torch.is_tensor(trajectories):
        trajectories = trajectories.detach().cpu().numpy()
    if torch.is_tensor(conditions):
        conditions = conditions.detach().cpu().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        trajectory = trajectories[i]  # [seq_len, 3] - 已经是numpy数组
        condition = conditions[i]     # [condition_dim] - 已经是numpy数组
        
        # 从条件中提取起点、目标点和障碍物
        start = condition[:3]      # [x, y, angle]
        target = condition[3:6]    # [x, y, angle]
        obstacles_flat = condition[6:]  # 展平的障碍物
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=range(len(trajectory)), 
                  cmap='viridis', s=30, alpha=0.6)
        
        # 绘制起点和目标点
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(target[0], target[1], 'ro', markersize=10, label='Target')
        
        # 绘制起点和目标点的方向
        arrow_length = 0.5
        start_dx = arrow_length * np.cos(start[2] * np.pi)
        start_dy = arrow_length * np.sin(start[2] * np.pi)
        target_dx = arrow_length * np.cos(target[2] * np.pi)
        target_dy = arrow_length * np.sin(target[2] * np.pi)
        
        ax.arrow(start[0], start[1], start_dx, start_dy, head_width=0.1, 
                head_length=0.1, fc='g', ec='g')
        ax.arrow(target[0], target[1], target_dx, target_dy, head_width=0.1, 
                head_length=0.1, fc='r', ec='r')
        
        # 绘制障碍物
        obstacles = obstacles_flat.reshape(-1, 4)  # 重塑为 [n_obs, 4]
        for obs in obstacles:
            if np.any(obs != 0):  # 只绘制非零障碍物
                x1, y1, x2, y2 = obs
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {i+1}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)





def plot_trajectory_comparison(ground_truth_traj, predicted_traj, conditions, num_samples=5):
    """
    可视化真实轨迹与预测轨迹的对照图 - 修复GPU张量转换问题
    """
    # 限制样本数量
    num_samples = min(num_samples, ground_truth_traj.shape[0])
    
    # 将张量从GPU移动到CPU并转换为numpy
    if torch.is_tensor(ground_truth_traj):
        ground_truth_traj = ground_truth_traj.detach().cpu().numpy()
    if torch.is_tensor(predicted_traj):
        predicted_traj = predicted_traj.detach().cpu().numpy()
    if torch.is_tensor(conditions):
        conditions = conditions.detach().cpu().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        gt_trajectory = ground_truth_traj[i]  # [seq_len, 3] - 真实轨迹
        pred_trajectory = predicted_traj[i]   # [seq_len, 3] - 预测轨迹
        condition = conditions[i]              # [condition_dim] - 条件信息
        
        # 从条件中提取起点、目标点和障碍物
        start = condition[:3]      # [x, y, angle]
        target = condition[3:6]    # [x, y, angle]
        obstacles_flat = condition[6:]  # 展平的障碍物
        
        # 绘制真实轨迹（蓝色）
        ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax.scatter(gt_trajectory[:, 0], gt_trajectory[:, 1], c=range(len(gt_trajectory)), 
                  cmap='Blues', s=40, alpha=0.6)
        
        # 绘制预测轨迹（红色）
        ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-', linewidth=2, label='Predicted', alpha=0.8, linestyle='--')
        ax.scatter(pred_trajectory[:, 0], pred_trajectory[:, 1], c=range(len(pred_trajectory)), 
                  cmap='Reds', s=30, alpha=0.6, marker='s')
        
        # 绘制起点和目标点
        ax.plot(start[0], start[1], 'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=1)
        ax.plot(target[0], target[1], 'mo', markersize=12, label='Target', markeredgecolor='black', markeredgewidth=1)
        
        # 绘制起点和目标点的方向
        arrow_length = 0.5
        start_dx = arrow_length * np.cos(start[2] * np.pi)
        start_dy = arrow_length * np.sin(start[2] * np.pi)
        target_dx = arrow_length * np.cos(target[2] * np.pi)
        target_dy = arrow_length * np.sin(target[2] * np.pi)
        
        ax.arrow(start[0], start[1], start_dx, start_dy, head_width=0.1, 
                head_length=0.1, fc='g', ec='g', alpha=0.7)
        ax.arrow(target[0], target[1], target_dx, target_dy, head_width=0.1, 
                head_length=0.1, fc='m', ec='m', alpha=0.7)
        
        # 绘制障碍物
        obstacles = obstacles_flat.reshape(-1, 4)  # 重塑为 [n_obs, 4]
        for obs in obstacles:
            if np.any(obs != 0):  # 只绘制非零障碍物
                x1, y1, x2, y2 = obs
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
        
        # 计算并显示误差
        mse = np.mean((gt_trajectory[:, :2] - pred_trajectory[:, :2]) ** 2)
        final_position_error = np.linalg.norm(gt_trajectory[-1, :2] - pred_trajectory[-1, :2])
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {i+1}\nMSE: {mse:.4f}, Final Error: {final_position_error:.4f}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 打印总体统计信息
    print(f"\n=== 轨迹预测性能统计 ===")
    print(f"样本数量: {num_samples}")
    print(f"平均MSE: {np.mean([np.mean((ground_truth_traj[i, :, :2] - predicted_traj[i, :, :2]) ** 2) for i in range(num_samples)]):.4f}")
    print(f"平均最终位置误差: {np.mean([np.linalg.norm(ground_truth_traj[i, -1, :2] - predicted_traj[i, -1, :2]) for i in range(num_samples)]):.4f}")