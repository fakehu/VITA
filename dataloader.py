
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TrajectoryDataset(Dataset):
    def __init__(self, data_list, obs_len=1600):
        self.data_list = data_list
        self.obs_len = obs_len
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_tuple = self.data_list[idx]

        diffusion_data = prepare_diffusion_data(data_tuple, self.obs_len)
        
        return {
            'trajectory': diffusion_data['trajectory'],  # [seq_len, 3]
            'condition': diffusion_data['condition']     # [condition_dim]
        }


# 创建数据加载器
def create_data_loader(data_tuples, batch_size=32, obs_len=1600):
    dataset = TrajectoryDataset(data_tuples, obs_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def prepare_diffusion_data(data_tuple, obs_len=1600, num_sample=32):
    """
    准备适合Diffusion训练的轨迹数据
    """
    data_dict = dict(data_tuple)

    trajectory_normalized, condition = normalize(data_dict, obs_len, num_sample)

    
    return {
        'trajectory': torch.FloatTensor(trajectory_normalized),  # [seq_len, 3]
        'condition': torch.FloatTensor(condition),  # 条件信息
    }

def sample_trajectory_fixed_steps(trajectory, num_samples=32):
    """
    平均间隔抽num_samples个点，保证头尾都在
    """
    # 确保轨迹是numpy数组
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)

    n_points = len(trajectory)
    
    # 生成等间隔的索引（自动包含头尾）
    indices = np.linspace(0, n_points - 1, num_samples, dtype=int)
    
    # 直接返回抽样后的轨迹
    return trajectory[indices]


def normalize(data_dict, obs_len=1600, num_samples=32):
    #原始数据
    start = np.array(data_dict['start'][:3])  # [x, y, angle]
    target = np.array(data_dict['target'][:3])  # [x, y, angle]
    raw_trajectory = np.array([point[:3] for point in data_dict['traj']])
    obstacles = np.array(data_dict['obs'])
    
    #trajectory抽样
    if len(raw_trajectory) != num_samples:
        trajectory = sample_trajectory_fixed_steps(raw_trajectory, num_samples)
    else:
        trajectory = raw_trajectory.copy()

    #中心化
    center = np.mean([start[:2], target[:2]], axis=0)
    trajectory[:, :2] = trajectory[:, :2] - center
    start[:2] = start[:2] - center
    target[:2] = target[:2] - center

    if len(obstacles) > 0:
        obstacles_normalized = obstacles.copy()
        obstacles_normalized[:, [0, 2]] = obstacles_normalized[:, [0, 2]] - center[0]
        obstacles_normalized[:, [1, 3]] = obstacles_normalized[:, [1, 3]] - center[1]
        obstacles_flat = obstacles_normalized.flatten()
        # 如果障碍物信息太长，进行截断
        max_obs_length = obs_len
        if len(obstacles_flat) > max_obs_length:
            obstacles_flat = obstacles_flat[:max_obs_length]
        elif len(obstacles_flat) < max_obs_length:
            obstacles_flat = np.pad(obstacles_flat, (0, max_obs_length - len(obstacles_flat)))
    else:
        obstacles_flat = np.zeros(obs_len)

    # 角度归一化到 [-1, 1]
    trajectory[:, 2] = trajectory[:, 2] / np.pi
    start[2] = start[2] / np.pi
    target[2] = target[2] / np.pi

    condition = np.concatenate([start, target, obstacles_flat])

    return trajectory, condition

def plot_trajectory_data_ori(loader, num_samples=5):
    """
    可视化训练加载器中的轨迹数据
    
    Args:
        loader: 数据加载器
        num_samples: 要可视化的样本数量
    """
    # 获取一个批次的数据
    batch = next(iter(loader))
    trajectories = batch['trajectory']  # [batch_size, seq_len, 3]
    conditions = batch['condition']     # [batch_size, condition_dim]
    
    # 限制样本数量
    num_samples = min(num_samples, trajectories.shape[0])
    
    # 创建子图
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        trajectory = trajectories[i].numpy()  # [seq_len, 3]
        condition = conditions[i].numpy()     # [condition_dim]
        
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

if __name__ == "__main__":
    import sys
    sys.path.append(r"E:\utils")
    from utils import load_jsonl

    #0_dataload
    data_path = r"./data.jsonl"
    ori_data = load_jsonl(data_path)
    train_loader = create_data_loader(ori_data, batch_size=8, obs_len=1600)

    # 可视化数据
    print("Visualizing training data...")
    plot_trajectory_data_ori(train_loader, num_samples=3)
