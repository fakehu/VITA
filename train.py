
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import os
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




def train_VITA_Large_VAE(model, train_loader, val_loader=None, epochs=100, lr=1e-4, device='cuda', save_dir='results'):
    """
    训练 VITA_Large_VAE 模型，标标准准的VITA模型
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 初始化损失记录
    train_losses = []
    train_fm_losses = []
    train_fld_losses = []
    train_ae_losses = []
    val_losses = [] if val_loader is not None else None
    learning_rates = []
    
    # 记录最佳验证损失（如果有验证集）
    best_val_loss = float('inf') if val_loader is not None else None
    best_epoch = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        epoch_fm_loss = 0
        epoch_fld_loss = 0
        epoch_ae_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in train_pbar:
            # 获取数据
            trajectories = batch['trajectory'].to(device)  # [batch_size, seq_len, 3]

            conditions = batch['condition'].to(device)     # [batch_size, condition_dim]
            
            batch_size = trajectories.shape[0]
            
            # 为 VITA_Large_VAE 准备数据
            # 随机时间步 t ~ U[0,1]
            t = torch.rand(batch_size, device=device)
            
            # 梯度置零
            optimizer.zero_grad()
            # print('准备前向推理')
            # 模型前向推理
            predicted_vector_field, \
            true_vector_field, \
            encoded_action_head, \
            encoded_action, \
            predicted_action_from_encoded_action, \
            predicted_action_from_encoded_action_head = model(conditions, t, trajectories)  
            # print('完成前向推理')
            # print('准备计算损失')
            # 计算损失
            lambda_fm = 0.33
            lambda_fld = 0.33
            lambda_ae = 0.33

            # 使用均方误差损失
            loss_fm = torch.nn.functional.mse_loss(predicted_vector_field, true_vector_field)
            loss_fld = torch.nn.functional.mse_loss(trajectories, predicted_action_from_encoded_action_head)
            loss_ae = torch.nn.functional.mse_loss(trajectories, predicted_action_from_encoded_action)

            loss = lambda_fm * loss_fm + lambda_fld * loss_fld + lambda_ae * loss_ae
            # print('完成损失计算')
            # print('准备反向传播')
            # 反向传播
            loss.backward()
            # print('完成反向传播')
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            # 累计损失
            epoch_loss += loss.item()
            epoch_fm_loss += loss_fm.item()
            epoch_fld_loss += loss_fld.item()
            epoch_ae_loss += loss_ae.item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'fm_loss': f'{loss_fm.item():.4f}',
                'fld_loss': f'{loss_fld.item():.4f}', 
                'ae_loss': f'{loss_ae.item():.4f}'
            })
                        
        # 计算平均损失
        avg_train_loss = epoch_loss / len(train_loader)
        avg_fm_loss = epoch_fm_loss / len(train_loader)
        avg_fld_loss = epoch_fld_loss / len(train_loader)
        avg_ae_loss = epoch_ae_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        train_fm_losses.append(avg_fm_loss)
        train_fld_losses.append(avg_fld_loss)
        train_ae_losses.append(avg_ae_loss)
        
        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    trajectories = batch['trajectory'].to(device)
                    conditions = batch['condition'].to(device)
                    batch_size = trajectories.shape[0]
                    t = torch.rand(batch_size, device=device)
                    
                    predicted_vector_field, true_vector_field, encoded_action_head, encoded_action, predicted_action_from_encoded_action, predicted_action_from_encoded_action_head = model(conditions, t, trajectories)
                    
                    loss_fm = torch.nn.functional.mse_loss(predicted_vector_field, true_vector_field)
                    loss_fld = torch.nn.functional.mse_loss(trajectories, predicted_action_from_encoded_action_head)
                    loss_ae = torch.nn.functional.mse_loss(trajectories, predicted_action_from_encoded_action)
                    loss = lambda_fm * loss_fm + lambda_fld * loss_fld + lambda_ae * loss_ae
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                }, save_dir, is_best=True)
                print(f'New best model saved with val_loss: {best_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')
        
        # 学习率调度
        scheduler.step()
        
        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if val_loader is not None else None,
                'best_val_loss': best_val_loss,
            }, save_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')
        
        # # 每10个epoch保存一次loss曲线
        # if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        #     plot_loss_curves(train_losses, train_fm_losses, train_fld_losses, train_ae_losses, 
        #                    val_losses, learning_rates, save_dir, epoch + 1)
    
    # 训练结束后保存最终的checkpoint和loss曲线
    save_checkpoint({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss if val_loader is not None else None,
        'best_val_loss': best_val_loss,
    }, save_dir, filename='checkpoint_final.pth')
    
    # 保存训练历史
    save_training_history({
        'train_losses': train_losses,
        'train_fm_losses': train_fm_losses,
        'train_fld_losses': train_fld_losses,
        'train_ae_losses': train_ae_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }, save_dir)
    
    # 绘制最终的loss曲线
    plot_loss_curves(train_losses, train_fm_losses, train_fld_losses, train_ae_losses, 
                   val_losses, learning_rates, save_dir, epochs, final=True)
    
    print(f"Training completed! Best model from epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_fm_losses': train_fm_losses,
        'train_fld_losses': train_fld_losses,
        'train_ae_losses': train_ae_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }

def save_checkpoint(state, save_dir, filename=None, is_best=False):
    """
    保存checkpoint
    """
    if is_best:
        filepath = os.path.join(save_dir, './checkpoints/checkpoint_best.pth')
    elif filename:
        filepath = os.path.join(save_dir, f"./checkpoints/{filename}")
    else:
        filepath = os.path.join(save_dir, './checkpoints/checkpoint.pth')
    
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def save_training_history(history, save_dir):
    """
    保存训练历史
    """
    history_path = os.path.join(save_dir, 'training_history.pth')
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")

def plot_loss_curves(train_losses, train_fm_losses, train_fld_losses, train_ae_losses, 
                    val_losses, learning_rates, save_dir, current_epoch, final=False):
    """
    绘制损失曲线和学习率曲线
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制总损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制各个损失分量
    ax2.plot(epochs, train_fm_losses, 'g-', label='FM Loss', linewidth=2)
    ax2.plot(epochs, train_fld_losses, 'orange', label='FLD Loss', linewidth=2)
    ax2.plot(epochs, train_ae_losses, 'purple', label='AE Loss', linewidth=2)
    ax2.set_title('Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制学习率曲线
    ax3.plot(epochs, learning_rates, 'm-', label='Learning Rate', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 绘制对数尺度的损失曲线
    ax4.semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses is not None:
        ax4.semilogy(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax4.set_title('Loss (Log Scale)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if final:
        filename = f'loss_curves_final.png'
    else:
        filename = f'loss_curves_epoch_{current_epoch}.png'
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_path}")
