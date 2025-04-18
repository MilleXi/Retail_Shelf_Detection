import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from model import create_model

class Trainer:
    """
    模型训练器
    """
    def __init__(self, 
                model, 
                train_loader, 
                val_loader, 
                optimizer, 
                scheduler=None, 
                device=None,
                num_epochs=100,
                save_dir='checkpoints',
                log_dir='logs',
                early_stopping_patience=10,
                checkpoint_interval=5):
        """
        初始化函数
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            log_dir: 日志保存目录
            early_stopping_patience: 早停耐心值
            checkpoint_interval: 检查点保存间隔
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_interval = checkpoint_interval
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建TensorBoard日志记录器
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 初始化训练指标
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_one_epoch(self, epoch):
        """
        训练一个轮次
        
        Args:
            epoch: 当前轮次
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        epoch_loss = 0
        
        # 创建进度条
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        # 遍历批次
        for images, targets, _ in progress_bar:
            # 将图像和目标移动到设备
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 训练步骤
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新损失
            epoch_loss += losses.item()
            
            # 更新进度条
            progress_bar.set_postfix(loss=losses.item())
        
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        
        # 更新学习率调度器
        if self.scheduler:
            self.scheduler.step()
        
        # 更新历史记录
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # 记录TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """
        验证模型
        
        Args:
            epoch: 当前轮次
            
        Returns:
            float: 平均损失
        """
        self.model.eval()
        val_loss = 0
        
        # 创建进度条
        progress_bar = tqdm(self.val_loader, desc='Validation')
        
        # 遍历批次
        with torch.no_grad():
            for images, targets, _ in progress_bar:
                # 将图像和目标移动到设备
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 验证步骤
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 更新损失
                val_loss += losses.item()
                
                # 更新进度条
                progress_bar.set_postfix(loss=losses.item())
        
        # 计算平均损失
        avg_val_loss = val_loss / len(self.val_loader)
        
        # 更新历史记录
        self.history['val_loss'].append(avg_val_loss)
        
        # 记录TensorBoard
        self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def train(self):
        """
        训练模型
        
        Returns:
            dict: 训练历史记录
        """
        print(f'Training on {self.device}')
        print(f'Number of training samples: {len(self.train_loader.dataset)}')
        print(f'Number of validation samples: {len(self.val_loader.dataset)}')
        
        # 记录开始时间
        start_time = time.time()
        
        # 遍历轮次
        for epoch in range(self.num_epochs):
            # 训练
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 打印信息
            print(f'Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 保存检查点
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # 检查是否是最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f'New best model saved with validation loss: {val_loss:.4f}')
            else:
                self.epochs_without_improvement += 1
            
            # 早停检查
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time
        
        # 打印训练时间
        print(f'Training completed in {str(datetime.timedelta(seconds=int(training_time)))}')
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.history
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            val_loss: 验证损失
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            filepath = os.path.join(self.save_dir, 'best_model.pth')
        else:
            filepath = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            int: 加载的轮次
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['val_loss']
        
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()

def train_model(root_dir, config):
    """
    训练模型
    
    Args:
        root_dir: 数据集根目录
        config: 配置字典
    
    Returns:
        tuple: (训练后的模型, 训练历史记录)
    """
    # 加载数据
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    model = create_model(
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        use_focal_loss=config['use_focal_loss']
    )
    
    # 创建优化器
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 创建学习率调度器
    if config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
    elif config['scheduler'] == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['gamma'],
            patience=config['patience'],
            min_lr=config['min_lr']
        )
    else:
        scheduler = None
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir'],
        log_dir=config['log_dir'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_interval=config['checkpoint_interval']
    )
    
    # 如果提供了检查点路径，加载检查点
    if config.get('checkpoint_path'):
        start_epoch = trainer.load_checkpoint(config['checkpoint_path'])
        print(f'Loaded checkpoint from epoch {start_epoch+1}')
    
    # 训练模型
    history = trainer.train()
    
    return model, history

if __name__ == '__main__':
    # 训练配置
    config = {
        'batch_size': 8,
        'num_workers': 4,
        'num_classes': 2,  # 背景 + 物品
        'pretrained': True,
        'use_focal_loss': True,
        'optimizer': 'adamw',
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'momentum': 0.9,  # 仅用于SGD
        'scheduler': 'cosine',
        'step_size': 10,  # 仅用于StepLR
        'gamma': 0.1,  # 学习率衰减因子
        'patience': 3,  # 仅用于ReduceLROnPlateau
        'min_lr': 1e-6,
        'num_epochs': 50,
        'save_dir': 'checkpoints',
        'log_dir': 'logs',
        'early_stopping_patience': 10,
        'checkpoint_interval': 5,
        'checkpoint_path': None  # 如果需要从检查点恢复，提供路径
    }
    
    # 训练模型
    model, history = train_model('SKU110K_fixed', config)