import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime

from train import train_model
from evaluate import evaluate_model

def set_seed(seed=42):
    """
    设置随机种子，保证实验可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_train_config(args):
    """
    获取训练配置
    
    Args:
        args: 命令行参数
        
    Returns:
        dict: 训练配置
    """
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_classes': 2,  # 背景 + 物品
        'pretrained': True,
        'use_focal_loss': True,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'scheduler': args.scheduler,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'patience': args.patience,
        'min_lr': args.min_lr,
        'num_epochs': args.num_epochs,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'early_stopping_patience': args.early_stopping_patience,
        'checkpoint_interval': args.checkpoint_interval,
        'checkpoint_path': args.resume_from
    }
    
    return config

def get_eval_config(args):
    """
    获取评估配置
    
    Args:
        args: 命令行参数
        
    Returns:
        dict: 评估配置
    """
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_classes': 2,  # 背景 + 物品
        'device': torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'),
        'confidence_threshold': args.confidence_threshold,
        'iou_threshold': args.iou_threshold,
        'output_dir': args.output_dir,
        'max_visualization_images': args.max_visualization_images
    }
    
    return config

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='SKU110K目标检测')
    
    # 通用参数
    parser.add_argument('--data_dir', type=str, default='SKU110K_fixed', help='数据集目录')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'train_eval'], 
                        help='运行模式')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    
    # 训练参数
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'sgd', 'adamw'], help='优化器')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量（仅用于SGD）')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['step', 'cosine', 'reduce', 'none'], help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=10, help='学习率衰减步长（仅用于StepLR）')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--patience', type=int, default=3, help='学习率减少耐心值（仅用于ReduceLROnPlateau）')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='检查点保存间隔')
    parser.add_argument('--resume_from', type=str, default=None, help='从检查点恢复训练')
    
    # 评估参数
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--max_visualization_images', type=int, default=20, help='最大可视化图像数量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式执行操作
    if args.mode == 'train' or args.mode == 'train_eval':
        # 训练模型
        print('=== 开始训练 ===')
        train_config = get_train_config(args)
        model, history = train_model(args.data_dir, train_config)
        
        # 如果要训练后评估
        if args.mode == 'train_eval':
            print('=== 开始评估 ===')
            eval_config = get_eval_config(args)
            metrics = evaluate_model(args.data_dir, args.model_path, eval_config)
    
    elif args.mode == 'eval':
        # 评估模型
        print('=== 开始评估 ===')
        eval_config = get_eval_config(args)
        metrics = evaluate_model(args.data_dir, args.model_path, eval_config)

if __name__ == '__main__':
    # 记录开始时间
    start_time = datetime.now()
    print(f'开始运行时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 运行主函数
    main()
    
    # 记录结束时间
    end_time = datetime.now()
    print(f'结束运行时间: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'总运行时间: {end_time - start_time}')