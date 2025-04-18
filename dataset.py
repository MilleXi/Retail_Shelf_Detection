import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SKU110KDataset(Dataset):
    """
    SKU110K数据集加载器
    """
    def __init__(self, root_dir, csv_file, transform=None, is_train=True):
        """
        初始化函数
        
        Args:
            root_dir (str): 数据集根目录
            csv_file (str): 标注文件路径
            transform (callable, optional): 数据转换
            is_train (bool): 是否为训练模式
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotations = pd.read_csv(os.path.join(root_dir, 'annotations', csv_file), header=None)
        self.transform = transform
        self.is_train = is_train
        
        # 整理数据集格式
        self.image_names = []
        self.all_boxes = []
        
        current_image = None
        current_boxes = []
        
        for _, row in self.annotations.iterrows():
            image_name, x1, y1, x2, y2, class_name, img_width, img_height = row
            
            if current_image != image_name:
                if current_image is not None:
                    self.image_names.append(current_image)
                    self.all_boxes.append(current_boxes)
                current_image = image_name
                current_boxes = []
            
            # 转换为相对坐标 (归一化)
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            img_width, img_height = float(img_width), float(img_height)
            
            # 确保边界框在图像内部
            x1, x2 = max(0, x1), min(img_width, x2)
            y1, y2 = max(0, y1), min(img_height, y2)
            
            # 计算相对坐标 (x_center, y_center, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # 添加类别标签 (SKU110K只有一个类别，标签为0)
            current_boxes.append([0, x_center, y_center, width, height])
        
        # 添加最后一个图像的边界框
        if current_image is not None:
            self.image_names.append(current_image)
            self.all_boxes.append(current_boxes)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取原始图像尺寸
        height, width = image.shape[:2]
        
        # 获取边界框
        boxes = np.array(self.all_boxes[idx])
        
        # 如果没有边界框，创建一个空数组
        if len(boxes) == 0:
            boxes = np.zeros((0, 5))
        
        # 转换回原始坐标 (x1, y1, x2, y2) 用于Albumentations
        if len(boxes) > 0:
            new_boxes = []
            for box in boxes:
                class_id, x_c, y_c, w, h = box
                x1 = (x_c - w/2) * width
                y1 = (y_c - h/2) * height
                x2 = (x_c + w/2) * width
                y2 = (y_c + h/2) * height
                new_boxes.append([x1, y1, x2, y2, class_id])
            boxes = np.array(new_boxes)
        
        # 应用数据增强
        if self.transform:
            # 创建Albumentations变换
            transformed = self.transform(image=image, bboxes=boxes[:, :4], category_ids=boxes[:, 4])
            image = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                bboxes = np.array(transformed['bboxes'])
                category_ids = np.array(transformed['category_ids'])
                
                # 转换回YOLOv5格式 (class, x_center, y_center, width, height)
                h, w = 640, 640  # 调整后的图像尺寸
                new_boxes = []
                for i, box in enumerate(bboxes):
                    x1, y1, x2, y2 = box
                    x_c = (x1 + x2) / (2 * w)
                    y_c = (y1 + y2) / (2 * h)
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    new_boxes.append([category_ids[i], x_c, y_c, width, height])
                boxes = np.array(new_boxes)
            else:
                boxes = np.zeros((0, 5))
        
        # 创建目标字典
        targets = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.zeros(len(boxes), dtype=torch.int64)  # 所有标签都是0
        }
        
        return image, targets, img_name

def get_train_transform():
    """
    获取训练数据增强
    
    Returns:
        A.Compose: Albumentations转换
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.5)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
    )

def get_valid_transform():
    """
    获取验证数据增强
    
    Returns:
        A.Compose: Albumentations转换
    """
    return A.Compose(
        [
            A.Resize(height=640, width=640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
    )

def collate_fn(batch):
    """
    数据批次整理函数
    
    Args:
        batch: 批次数据
        
    Returns:
        tuple: 整理后的批次数据
    """
    images = []
    targets = []
    img_names = []
    
    for image, target, img_name in batch:
        images.append(image)
        targets.append(target)
        img_names.append(img_name)
        
    return torch.stack(images, 0), targets, img_names

def get_dataloaders(root_dir, batch_size=8, num_workers=4):
    """
    获取数据加载器
    
    Args:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 工作进程数
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = SKU110KDataset(
        root_dir=root_dir,
        csv_file='annotations_train.csv',
        transform=get_train_transform(),
        is_train=True
    )
    
    val_dataset = SKU110KDataset(
        root_dir=root_dir,
        csv_file='annotations_val.csv',
        transform=get_valid_transform(),
        is_train=False
    )
    
    test_dataset = SKU110KDataset(
        root_dir=root_dir,
        csv_file='annotations_test.csv',
        transform=get_valid_transform(),
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader