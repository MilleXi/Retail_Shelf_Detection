import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import cv2
from torchvision.ops import box_iou, nms
import torch.nn.functional as F
import pandas as pd
import json

def xywh_to_xyxy(box):
    """
    将 [x_center, y_center, width, height] 格式转换为 [x1, y1, x2, y2] 格式
    
    Args:
        box: [x_center, y_center, width, height] 格式的边界框
        
    Returns:
        [x1, y1, x2, y2] 格式的边界框
    """
    x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]

def xyxy_to_xywh(box):
    """
    将 [x1, y1, x2, y2] 格式转换为 [x_center, y_center, width, height] 格式
    
    Args:
        box: [x1, y1, x2, y2] 格式的边界框
        
    Returns:
        [x_center, y_center, width, height] 格式的边界框
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [x_center, y_center, width, height]

def draw_boxes(image, boxes, scores=None, labels=None, color=(0, 255, 0), thickness=2, normalized=False):
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        boxes: 边界框列表，格式为 [x1, y1, x2, y2]
        scores: 分数列表，可选
        labels: 标签列表，可选
        color: 边界框颜色
        thickness: 线条粗细
        normalized: 坐标是否归一化
        
    Returns:
        绘制有边界框的图像
    """
    # 确保图像是 numpy 数组
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        
    # 转换为 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 创建副本
    img_with_boxes = image.copy()
    
    # 获取图像尺寸
    height, width = img_with_boxes.shape[:2]
    
    # 绘制边界框
    for i, box in enumerate(boxes):
        # 转换归一化坐标
        if normalized:
            x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        else:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 绘制矩形
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # 如果有分数，绘制分数
        if scores is not None and i < len(scores):
            score_text = f"{scores[i]:.2f}"
            cv2.putText(img_with_boxes, score_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 如果有标签，绘制标签
        if labels is not None and i < len(labels):
            label_text = f"Class: {labels[i]}"
            cv2.putText(img_with_boxes, label_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_with_boxes

def visualize_batch(images, targets, predictions=None, max_images=4, confidence_threshold=0.3):
    """
    可视化批次数据
    
    Args:
        images: 批次图像
        targets: 批次目标
        predictions: 批次预测，可选
        max_images: 最大可视化图像数量
        confidence_threshold: 置信度阈值
        
    Returns:
        可视化结果图像列表
    """
    results = []
    
    # 限制图像数量
    num_images = min(len(images), max_images)
    
    for i in range(num_images):
        # 转换图像格式
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        
        # 获取真实边界框
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        
        # 绘制真实边界框
        img_with_gt = draw_boxes(img, gt_boxes, color=(0, 255, 0))
        
        # 如果有预测，绘制预测边界框
        if predictions is not None:
            pred_boxes = predictions[i]['boxes'].cpu().numpy()
            pred_scores = predictions[i]['scores'].cpu().numpy()
            
            # 应用置信度阈值
            mask = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            
            # 绘制预测边界框
            img_with_boxes = draw_boxes(img_with_gt, pred_boxes, pred_scores, color=(0, 0, 255))
            results.append(img_with_boxes)
        else:
            results.append(img_with_gt)
    
    return results

def plot_results(history, output_dir='logs'):
    """
    绘制训练结果曲线
    
    Args:
        history: 训练历史记录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    
    # 训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 学习率
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.close()

def save_metrics(metrics, output_path):
    """
    保存评估指标
    
    Args:
        metrics: 评估指标
        output_path: 输出路径
    """
    # 保存为JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 打印指标
    print('评估指标:')
    print(f"AP: {metrics['ap']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Precision at max F1: {metrics['precision_at_max_f1']:.4f}")
    print(f"Recall at max F1: {metrics['recall_at_max_f1']:.4f}")

def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: 第一个边界框，格式为 [x1, y1, x2, y2]
        box2: 第二个边界框，格式为 [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    # 转换为张量
    box1 = torch.tensor(box1).unsqueeze(0)
    box2 = torch.tensor(box2).unsqueeze(0)
    
    # 计算IoU
    iou = box_iou(box1, box2).item()
    
    return iou

def parse_csv_annotation(csv_path):
    """
    解析CSV格式的标注文件
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        标注字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None)
    
    # 初始化标注字典
    annotations = {}
    
    # 解析每一行
    for _, row in df.iterrows():
        image_name, x1, y1, x2, y2, class_name, img_width, img_height = row
        
        # 如果图像不在字典中，添加
        if image_name not in annotations:
            annotations[image_name] = []
        
        # 添加边界框
        annotations[image_name].append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'class': class_name,
            'image_size': [int(img_width), int(img_height)]
        })
    
    return annotations

def improved_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3):
    """
    改进的非极大值抑制算法
    
    Args:
        boxes: 边界框，格式为 [x1, y1, x2, y2]
        scores: 分数
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
        
    Returns:
        保留的边界框索引
    """
    # 转换为张量
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    
    # 应用分数阈值
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # 使用NMS算法
    keep = nms(boxes, scores, iou_threshold)
    
    return keep

def soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3, sigma=0.5, method='gaussian'):
    """
    Soft-NMS算法
    
    Args:
        boxes: 边界框，格式为 [x1, y1, x2, y2]
        scores: 分数
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
        sigma: 高斯衰减参数
        method: 衰减方法，'linear'或'gaussian'
        
    Returns:
        保留的边界框索引和更新后的分数
    """
    # 转换为numpy数组
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # 复制边界框和分数
    boxes_copy = boxes.copy()
    scores_copy = scores.copy()
    
    # 索引矩阵
    N = boxes.shape[0]
    indices = np.arange(N)
    
    # 按分数排序
    order = scores_copy.argsort()[::-1]
    
    # 保留的边界框索引
    keep = []
    
    # Soft-NMS主循环
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算IoU
        ious = np.zeros(order.size)
        for j in range(1, order.size):
            idx = order[j]
            box1 = boxes_copy[i]
            box2 = boxes_copy[idx]
            xx1 = max(box1[0], box2[0])
            yy1 = max(box1[1], box2[1])
            xx2 = min(box1[2], box2[2])
            yy2 = min(box1[3], box2[3])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            
            inter = w * h
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            iou = inter / (area1 + area2 - inter)
            ious[j] = iou
        
        # 更新分数
        if method == 'linear':
            weight = np.where(ious > iou_threshold, 1 - ious, 1)
        else:  # gaussian
            weight = np.exp(-(ious * ious) / sigma)
        
        scores_copy[order[1:]] *= weight[1:]
        
        # 应用分数阈值
        order = order[1:]
        order = order[scores_copy[order] >= score_threshold]
    
    return np.array(keep), scores_copy

def calculate_ap(precision, recall):
    """
    计算平均精度(AP)
    
    Args:
        precision: 精度列表
        recall: 召回率列表
        
    Returns:
        AP值
    """
    # 确保精度和召回率是numpy数组
    precision = np.array(precision)
    recall = np.array(recall)
    
    # 按召回率排序
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    
    # 计算AP(面积)
    ap = 0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    
    return ap