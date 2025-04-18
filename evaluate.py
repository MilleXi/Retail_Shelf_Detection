import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
from torchvision.ops import box_iou
from collections import defaultdict

from dataset import get_dataloaders
from model import create_model

class Evaluator:
    """
    模型评估器
    """
    def __init__(self, model, test_loader, device=None, confidence_threshold=0.5, iou_threshold=0.5):
        """
        初始化函数
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            device: 设备
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 将模型移动到设备
        self.model.to(self.device)
        self.model.eval()
    
    def calculate_ap(self, precision, recall):
        """
        计算平均精度(AP)
        
        Args:
            precision: 精度列表
            recall: 召回率列表
            
        Returns:
            float: AP值
        """
        # 对精度和召回率进行排序
        order = np.argsort(recall)
        recall = np.array(recall)[order]
        precision = np.array(precision)[order]
        
        # 计算AP(面积)
        ap = 0
        for i in range(len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
        
        return ap
    
    def calculate_map(self, detections, ground_truths):
        """
        计算平均精度均值(mAP)
        
        Args:
            detections: 检测结果列表
            ground_truths: 真实标注列表
            
        Returns:
            dict: 评估指标
        """
        # 初始化指标
        metrics = {
            'precision': [],
            'recall': [],
            'ap': 0,
            'f1': 0
        }
        
        # 按置信度排序
        all_detections = []
        for img_id, dets in detections.items():
            for det in dets:
                all_detections.append({
                    'img_id': img_id,
                    'box': det['box'],
                    'score': det['score']
                })
        
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算TP和FP
        tp = []
        fp = []
        
        # 跟踪已匹配的真实框
        matched_gts = defaultdict(set)
        
        # 计算检测总数和真实标注总数
        n_detections = len(all_detections)
        n_ground_truths = sum(len(gts) for gts in ground_truths.values())
        
        for det in all_detections:
            img_id = det['img_id']
            det_box = torch.tensor([det['box']], dtype=torch.float32)
            
            # 获取当前图像的真实标注
            gt_boxes = ground_truths.get(img_id, [])
            
            if len(gt_boxes) == 0:
                # 如果没有真实标注，则为假阳性
                tp.append(0)
                fp.append(1)
                continue
            
            # 计算当前检测框与所有真实框的IoU
            gt_tensors = [torch.tensor([gt['box']], dtype=torch.float32) for gt in gt_boxes]
            if gt_tensors:
                gt_tensor = torch.cat(gt_tensors, dim=0)
                ious = box_iou(det_box, gt_tensor).squeeze(0)
                
                # 获取最大IoU及其索引
                max_iou, max_idx = torch.max(ious, dim=0)
                
                # 检查IoU是否超过阈值，并且该真实框尚未匹配
                if max_iou >= self.iou_threshold and max_idx.item() not in matched_gts[img_id]:
                    # 为真阳性
                    tp.append(1)
                    fp.append(0)
                    matched_gts[img_id].add(max_idx.item())
                else:
                    # 为假阳性
                    tp.append(0)
                    fp.append(1)
            else:
                # 如果无法计算IoU，则为假阳性
                tp.append(0)
                fp.append(1)
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精度和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (n_ground_truths + 1e-10)
        
        # 计算AP
        ap = self.calculate_ap(precision, recall)
        
        # 计算F1分数
        if len(precision) > 0 and len(recall) > 0:
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            max_f1_idx = np.argmax(f1)
            max_f1 = f1[max_f1_idx]
            precision_at_max_f1 = precision[max_f1_idx]
            recall_at_max_f1 = recall[max_f1_idx]
        else:
            max_f1 = 0
            precision_at_max_f1 = 0
            recall_at_max_f1 = 0
        
        # 设置指标
        metrics['precision'] = precision.tolist()
        metrics['recall'] = recall.tolist()
        metrics['ap'] = float(ap)
        metrics['f1'] = float(max_f1)
        metrics['precision_at_max_f1'] = float(precision_at_max_f1)
        metrics['recall_at_max_f1'] = float(recall_at_max_f1)
        
        return metrics
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            dict: 评估结果
        """
        # 初始化检测和真实标注字典
        detections = {}
        ground_truths = {}
        
        # 进行推理
        with torch.no_grad():
            for images, targets, img_names in tqdm(self.test_loader, desc='Evaluating'):
                # 将图像移动到设备
                images = list(image.to(self.device) for image in images)
                
                # 进行预测
                predictions = self.model.predict(images, confidence_threshold=self.confidence_threshold)
                
                # 解析预测结果
                for i, (pred, target, img_name) in enumerate(zip(predictions, targets, img_names)):
                    # 获取预测框、分数和标签
                    pred_boxes = pred['boxes'].cpu()
                    pred_scores = pred['scores'].cpu()
                    pred_labels = pred['labels'].cpu()
                    
                    # 获取真实框和标签
                    gt_boxes = target['boxes'].cpu()
                    
                    # 保存检测结果
                    img_id = img_name  # 使用图像名称作为ID
                    detections[img_id] = []
                    for j in range(len(pred_boxes)):
                        if pred_scores[j] >= self.confidence_threshold:
                            detections[img_id].append({
                                'box': pred_boxes[j].numpy(),
                                'score': pred_scores[j].item(),
                                'label': pred_labels[j].item()
                            })
                    
                    # 保存真实标注
                    ground_truths[img_id] = []
                    for j in range(len(gt_boxes)):
                        ground_truths[img_id].append({
                            'box': gt_boxes[j].numpy(),
                            'label': 1  # 只有一个类别
                        })
        
        # 计算mAP
        metrics = self.calculate_map(detections, ground_truths)
        
        # 打印结果
        print(f'AP: {metrics["ap"]:.4f}')
        print(f'F1: {metrics["f1"]:.4f}')
        print(f'Precision at max F1: {metrics["precision_at_max_f1"]:.4f}')
        print(f'Recall at max F1: {metrics["recall_at_max_f1"]:.4f}')
        
        return metrics
    
    def visualize_results(self, output_dir='results', max_images=10):
        """
        可视化结果
        
        Args:
            output_dir: 输出目录
            max_images: 最大可视化图像数量
            
        Returns:
            list: 可视化图像路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化图像路径列表
        visualization_paths = []
        
        # 计数器
        count = 0
        
        # 进行推理
        with torch.no_grad():
            for images, targets, img_names in tqdm(self.test_loader, desc='Visualizing'):
                # 将图像移动到设备
                images = list(image.to(self.device) for image in images)
                
                # 获取原始图像
                original_images = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                
                # 进行预测
                predictions = self.model.predict(images, confidence_threshold=self.confidence_threshold)
                
                # 可视化预测结果
                for i, (img, pred, target, img_name) in enumerate(zip(original_images, predictions, targets, img_names)):
                    # 转换图像格式
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # 获取预测框、分数和标签
                    pred_boxes = pred['boxes'].cpu().numpy()
                    pred_scores = pred['scores'].cpu().numpy()
                    
                    # 获取真实框
                    gt_boxes = target['boxes'].cpu().numpy()
                    
                    # 创建图像副本
                    img_with_boxes = img.copy()
                    
                    # 绘制真实框
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制预测框
                    for box, score in zip(pred_boxes, pred_scores):
                        if score >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # 添加置信度标签
                            text = f'{score:.2f}'
                            cv2.putText(img_with_boxes, text, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # 保存结果
                    output_path = os.path.join(output_dir, img_name)
                    cv2.imwrite(output_path, img_with_boxes)
                    visualization_paths.append(output_path)
                    
                    # 增加计数器
                    count += 1
                    if count >= max_images:
                        return visualization_paths
        
        return visualization_paths
    
    def plot_precision_recall_curve(self, metrics, output_dir='results'):
        """
        绘制精度-召回率曲线
        
        Args:
            metrics: 评估指标
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['recall'], metrics['precision'], 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AP = {metrics["ap"]:.4f})')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
        plt.close()

def evaluate_model(root_dir, model_path, config):
    """
    评估模型
    
    Args:
        root_dir: 数据集根目录
        model_path: 模型路径
        config: 配置字典
        
    Returns:
        dict: 评估指标
    """
    # 加载数据
    _, _, test_loader = get_dataloaders(
        root_dir=root_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    model = create_model(
        num_classes=config['num_classes'],
        pretrained=False,
        use_focal_loss=False
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=config['device'],
        confidence_threshold=config['confidence_threshold'],
        iou_threshold=config['iou_threshold']
    )
    
    # 评估模型
    metrics = evaluator.evaluate()
    
    # 可视化结果
    visualization_paths = evaluator.visualize_results(
        output_dir=config['output_dir'],
        max_images=config['max_visualization_images']
    )
    
    # 绘制PR曲线
    evaluator.plot_precision_recall_curve(metrics, output_dir=config['output_dir'])
    
    return metrics

if __name__ == '__main__':
    # 评估配置
    config = {
        'batch_size': 4,
        'num_workers': 4,
        'num_classes': 2,  # 背景 + 物品
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'confidence_threshold': 0.5,
        'iou_threshold': 0.5,
        'output_dir': 'results',
        'max_visualization_images': 20
    }
    
    # 评估模型
    metrics = evaluate_model('SKU110K_fixed', 'checkpoints/best_model.pth', config)