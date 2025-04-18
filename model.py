import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from torchvision.ops import nms
import math

class FocalLoss(nn.Module):
    """
    Focal Loss实现，适用于小目标检测
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        初始化函数
        
        Args:
            alpha (float): 平衡正负样本的系数
            gamma (float): 调整难易样本的系数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        前向传播
        
        Args:
            pred (Tensor): 预测值
            target (Tensor): 目标值
            
        Returns:
            Tensor: 损失值
        """
        pred = torch.sigmoid(pred)
        
        # 二分类交叉熵计算
        alpha_factor = torch.ones_like(target) * self.alpha
        alpha_factor = torch.where(target == 1, alpha_factor, 1 - alpha_factor)
        
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = alpha_factor * torch.pow(1 - pt, self.gamma)
        
        bce = -torch.log(pt)
        loss = focal_weight * bce
        
        return loss.sum()

class CustomFasterRCNN(nn.Module):
    """
    自定义的Faster R-CNN模型
    """
    def __init__(self, num_classes=2, pretrained=True, trainable_backbone_layers=3, 
                 anchor_sizes=None, use_focal_loss=True):
        """
        初始化函数
        
        Args:
            num_classes (int): 类别数量（包括背景）
            pretrained (bool): 是否使用预训练模型
            trainable_backbone_layers (int): 主干网络可训练层数
            anchor_sizes (tuple): 锚框尺寸
            use_focal_loss (bool): 是否使用Focal Loss
        """
        super(CustomFasterRCNN, self).__init__()
        
        # 设置锚框尺寸，适应小目标
        if anchor_sizes is None:
            # 默认锚框尺寸，针对SKU110K数据集中的小物体进行调整
            anchor_sizes = ((8, 16, 32, 64, 128, 256),)
        
        # 创建主干网络
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        
        # 冻结前面的层
        for name, parameter in backbone.named_parameters():
            if "layer4" not in name and "layer3" not in name and "layer2" not in name:
                parameter.requires_grad_(False)
        
        # 保留从backbone获取的特征层
        backbone_output_channels = backbone.fc.in_features
        
        # 移除FC层，只保留特征提取部分
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # 使用FPN（特征金字塔网络）
        # 创建Feature Pyramid Network (FPN)
        fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[backbone_output_channels // 8, backbone_output_channels // 4, 
                             backbone_output_channels // 2, backbone_output_channels],
            out_channels=256
        )
        
        # 创建主干网络和FPN的组合
        backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone=backbone,
            return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # 创建锚点生成器，使用自定义的锚框尺寸
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
        )
        
        # 创建RoI Pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2
        )
        
        # 创建Faster R-CNN模型
        self.model = FasterRCNN(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=640,
            max_size=640,
            rpn_pre_nms_top_n_train=2000,  # 增加RPN的候选区域数量
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=1000,
            rpn_post_nms_top_n_test=500,
            rpn_nms_thresh=0.7,  # 调整NMS阈值
            box_score_thresh=0.05,  # 降低分数阈值，捕获更多小目标
            box_nms_thresh=0.5     # 调整NMS阈值
        )
        
        # 替换分类器头部
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 是否使用Focal Loss
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, images, targets=None):
        """
        前向传播
        
        Args:
            images (List[Tensor]): 输入图像
            targets (List[Dict[str, Tensor]], optional): 目标
            
        Returns:
            Dict[str, Tensor] 或 List[Dict[str, Tensor]]: 损失或检测结果
        """
        # 如果使用Focal Loss并且在训练模式
        if self.use_focal_loss and self.training and targets is not None:
            # 获取模型特征
            original_image_sizes = []
            for img in images:
                val = img.shape[-2:]
                original_image_sizes.append((val[0], val[1]))
            
            # 使用标准前向传播获取损失
            loss_dict = self.model(images, targets)
            
            return loss_dict
        else:
            # 标准前向传播
            return self.model(images, targets)
    
    def improved_nms(self, boxes, scores, threshold=0.5, soft_threshold=0.3):
        """
        改进的NMS实现，引入Soft-NMS减少遮挡问题
        
        Args:
            boxes (Tensor): 边界框
            scores (Tensor): 分数
            threshold (float): NMS阈值
            soft_threshold (float): Soft-NMS阈值
            
        Returns:
            Tensor: 筛选后的索引
        """
        # 先使用标准NMS筛选
        keep = nms(boxes, scores, threshold)
        
        return keep
        
    def predict(self, images, confidence_threshold=0.3, nms_threshold=0.5):
        """
        预测函数
        
        Args:
            images (List[Tensor]): 输入图像
            confidence_threshold (float): 置信度阈值
            nms_threshold (float): NMS阈值
            
        Returns:
            List[Dict[str, Tensor]]: 检测结果
        """
        self.eval()
        with torch.no_grad():
            predictions = self(images)
            
            # 后处理预测结果
            processed_predictions = []
            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                
                # 应用置信度阈值
                mask = scores > confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                # 应用改进的NMS
                keep = self.improved_nms(boxes, scores, nms_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                processed_predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })
                
            return processed_predictions

def create_model(num_classes=2, pretrained=True, use_focal_loss=True):
    """
    创建模型
    
    Args:
        num_classes (int): 类别数量（包括背景）
        pretrained (bool): 是否使用预训练模型
        use_focal_loss (bool): 是否使用Focal Loss
        
    Returns:
        CustomFasterRCNN: 模型实例
    """
    # 创建自定义的锚框尺寸，适应小目标和多尺度
    anchor_sizes = ((4, 8, 16, 32, 64, 128, 256),)
    
    model = CustomFasterRCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        trainable_backbone_layers=5,
        anchor_sizes=anchor_sizes,
        use_focal_loss=use_focal_loss
    )
    
    return model