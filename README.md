# SKU110K 目标检测项目

## 项目概述

本项目是基于SKU110K数据集的小目标检测系统，使用改进的Faster R-CNN模型实现超市货架上商品的检测。主要特点包括：

- 多尺度训练（0.5x-1.5x缩放）
- 640x640分辨率的输入图像
- 针对小目标优化的Anchor Box尺寸
- 使用Focal Loss优化小目标检测
- 改进的NMS（非极大值抑制）提高检测精度

## 数据集

SKU110K_fixed数据集包含超市货架上的商品图像，结构如下：

```
SKU110K_fixed/
├── annotations/
│   ├── annotations_train.csv
│   ├── annotations_val.csv
│   └── annotations_test.csv
└── images/
    ├── train_0.jpg
    ├── train_1.jpg
    └── ...
```

注解文件格式为CSV，列包括：`image_name,x1,y1,x2,y2,class,image_width,image_height`

如果您还没有SKU-110K数据集，可以从原始论文作者提供的链接下载：
[SKU-110K数据集](https://github.com/eg4000/SKU110K_CVPR19)

下载后将数据解压到项目目录下的`SKU110K_fixed`文件夹中。

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 系统要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+（推荐使用GPU加速）

## 项目结构

```
.
├── README.md               # 项目说明文档
├── dataset.py              # 数据集加载和处理
├── model.py                # 模型定义
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── utils.py                # 工具函数
├── main.py                 # 主程序入口
└── requirements.txt        # 依赖包列表
```

## 使用说明

### 训练模型

```bash
python main.py --mode train --data_dir SKU110K_fixed --batch_size 8 --num_epochs 50
```

主要参数说明：
- `--data_dir`: 数据集目录
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率（默认：0.0001）
- `--optimizer`: 优化器选择（默认：adamw）
- `--scheduler`: 学习率调度器选择（默认：cosine）

### 评估模型

```bash
python main.py --mode eval --data_dir SKU110K_fixed --model_path checkpoints/best_model.pth
```

主要参数说明：
- `--model_path`: 模型权重路径
- `--confidence_threshold`: 置信度阈值（默认：0.5）
- `--iou_threshold`: IoU阈值（默认：0.5）
- `--output_dir`: 输出目录（默认：results）

### 训练并评估模型

```bash
python main.py --mode train_eval --data_dir SKU110K_fixed
```

## 模型说明

本项目使用了基于ResNet50主干网络的Faster R-CNN模型，并进行了以下改进：

1. **特征金字塔网络（FPN）**：通过融合不同尺度的特征，提升小目标检测性能
2. **自定义Anchor Box设计**：针对SKU110K数据集的商品尺寸，优化了锚框尺寸
3. **Focal Loss**：解决正负样本不平衡问题，提高对难样本的关注度
4. **多尺度训练**：图像缩放范围为0.5x-1.5x，提高模型对不同尺寸目标的适应性
5. **改进的NMS**：优化非极大值抑制算法，减少遮挡问题

## 代码模块说明

### dataset.py

负责数据集加载和预处理，主要功能：
- 解析CSV格式的标注文件
- 数据增强（随机裁剪、翻转、色彩抖动等）
- 多尺度训练支持
- 批次数据整理

### model.py

定义网络模型结构，主要内容：
- 自定义的Faster R-CNN模型
- 特征金字塔网络（FPN）集成
- Focal Loss实现
- 改进的NMS（非极大值抑制）实现

### train.py

模型训练流程，主要功能：
- 训练和验证循环
- 学习率调度
- 早停机制
- 检查点保存
- 训练过程可视化

### evaluate.py

模型评估模块，主要功能：
- 模型性能评估（AP、F1等指标）
- 结果可视化
- 精度-召回率曲线绘制

### utils.py

工具函数集合，主要功能：
- 坐标格式转换
- 边界框绘制
- IoU计算
- 数据可视化

### main.py

主程序入口，提供命令行接口。

## 结果展示

训练完成后，可以在`results`目录下查看以下输出：
- 检测结果可视化
- 精度-召回率曲线
- 评估指标结果

## 引用

如果使用本项目代码，请引用SKU110K原始数据集论文：

```
@article{goldman2019precise,
  title={Precise Detection in Densely Packed Scenes},
  author={Goldman, Eran and Herzig, Roei and Eisenschtat, Aviv and Goldberger, Jacob and Hassner, Tal},
  journal={arXiv preprint arXiv:1904.00853},
  year={2019}
}
```

## 许可证

MIT License