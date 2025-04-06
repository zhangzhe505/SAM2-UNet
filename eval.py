import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def map_pixels_to_classes(pixels, num_classes=4):
    """将像素值映射到类别索引"""
    if num_classes == 4:
        # 四类映射: [0, 85, 170, 255] -> [0, 1, 2, 3]
        # 0-84 -> 0, 85-169 -> 1, 170-254 -> 2, 255 -> 3
        thresholds = [85, 170, 255]
        class_values = [0, 85, 170, 255]
    elif num_classes == 3:
        # 三类映射: [0, 127, 255] -> [0, 1, 2]
        # 0-126 -> 0, 127-254 -> 1, 255 -> 2
        thresholds = [127, 255]
        class_values = [0, 127, 255]
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")
    
    mapped = np.zeros_like(pixels)
    for i in range(len(class_values)):
        if i < len(class_values) - 1:
            mask = (pixels >= class_values[i]) & (pixels < class_values[i + 1])
        else:
            mask = pixels == class_values[i]
        mapped[mask] = i
    return mapped

def calculate_metrics(pred, gt, num_classes):
    """计算分割指标"""
    metrics = {
        'dice': np.zeros(num_classes),
        'iou': np.zeros(num_classes),
        'pixel_acc': 0,
        'class_pixel_acc': np.zeros(num_classes),
        'class_precision': np.zeros(num_classes),
        'class_recall': np.zeros(num_classes)
    }
    
    # 计算每个类别的指标
    for class_idx in range(num_classes):
        # 获取当前类别的预测和真实值
        pred_class = (pred == class_idx)
        gt_class = (gt == class_idx)
        
        # 计算交集和并集
        intersection = np.sum(pred_class & gt_class)
        union = np.sum(pred_class | gt_class)
        
        # 计算 Dice 系数
        if union > 0:
            metrics['dice'][class_idx] = (2 * intersection) / (np.sum(pred_class) + np.sum(gt_class))
        
        # 计算 IoU
        if union > 0:
            metrics['iou'][class_idx] = intersection / union
        
        # 计算精确度和召回率
        if np.sum(gt_class) > 0:
            metrics['class_precision'][class_idx] = intersection / (np.sum(pred_class) + 1e-6)
            metrics['class_recall'][class_idx] = intersection / (np.sum(gt_class) + 1e-6)
    
    # 计算像素准确率
    metrics['pixel_acc'] = np.sum(pred == gt) / (pred.size)
    
    # 计算每个类别的像素准确率
    for class_idx in range(num_classes):
        if np.sum(gt == class_idx) > 0:
            metrics['class_pixel_acc'][class_idx] = np.sum((pred == class_idx) & (gt == class_idx)) / np.sum(gt == class_idx)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation metrics')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to prediction masks')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth masks')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    
    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.pred_path):
        raise ValueError(f"Prediction path does not exist: {args.pred_path}")
    if not os.path.exists(args.gt_path):
        raise ValueError(f"Ground truth path does not exist: {args.gt_path}")

    # 获取所有预测文件
    pred_files = sorted([f for f in os.listdir(args.pred_path) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(args.gt_path) if f.endswith('.png')])

    if len(pred_files) != len(gt_files):
        raise ValueError(f"Number of prediction files ({len(pred_files)}) does not match ground truth files ({len(gt_files)})")

    # 初始化统计变量
    total_metrics = {
        'dice': np.zeros(args.num_classes),
        'iou': np.zeros(args.num_classes),
        'pixel_acc': 0,
        'class_pixel_acc': np.zeros(args.num_classes),
        'class_precision': np.zeros(args.num_classes),
        'class_recall': np.zeros(args.num_classes)
    }
    dice_values = [[] for _ in range(args.num_classes)]
    iou_values = [[] for _ in range(args.num_classes)]
    precision_values = [[] for _ in range(args.num_classes)]
    recall_values = [[] for _ in range(args.num_classes)]

    # 遍历所有文件
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        # 读取预测和真实值
        pred = np.array(Image.open(os.path.join(args.pred_path, pred_file)))
        gt = np.array(Image.open(os.path.join(args.gt_path, gt_file)))

        # 将真实值像素值映射到类别索引
        gt = map_pixels_to_classes(gt, args.num_classes)

        # 检查像素值范围
        if pred.max() >= args.num_classes or gt.max() >= args.num_classes:
            raise ValueError(f"Pixel values exceed class range: pred.max()={pred.max()}, gt.max()={gt.max()}")

        # 计算指标
        metrics = calculate_metrics(pred, gt, args.num_classes)

        # 累加指标
        for class_idx in range(args.num_classes):
            total_metrics['dice'][class_idx] += metrics['dice'][class_idx]
            total_metrics['iou'][class_idx] += metrics['iou'][class_idx]
            total_metrics['class_pixel_acc'][class_idx] += metrics['class_pixel_acc'][class_idx]
            total_metrics['class_precision'][class_idx] += metrics['class_precision'][class_idx]
            total_metrics['class_recall'][class_idx] += metrics['class_recall'][class_idx]

            # 记录每个类别的指标分布
            if metrics['dice'][class_idx] > 0:
                dice_values[class_idx].append(metrics['dice'][class_idx])
            if metrics['iou'][class_idx] > 0:
                iou_values[class_idx].append(metrics['iou'][class_idx])
            if metrics['class_precision'][class_idx] > 0:
                precision_values[class_idx].append(metrics['class_precision'][class_idx])
            if metrics['class_recall'][class_idx] > 0:
                recall_values[class_idx].append(metrics['class_recall'][class_idx])

        total_metrics['pixel_acc'] += metrics['pixel_acc']

    # 计算平均值
    num_files = len(pred_files)
    for class_idx in range(args.num_classes):
        total_metrics['dice'][class_idx] /= num_files
        total_metrics['iou'][class_idx] /= num_files
        total_metrics['class_pixel_acc'][class_idx] /= num_files
        total_metrics['class_precision'][class_idx] /= num_files
        total_metrics['class_recall'][class_idx] /= num_files

    total_metrics['pixel_acc'] /= num_files

    # 打印总体指标
    print("\nOverall Metrics:")
    print(f"Mean Dice: {total_metrics['dice'].mean():.4f}")
    print(f"Mean IoU: {total_metrics['iou'].mean():.4f}")
    print(f"Pixel Accuracy: {total_metrics['pixel_acc']:.4f}")
    print(f"Mean Precision: {total_metrics['class_precision'].mean():.4f}")
    print(f"Mean Recall: {total_metrics['class_recall'].mean():.4f}")

    # 打印每个类别的指标
    print("\nPer-Class Metrics:")
    for class_idx in range(args.num_classes):
        print(f"\nClass {class_idx}:")
        print(f"Dice: {total_metrics['dice'][class_idx]:.4f}")
        print(f"IoU: {total_metrics['iou'][class_idx]:.4f}")
        print(f"Pixel Accuracy: {total_metrics['class_pixel_acc'][class_idx]:.4f}")
        print(f"Precision: {total_metrics['class_precision'][class_idx]:.4f}")
        print(f"Recall: {total_metrics['class_recall'][class_idx]:.4f}")

        # 打印指标分布
        if dice_values[class_idx]:
            print(f"Dice Range: {min(dice_values[class_idx]):.4f} - {max(dice_values[class_idx]):.4f}")
        if iou_values[class_idx]:
            print(f"IoU Range: {min(iou_values[class_idx]):.4f} - {max(iou_values[class_idx]):.4f}")
        if precision_values[class_idx]:
            print(f"Precision Range: {min(precision_values[class_idx]):.4f} - {max(precision_values[class_idx]):.4f}")
        if recall_values[class_idx]:
            print(f"Recall Range: {min(recall_values[class_idx]):.4f} - {max(recall_values[class_idx]):.4f}")

if __name__ == "__main__":
    main()
