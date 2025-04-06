import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="路径到预测结果")
    parser.add_argument("--gt_path", type=str, required=True, help="路径到真实标签")
    parser.add_argument("--num_classes", type=int, default=4, help="类别数量")
    args = parser.parse_args()
    
    # 获取所有预测文件
    pred_files = sorted([f for f in os.listdir(args.pred_path) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(args.gt_path) if f.endswith('.png')])
    
    if len(pred_files) != len(gt_files):
        raise ValueError(f"预测文件数量({len(pred_files)})和真值文件数量({len(gt_files)})不匹配")
    
    # 初始化总指标
    total_metrics = {
        'dice': np.zeros(args.num_classes),
        'iou': np.zeros(args.num_classes),
        'pixel_acc': 0,
        'class_pixel_acc': np.zeros(args.num_classes),
        'class_precision': np.zeros(args.num_classes),
        'class_recall': np.zeros(args.num_classes)
    }
    
    # 为每个类别创建列表来存储每个图像的指标
    class_metrics = {}
    for class_idx in range(args.num_classes):
        class_metrics[class_idx] = {
            'dice': [],
            'iou': [],
            'pixel_acc': [],
            'precision': [],
            'recall': []
        }
    
    # 计算每个图像的指标
    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        # 读取预测和真值
        pred = np.array(Image.open(os.path.join(args.pred_path, pred_file)))
        gt = np.array(Image.open(os.path.join(args.gt_path, gt_file)))
        
        # 确保尺寸匹配
        if pred.shape != gt.shape:
            raise ValueError(f"预测图像({pred.shape})和真值图像({gt.shape})尺寸不匹配")
        
        # 确保像素值在有效范围内
        if pred.max() >= args.num_classes or gt.max() >= args.num_classes:
            raise ValueError(f"像素值超出类别范围: pred.max()={pred.max()}, gt.max()={gt.max()}")
        
        # 计算指标
        metrics = calculate_metrics(pred, gt, args.num_classes)
        
        # 累加指标
        total_metrics['dice'] += metrics['dice']
        total_metrics['iou'] += metrics['iou']
        total_metrics['pixel_acc'] += metrics['pixel_acc']
        total_metrics['class_pixel_acc'] += metrics['class_pixel_acc']
        total_metrics['class_precision'] += metrics['class_precision']
        total_metrics['class_recall'] += metrics['class_recall']
        
        # 保存每个类别的指标
        for class_idx in range(args.num_classes):
            if metrics['dice'][class_idx] > 0:
                class_metrics[class_idx]['dice'].append(metrics['dice'][class_idx])
                class_metrics[class_idx]['iou'].append(metrics['iou'][class_idx])
                class_metrics[class_idx]['pixel_acc'].append(metrics['class_pixel_acc'][class_idx])
                class_metrics[class_idx]['precision'].append(metrics['class_precision'][class_idx])
                class_metrics[class_idx]['recall'].append(metrics['class_recall'][class_idx])
    
    # 计算平均指标
    num_images = len(pred_files)
    for key in total_metrics:
        total_metrics[key] /= num_images
    
    # 打印整体指标
    print("\n整体指标:")
    print(f"平均 Dice 系数: {total_metrics['dice'].mean():.4f}")
    print(f"平均 IoU: {total_metrics['iou'].mean():.4f}")
    print(f"像素准确率: {total_metrics['pixel_acc']:.4f}")
    print(f"平均精确度: {total_metrics['class_precision'].mean():.4f}")
    print(f"平均召回率: {total_metrics['class_recall'].mean():.4f}")
    
    # 打印每个类别的指标
    print("\n每个类别的指标:")
    class_names = ['背景', '类别1', '类别2', '类别3']
    for class_idx in range(args.num_classes):
        print(f"\n{class_names[class_idx]}:")
        print(f"Dice 系数: {total_metrics['dice'][class_idx]:.4f}")
        print(f"IoU: {total_metrics['iou'][class_idx]:.4f}")
        print(f"像素准确率: {total_metrics['class_pixel_acc'][class_idx]:.4f}")
        print(f"精确度: {total_metrics['class_precision'][class_idx]:.4f}")
        print(f"召回率: {total_metrics['class_recall'][class_idx]:.4f}")
        
        # 打印每个类别的指标分布
        if len(class_metrics[class_idx]['dice']) > 0:
            print(f"\n指标分布:")
            print(f"Dice 系数 (min/max/mean): ")
            print(f"最小值: {min(class_metrics[class_idx]['dice']):.4f}")
            print(f"最大值: {max(class_metrics[class_idx]['dice']):.4f}")
            print(f"平均值: {np.mean(class_metrics[class_idx]['dice']):.4f}")
            
            print(f"精确度 (min/max/mean): ")
            print(f"最小值: {min(class_metrics[class_idx]['precision']):.4f}")
            print(f"最大值: {max(class_metrics[class_idx]['precision']):.4f}")
            print(f"平均值: {np.mean(class_metrics[class_idx]['precision']):.4f}")
            
            print(f"召回率 (min/max/mean): ")
            print(f"最小值: {min(class_metrics[class_idx]['recall']):.4f}")
            print(f"最大值: {max(class_metrics[class_idx]['recall']):.4f}")
            print(f"平均值: {np.mean(class_metrics[class_idx]['recall']):.4f}")

if __name__ == "__main__":
    main()
