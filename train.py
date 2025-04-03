import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--num_classes", type=int, default=4,
                    help="number of segmentation classes (including background)")
parser.add_argument("--dataset_path", type=str, default=None,
                    help="base directory of the dataset (optional)")
parser.add_argument("--val_image_path", type=str, default=None,
                    help="path to validation images")
parser.add_argument("--val_mask_path", type=str, default=None,
                    help="path to validation masks")
args = parser.parse_args()


def multi_class_loss(pred, mask):
    """多类别分割损失函数，结合交叉熵损失和Dice损失"""
    # 交叉熵损失
    ce_loss = F.cross_entropy(pred, mask)
    
    # Dice损失
    pred_softmax = F.softmax(pred, dim=1)
    batch_size = pred.size(0)
    
    # 对每个类别计算Dice系数
    dice_loss = 0
    for cls in range(args.num_classes):
        # 将当前类别作为前景，其他类别作为背景
        pred_cls = pred_softmax[:, cls]
        target_cls = (mask == cls).float()
        
        # 计算此类别的Dice系数
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice_coef = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss += (1 - dice_coef).mean()
    
    # 平均所有类别的Dice损失
    dice_loss /= args.num_classes
    
    # 返回总损失（可以调整权重）
    return ce_loss + dice_loss


def main(args):
    # 如果提供了dataset_path，则使用标准目录结构
    if args.dataset_path:
        train_image_path = os.path.join(args.dataset_path, 'train', 'images')
        train_mask_path = os.path.join(args.dataset_path, 'train', 'masks')
        val_image_path = os.path.join(args.dataset_path, 'val', 'images')
        val_mask_path = os.path.join(args.dataset_path, 'val', 'masks')
    else:
        train_image_path = args.train_image_path
        train_mask_path = args.train_mask_path
        val_image_path = args.val_image_path
        val_mask_path = args.val_mask_path
    
    # 创建数据集和数据加载器
    train_dataset = FullDataset(train_image_path, train_mask_path, 352, mode='train', num_classes=args.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    # 如果提供验证集路径，则创建验证数据加载器
    val_loader = None
    if val_image_path and val_mask_path:
        val_dataset = FullDataset(val_image_path, val_mask_path, 352, mode='val', num_classes=args.num_classes)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    else:
        print(f"训练集大小: {len(train_dataset)}, 未提供验证集")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = SAM2UNet(args.hiera_path, num_classes=args.num_classes)
    model.to(device)
    
    # 优化器和学习率调度器
    optimizer = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], 
                          lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)
    
    # 记录最佳验证损失（用于模型选择）
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(args.epoch):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            out0, out1, out2 = model(images)
            
            # 计算损失
            loss0 = multi_class_loss(out0, masks)
            loss1 = multi_class_loss(out1, masks)
            loss2 = multi_class_loss(out2, masks)
            loss = loss0 + loss1 + loss2
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印状态信息
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Avg Train Loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['label'].to(device)
                    
                    out0, out1, out2 = model(images)
                    
                    loss0 = multi_class_loss(out0, masks)
                    loss1 = multi_class_loss(out1, masks)
                    loss2 = multi_class_loss(out2, masks)
                    loss = loss0 + loss1 + loss2
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-best.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'SAM2-UNet-{epoch+1}.pth'))
            print(f"Saved checkpoint: SAM2-UNet-{epoch+1}.pth")


def seed_torch(seed=1024):
    """设置随机数种子以确保结果可重现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1024)  # 为了结果可重现
    main(args)