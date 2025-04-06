import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2UNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
parser.add_argument("--num_classes", type=int, default=4,
                    help="number of segmentation classes (including background)")
parser.add_argument("--colored_output", action="store_true", 
                    help="save colored segmentation results")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352, num_classes=args.num_classes)
model = SAM2UNet(num_classes=args.num_classes).to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()

# 创建保存目录
os.makedirs(args.save_path, exist_ok=True)
if args.colored_output:
    colored_save_path = os.path.join(args.save_path, "colored")
    os.makedirs(colored_save_path, exist_ok=True)

# 为分割掩码创建颜色映射
color_map = {
    0: [0, 0, 0],       # 背景: 黑色
    1: [255, 0, 0],     # 类别1: 红色
    2: [0, 255, 0],     # 类别2: 绿色
    3: [0, 0, 255]      # 类别3: 蓝色
}

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image)
        
        # 对于多类别分割，使用softmax而不是sigmoid
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        
        # 获取每个像素的预测类别
        if args.num_classes > 1:
            # 获取最可能的类别
            pred = F.softmax(res, dim=1)
            pred_class = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            
            # 保存类别预测结果（灰度图）
            pred_save_path = os.path.join(args.save_path, name[:-4] + ".png")
            imageio.imsave(pred_save_path, pred_class.astype(np.uint8))
            
            # 创建彩色可视化
            if args.colored_output:
                colored_mask = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
                for class_idx in range(args.num_classes):
                    colored_mask[pred_class == class_idx] = color_map[class_idx]
                colored_pred_save_path = os.path.join(colored_save_path, name[:-4] + "_colored.png")
                imageio.imsave(colored_pred_save_path, colored_mask)
        else:
            # 对于二分类分割，保持原有的处理方式
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)
            imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
        
        print(f"处理 {name} 完成")

print("测试完成！结果已保存到:", args.save_path)
