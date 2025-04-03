import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 将标签转换为长整型张量（用于表示类别索引），并确保标签值在有效范围内
        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))
        return {'image': F.to_tensor(image), 'label': label_tensor}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, num_classes=4):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.num_classes = num_classes
        
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.mask_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def mask_loader(self, path):
        """加载掩码为灰度图像，并将灰度值映射到类别索引（0-3）"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            mask = np.array(img.convert('L'))
            
            # 根据灰度值范围映射到类别索引
            unique_values = np.unique(mask)
            
            # 创建新的掩码数组
            mapped_mask = np.zeros_like(mask)
            
            # 映射规则
            if 0 in unique_values:
                mapped_mask[mask == 0] = 0  # 背景
            
            # 检查非零值并映射
            non_zero_values = unique_values[unique_values > 0]
            if len(non_zero_values) > 0:
                # 如果只有一个非零值，映射为类别1
                if len(non_zero_values) == 1:
                    mapped_mask[mask == non_zero_values[0]] = 1
                
                # 如果有2个非零值，映射为类别1和2
                elif len(non_zero_values) == 2:
                    mapped_mask[mask == non_zero_values[0]] = 1
                    mapped_mask[mask == non_zero_values[1]] = 2
                
                # 如果有3个或更多非零值，映射前3个为类别1、2、3
                elif len(non_zero_values) >= 3:
                    sorted_values = np.sort(non_zero_values)
                    mapped_mask[mask == sorted_values[0]] = 1
                    mapped_mask[mask == sorted_values[1]] = 2
                    # 将剩余的较高值都映射到类别3
                    for val in sorted_values[2:]:
                        mapped_mask[mask == val] = 3
            
            return Image.fromarray(mapped_mask.astype(np.uint8))

    def binary_loader(self, path):
        """保留旧的二进制加载方法，以兼容旧代码"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')