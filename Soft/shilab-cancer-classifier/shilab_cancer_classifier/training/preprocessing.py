"""
图像预处理模块

"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import traceback


def pad_to_square(image, fill_color=(255, 255, 255)):
    """
    将图像填充为正方形，然后调整大小为 224x224
    完全按照原代码实现
    """
    try:
        # 获取原始图像的大小
        width, height = image.size
        
        # 计算填充后的目标大小
        max_side = max(width, height)
        
        # 创建一个新的白色背景图像
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        
        # 将原始图像粘贴到新图像的中心
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        
        # 调整到224x224大小，保持宽高比
        new_image = new_image.resize((224, 224), Image.LANCZOS)
        
        return new_image
    except Exception as e:
        print(f"Error in pad_to_square: {e}")
        print(f"Image info: {image}")
        # 如果出错，返回一个空白的224x224图像
        return Image.new('RGB', (224, 224), fill_color)


def pad_to_square_transform(img):
    """使用这个函数作为转换"""
    try:
        return pad_to_square(img)
    except Exception as e:
        print(f"Error in pad_to_square_transform: {e}")
        # 返回一个默认图像
        return Image.new('RGB', (224, 224), (255, 255, 255))


def pad_to_square_299(image, fill_color=(255, 255, 255)):
    """
    将图像填充为正方形，然后调整大小为 299x299（专为 Inception V3 设计）
    完全按照原代码实现
    """
    try:
        # 获取原始图像的大小
        width, height = image.size
        
        # 计算填充后的目标大小
        max_side = max(width, height)
        
        # 创建一个新的白色背景图像
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        
        # 将原始图像粘贴到新图像的中心
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        
        # 调整到 299x299 大小，保持宽高比
        new_image = new_image.resize((299, 299), Image.LANCZOS)
        
        return new_image
    except Exception as e:
        print(f"Error in pad_to_square_299: {e}")
        print(f"Image info: {image}")
        # 如果出错，返回一个空白的 299x299 图像
        return Image.new('RGB', (299, 299), fill_color)


def pad_to_square_299_transform(img):
    """使用这个函数作为转换"""
    try:
        return pad_to_square_299(img)
    except Exception as e:
        print(f"Error in pad_to_square_299_transform: {e}")
        # 返回一个默认图像
        return Image.new('RGB', (299, 299), (255, 255, 255))


def calculate_mean_std(data_dir, model_name=None):
    """
    计算数据集的均值和标准差
    完全按照原代码实现
    
    参数:
    data_dir: 数据目录
    model_name: 模型名称，用于确定图像大小
    
    返回:
    均值和标准差
    """
    print(f"开始计算均值和标准差，数据目录: {data_dir}")
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])  # 返回ImageNet默认值
    
    try:
        # 根据模型名称选择合适的转换
        if model_name and 'Inception' in model_name:
            # Inception V3 使用 299x299 输入
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_299_transform),
                transforms.ToTensor()
            ])
            print(f"使用 299x299 图像大小计算均值和标准差 (用于 {model_name})")
        else:
            # 其他模型使用 224x224 输入
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_transform),
                transforms.ToTensor()
            ])
            print("使用 224x224 图像大小计算均值和标准差")
        
        # 加载数据集
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"数据集加载成功，共 {len(dataset)} 张图像")
        
        # 初始化均值和标准差
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_images = 0
        
        for i, (images, _) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"处理批次 {i}/{len(dataloader)}")
            
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_size
        
        mean /= total_images
        std /= total_images
        
        print(f"均值和标准差计算完成: mean={mean}, std={std}")
        return mean, std
    
    except Exception as e:
        print(f"计算均值和标准差时出错: {e}")
        traceback.print_exc()
        # 返回ImageNet的默认均值和标准差
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
