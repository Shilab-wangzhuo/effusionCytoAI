# data/data_utils.py 
# 用来存放数据处理相关的函数

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

# 修改细胞对齐函数，返回旋转后的图像和旋转角度（目前用不到）
def align_cell_by_major_axis(image):
    """
    根据细胞的主轴对齐细胞图像
    
    参数:
    image: 输入图像 (OpenCV格式，BGR)
    
    返回:
    aligned_image: 对齐后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 阈值分割，将细胞与背景分离
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有找到轮廓，返回原图
    if not contours:
        return image
    
    # 找到最大的轮廓（假设是细胞）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算最小包围矩形
    rect = cv2.minAreaRect(largest_contour)
    center, (width, height), angle = rect
    
    # 确保宽度大于高度，使主轴始终水平
    if height > width:
        angle += 90
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的图像大小，确保不会裁剪
    h, w = image.shape[:2]
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 应用旋转
    aligned_image = cv2.warpAffine(image, M, (new_w, new_h), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))  # 使用白色填充背景
    
    return aligned_image

# CellImageDataset类（包含“可选的”对图像进行主轴对齐功能和将PIL图像对象转换为PyTorch模型可以处理的格式）
class CellImageDataset(Dataset):
    """
    细胞图像数据集类，支持主轴对齐
    """
    def __init__(self, image_paths, transform=None, source_labels=None, align_cells=False):
        """
        初始化
        
        参数:
        image_paths: 图像路径列表
        transform: PyTorch变换
        source_labels: 源标签列表
        align_cells: 是否进行主轴对齐
        """
        self.image_paths = image_paths
        self.transform = transform
        self.source_labels = source_labels if source_labels is not None else [0] * len(image_paths)
        self.align_cells = align_cells
        
        # 缓存已对齐的图像，避免重复计算
        self.aligned_images_cache = {}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        source_label = self.source_labels[idx]
        
        try: 
            # 如果需要对齐且图像不在缓存中，进行对齐处理
            if self.align_cells:
                if img_path not in self.aligned_images_cache:
                    # 使用OpenCV读取图像
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"无法读取图像: {img_path}")
                    
                    # 转换为RGB（因为OpenCV默认是BGR）
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 进行主轴对齐
                    aligned_img = align_cell_by_major_axis(img)
                    
                    # 将NumPy数组转换为PIL图像
                    img = Image.fromarray(aligned_img)
                    
                    # 缓存对齐后的图像
                    self.aligned_images_cache[img_path] = img
                else:
                    # 使用缓存的对齐图像
                    img = self.aligned_images_cache[img_path]
            else:
                # 不需要对齐，直接读取图像
                img = Image.open(img_path).convert('RGB') # PIL (Python Imaging Library) 的Image.open()函数会按照图像的原始格式打开图像，而不是默认转换为RGB模式。
                
            
            # 应用变换
            if self.transform:
                img = self.transform(img) #将PIL图像对象转换为PyTorch模型可以处理的格式。
            
            return img, source_label, idx
            
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            # 返回一个空白图像
            if self.transform:
                dummy_img = torch.zeros((3, 224, 224))
            else:
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy_img, source_label, idx

# 收集目录下所有支持格式的图片路径 （用于收集reference和candidate域的图片路径）
def collect_image_paths(directory, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']):
    """收集目录下所有支持格式的图片路径"""
    paths = []
    if os.path.exists(directory):
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    return paths

# 收集参考域数据（包含恶性和良性细胞）
def collect_reference_data(malignant_dir, benign_dir):
    """收集参考域数据"""
    image_paths = []
    labels = []
    
    # 映射关系
    categories = {
        "LUAD": 1,
        "LUSC": 2,
        "SCLC": 3
    }
    
    # 检查恶性目录下是否有子文件夹
    import os
    subdirs = [d for d in os.listdir(malignant_dir) 
               if os.path.isdir(os.path.join(malignant_dir, d))]
    
    if subdirs and any(subdir in categories for subdir in subdirs):
        # 如果有子文件夹且子文件夹名在categories中，则按原来的方式处理
        for cat_name, label in categories.items():
            subdir_path = os.path.join(malignant_dir, cat_name)
            if os.path.exists(subdir_path):
                paths = collect_image_paths(subdir_path)
                image_paths.extend(paths)
                labels.extend([label] * len(paths))
    else:
        # 如果没有子文件夹或子文件夹名不在categories中，则将整个恶性目录作为一个整体
        malignant_paths = collect_image_paths(malignant_dir)
        image_paths.extend(malignant_paths)
        labels.extend([5] * len(malignant_paths))  # 5代表malignant
        
    # 收集良性
    benign_paths = collect_image_paths(benign_dir)
    image_paths.extend(benign_paths)
    labels.extend([4] * len(benign_paths))
    
    return image_paths, labels

# 从路径中提取病理ID
def extract_patient_id(patient_folder):
    """从路径中提取病理ID"""
    return os.path.basename(patient_folder)