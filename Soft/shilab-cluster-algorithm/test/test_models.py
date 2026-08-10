#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 models.py 中的 get_model_and_transform 函数
使用指定目录下的第五折模型权重进行测试
"""

import os
import torch
from src.models.models import get_model_and_transform
from torchvision import transforms
from PIL import Image
import glob

def test_get_model_and_transform():
    """
    测试 get_model_and_transform 函数
    """
    # 模型权重目录
    model_dir = r"H:\limr\project\urine\result\classification_model\UR_cluster\train_val_models"
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 目录 {model_dir} 不存在")
        return
    
    # 获取所有第五折的模型权重文件
    fold5_pattern = os.path.join(model_dir, "*", "*_fold_5.pth")
    fold5_models = glob.glob(fold5_pattern)
    
    if not fold5_models:
        print(f"在 {model_dir} 中未找到第五折模型权重文件 (*_fold_5.pth)")
        return
    
    print(f"找到 {len(fold5_models)} 个第五折模型权重文件:")
    for model_path in fold5_models:
        print(f"  - {os.path.basename(model_path)}")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 测试每个模型
    successful_models = []
    failed_models = []
    
    for model_path in fold5_models:
        # 从文件名推断模型类型
        filename = os.path.basename(model_path)
        model_name = filename.split('_fold_')[0]  # 提取模型名称部分
        
        # 尝试匹配标准模型名称（根据SUPPORTED_MODELS中的命名）
        # 这里我们尝试将小写的模型名转换为标准命名
        model_type_mapping = {
            'alexnet': 'AlexNet',
            'resnext': 'ResNeXt', 
            'convnextlarge': 'ConvNextLarge',
            'googlenet': 'GoogLeNet',
            'resnet50': 'ResNet50',
            'resnet101': 'ResNet101',
            'densenet121': 'DenseNet121',
            'densenet161': 'DenseNet161',
            'vgg16': 'VGG16',
            'inceptionv3': 'InceptionV3',
            'efficientnetb0': 'EfficientNetB0',
            'efficientnetv2large': 'EfficientNetV2Large',
            'mobilenetv2': 'MobileNetV2',
            'mobilenetv3': 'MobileNetV3',
            'regnet': 'RegNet',
            'shufflenetv2': 'ShuffleNetV2',
            'squeezenet': 'SqueezeNet',
            'vit_b16': 'ViT_B16',
            'vit_l16': 'ViT_L16',
        }
        
        # 尝试匹配模型类型
        matched_model_type = None
        for key, value in model_type_mapping.items():
            if key in model_name.lower():
                matched_model_type = value
                break
        
        if matched_model_type is None:
            print(f"跳过 {filename} - 无法匹配到支持的模型类型")
            continue
        
        print(f"\n正在测试模型: {matched_model_type} (权重: {filename})")
        
        try:
            # 调用 get_model_and_transform 函数
            model, transform, input_size = get_model_and_transform(
                model_type=matched_model_type,
                model_path=model_path,
                device=device
            )
            
            # 验证返回的对象
            print(f"  ✓ 模型类型: {type(model)}")
            print(f"  ✓ 输入尺寸: {input_size}")
            print(f"  ✓ Transform类型: {type(transform)}")
            
            # 尝试创建一个虚拟输入进行简单的前向传播测试
            dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
            
            successful_models.append((matched_model_type, model_path))
            print(f"  ✓ {matched_model_type} 测试成功!")
            
        except Exception as e:
            print(f"  ✗ {matched_model_type} 测试失败: {str(e)}")
            failed_models.append((matched_model_type, model_path, str(e)))
    
    # 输出总结
    print(f"\n{'='*60}")
    print("测试总结:")
    print(f"成功: {len(successful_models)} 个模型")
    print(f"失败: {len(failed_models)} 个模型")
    
    if successful_models:
        print("\n成功的模型:")
        for model_type, path in successful_models:
            print(f"  - {model_type}")
    
    if failed_models:
        print("\n失败的模型:")
        for model_type, path, error in failed_models:
            print(f"  - {model_type}: {error}")


def test_specific_model(model_type, model_path):
    """
    测试特定模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"测试模型类型: {model_type}")
        print(f"模型路径: {model_path}")
        
        model, transform, input_size = get_model_and_transform(
            model_type=model_type,
            model_path=model_path,
            device=device
        )
        
        print(f"模型加载成功! 输入尺寸: {input_size}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"前向传播成功，输出形状: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始测试 get_model_and_transform 函数...")
    test_get_model_and_transform()