# cross_cluster_matching\models\models.py  改版
# 用来存放模型定义的类和获取模型及转换函数的工厂函数

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os


# AlexNet特征提取器
class CustomAlexNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomAlexNetFeatureExtractor, self).__init__()
        self.alexnet = models.alexnet(pretrained=False)
        
        # 修改最后一层以匹配权重文件的结构（如果有必要）
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)
        
        # 1. 定义特征提取部分：features + avgpool + flatten
        # 2. 定义分类器特征部分：取 classifier 的前6层 (索引0到5)
        #    包含: Dropout -> Linear -> ReLU -> Dropout -> Linear -> ReLU
        #    这样输出的就是标准的 4096 维 embedding
        self.feature_extractor_part = self.alexnet.classifier[:6] 
        
        # 加载权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        # 基础特征提取
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        if layer == 'penultimate':
            # 使用切片好的层，自动包含 Linear+ReLU
            x = self.feature_extractor_part(x)
            return x
        else:
            # 完整的分类输出
            x = self.alexnet.classifier(x)
            return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('alexnet.') for k in state_dict.keys()):
                # 移除'alexnet.'前缀
                new_state_dict = {k[8:]: v for k, v in state_dict.items()}
                self.alexnet.load_state_dict(new_state_dict, strict=False)
                print("AlexNet模型权重加载成功（已处理'alexnet.'前缀）！")
            else:
                # 直接加载
                self.alexnet.load_state_dict(state_dict, strict=False)
                print("AlexNet模型权重加载成功！")
        except Exception as e:
            print(f"加载AlexNet模型权重时出错: {e}")
            raise RuntimeError(f"无法加载AlexNet模型权重: {e}")

# ResNeXt模型定义 - 修改为特征提取器
class CustomResNeXtFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNeXtFeatureExtractor, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=False)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)
        self.load_weights(model_path)
        
        # 移除最后的全连接层
        self.penultimate_layer = nn.Sequential(*list(self.resnext.children())[:-1])
        
    def forward(self, x, layer='penultimate'):
        # 修正：使用 flatten 替代 squeeze，防止 batch_size=1 时维度消失
        if layer == 'penultimate':
            x = self.penultimate_layer(x)
            return torch.flatten(x, 1) # [Batch, 2048]
        else:
            raise ValueError(f"不支持的层: {layer}")
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu') # 添加 map_location
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('resnext.'):
                    new_state_dict[k[8:]] = v
                else:
                    new_state_dict[k] = v
            self.resnext.load_state_dict(new_state_dict, strict=False)
            print("ResNeXt模型权重加载成功！")
        except Exception as e:
            print(f"加载ResNeXt模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ResNeXt模型权重: {e}")

# ConvNext Large特征提取器
class CustomConvNextLargeFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomConvNextLargeFeatureExtractor, self).__init__()
        self.convnext = models.convnext_large(pretrained=False)
        
        # 修改分类层
        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)
        
        self.load_weights(model_path)
        
        # 修正：ConvNeXt 的 classifier 包含 [0]LayerNorm2d -> [1]Flatten -> [2]Linear
        # 我们只替换最后的 Linear，保留 LayerNorm 和 Flatten
        self.classifier_head = self.convnext.classifier[2]
        self.convnext.classifier[2] = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 官方 forward: features -> avgpool -> classifier(norm->flatten->identity)
        features = self.convnext(x)
        
        if layer == 'penultimate':
            return features # [Batch, 1536]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('convnext.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.convnext.load_state_dict(new_state_dict, strict=False)
            else:
                self.convnext.load_state_dict(state_dict, strict=False)
            print("ConvNext模型权重加载成功！")
        except Exception as e:
            print(f"加载ConvNext模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ConvNext模型权重: {e}")

# GoogLeNet特征提取器
class CustomGoogLeNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomGoogLeNetFeatureExtractor, self).__init__()
        
        # 1. 加载标准模型
        self.googlenet = models.googlenet(pretrained=False, aux_logits=False)
        
        # 2. 调整 FC 层以匹配你训练时的结构 (为了能正确加载权重)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重 (必须在修改结构后，替换 Identity 前进行)
        self.load_weights(model_path)
        
        # 4. 【关键步骤】分离分类头和特征提取器
        # 将加载好权重的 FC 层保存下来，用于 'final' 模式
        self.classifier_head = self.googlenet.fc
        
        # 将模型内部的 FC 替换为 Identity (空操作)
        # 这样调用 self.googlenet(x) 时，输出的就是 FC 之前的 1024 维特征
        self.googlenet.fc = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 调用官方的 forward 流程
        # 因为 fc 变成了 Identity，所以这里返回的是 [Batch, 1024] 的特征
        # 注意：GoogLeNet 官方 forward 包含 Dropout，提取特征时务必使用 model.eval()
        features = self.googlenet(x)
        
        if layer == 'penultimate':
            return features
        else:
            # 如果需要分类结果，手动通过保存的 FC 层
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            # 增加 map_location 以防在无 GPU 环境报错
            state_dict = torch.load(model_path, map_location='cpu')
            
            if all(k.startswith('googlenet.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.googlenet.load_state_dict(new_state_dict, strict=False)
                print("GoogLeNet模型权重加载成功（已处理'googlenet.'前缀）！")
            else:
                self.googlenet.load_state_dict(state_dict, strict=False)
                print("GoogLeNet模型权重加载成功！")
        except Exception as e:
            print(f"加载GoogLeNet模型权重时出错: {e}")
            raise RuntimeError(f"无法加载GoogLeNet模型权重: {e}")

# ResNet50特征提取器
class CustomResNet50FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNet50FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        # 移除最后的全连接层以获取特征
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('resnet.') for k in state_dict.keys()):
                # 移除'resnet.'前缀
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.resnet.load_state_dict(new_state_dict, strict=False)
                print("ResNet50模型权重加载成功（已处理'resnet.'前缀）！")
            else:
                # 直接加载
                self.resnet.load_state_dict(state_dict, strict=False)
                print("ResNet50模型权重加载成功！")
        except Exception as e:
            print(f"加载ResNet50模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ResNet50模型权重: {e}")

# ResNet101特征提取器
class CustomResNet101FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomResNet101FeatureExtractor, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        # 移除最后的全连接层以获取特征
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('resnet.') for k in state_dict.keys()):
                # 移除'resnet.'前缀
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.resnet.load_state_dict(new_state_dict, strict=False)
                print("ResNet101模型权重加载成功（已处理'resnet.'前缀）！")
            else:
                # 直接加载
                self.resnet.load_state_dict(state_dict, strict=False)
                print("ResNet101模型权重加载成功！")
        except Exception as e:
            print(f"加载ResNet101模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ResNet101模型权重: {e}")

# DenseNet121特征提取器
class CustomDenseNet121FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomDenseNet121FeatureExtractor, self).__init__()
        # 1. 初始化模型
        self.densenet = models.densenet121(pretrained=False)
        
        # 2. 修改分类层结构以匹配权重文件
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重 (必须在替换 Identity 之前)
        self.load_weights(model_path)
        
        # 4. 保存分类头用于 'final' 模式，并将原位置换成 Identity
        self.classifier_head = self.densenet.classifier
        self.densenet.classifier = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 直接调用 densenet 的 forward
        # 流程: features -> relu -> avgpool -> flatten -> identity
        features = self.densenet(x)
        
        if layer == 'penultimate':
            return features # 返回 1024 维特征
        else:
            return self.classifier_head(features) # 返回 logits
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('densenet.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.densenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.densenet.load_state_dict(state_dict, strict=False)
            print("DenseNet121模型权重加载成功！")
        except Exception as e:
            print(f"加载DenseNet121模型权重时出错: {e}")
            raise RuntimeError(f"无法加载DenseNet121模型权重: {e}")

# DenseNet161特征提取器
class CustomDenseNet161FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomDenseNet161FeatureExtractor, self).__init__()
        # 1. 初始化模型
        self.densenet = models.densenet161(pretrained=False)
        
        # 2. 修改分类层结构
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重
        self.load_weights(model_path)
        
        # 4. 分离分类头
        self.classifier_head = self.densenet.classifier
        self.densenet.classifier = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 流程: features -> relu -> avgpool -> flatten -> identity
        features = self.densenet(x)
        
        if layer == 'penultimate':
            return features # 返回 2208 维特征
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('densenet.') for k in state_dict.keys()):
                new_state_dict = {k[9:]: v for k, v in state_dict.items()}
                self.densenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.densenet.load_state_dict(state_dict, strict=False)
            print("DenseNet161模型权重加载成功！")
        except Exception as e:
            print(f"加载DenseNet161模型权重时出错: {e}")
            raise RuntimeError(f"无法加载DenseNet161模型权重: {e}")

# VGG16特征提取器
class CustomVGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomVGG16FeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        
        # 1. 修改分类层以匹配权重文件结构
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
        
        # 2. 加载权重
        self.load_weights(model_path)
        
        # 3. 【关键修改】重构分类器
        # VGG的classifier结构: 
        # [0]Linear -> [1]ReLU -> [2]Dropout -> [3]Linear -> [4]ReLU -> [5]Dropout -> [6]Linear(分类)
        # 我们需要保留前6层 (0-5)，这样输出就是 4096 维
        self.classifier_head = self.vgg.classifier[6] # 保存最后一层备用
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
    
    def forward(self, x, layer='penultimate'):
        # 直接调用 vgg(x)，因为它现在只包含到倒数第二层的逻辑
        features = self.vgg(x)
        
        if layer == 'penultimate':
            return features # 输出 [Batch, 4096]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('vgg.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vgg.load_state_dict(new_state_dict, strict=False)
            else:
                self.vgg.load_state_dict(state_dict, strict=False)
            print("VGG16模型权重加载成功！")
        except Exception as e:
            print(f"加载VGG16模型权重时出错: {e}")
            raise RuntimeError(f"无法加载VGG16模型权重: {e}")

# InceptionV3特征提取器
class CustomInceptionV3FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomInceptionV3FeatureExtractor, self).__init__()
        # 1. 初始化模型
        self.inception = models.inception_v3(pretrained=False, aux_logits=False)
        
        # 2. 修改分类层以匹配权重
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重
        self.load_weights(model_path)
        
        # 4. 【关键修改】替换 FC 层为直通层
        self.classifier_head = self.inception.fc
        self.inception.fc = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # InceptionV3 对输入尺寸敏感，通常建议 299x299
        # 如果 aux_logits=False，它直接返回 Tensor
        # 流程: InceptionBlocks -> AvgPool -> Dropout -> Flatten -> Identity
        features = self.inception(x)
        
        if layer == 'penultimate':
            return features # 输出 [Batch, 2048]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('inception.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.inception.load_state_dict(new_state_dict, strict=False)
            else:
                self.inception.load_state_dict(state_dict, strict=False)
            print("InceptionV3模型权重加载成功！")
        except Exception as e:
            print(f"加载InceptionV3模型权重时出错: {e}")
            raise RuntimeError(f"无法加载InceptionV3模型权重: {e}")

# EfficientNetB0特征提取器
class CustomEfficientNetB0FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomEfficientNetB0FeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 移除最后的分类层以获取特征
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('efficientnet.') for k in state_dict.keys()):
                # 移除'efficientnet.'前缀
                new_state_dict = {k[13:]: v for k, v in state_dict.items()}
                self.efficientnet.load_state_dict(new_state_dict, strict=False)
                print("EfficientNetB0模型权重加载成功（已处理'efficientnet.'前缀）！")
            else:
                # 直接加载
                self.efficientnet.load_state_dict(state_dict, strict=False)
                print("EfficientNetB0模型权重加载成功！")
        except Exception as e:
            print(f"加载EfficientNetB0模型权重时出错: {e}")
            raise RuntimeError(f"无法加载EfficientNetB0模型权重: {e}")

# EfficientNet_v2_l特征提取器
class CustomEfficientNetV2LargeFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomEfficientNetV2LargeFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_v2_l(pretrained=False)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 移除最后的分类层以获取特征
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('efficientnet.') for k in state_dict.keys()):
                # 移除'efficientnet.'前缀
                new_state_dict = {k[13:]: v for k, v in state_dict.items()}
                self.efficientnet.load_state_dict(new_state_dict, strict=False)
                print("EfficientNetV2L模型权重加载成功（已处理'efficientnet.'前缀）！")
            else:
                # 直接加载
                self.efficientnet.load_state_dict(state_dict, strict=False)
                print("EfficientNetV2L模型权重加载成功！")
        except Exception as e:
            print(f"加载EfficientNetV2L模型权重时出错: {e}")
            raise RuntimeError(f"无法加载EfficientNetV2L模型权重: {e}")

# MobileNetV2特征提取器
class CustomMobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomMobileNetV2FeatureExtractor, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        
        # 1. 修改分类层以匹配权重
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 2. 加载权重
        self.load_weights(model_path)
        
        # 3. 分离分类头
        # MobileNetV2 classifier: [0]Dropout -> [1]Linear
        # 我们需要的是进入 classifier 之前的特征 (即池化后的 1280 维)
        # 注意：MobileNetV2 的 forward 逻辑是 features -> avgpool -> flatten -> classifier
        # 所以我们将 classifier 替换为 Identity 即可
        self.classifier_head = self.mobilenet.classifier
        self.mobilenet.classifier = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 官方 forward: features -> adaptive_avg_pool2d((1,1)) -> flatten -> classifier
        features = self.mobilenet(x)
        
        if layer == 'penultimate':
            return features # [Batch, 1280]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('mobilenet.') for k in state_dict.keys()):
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.mobilenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.mobilenet.load_state_dict(state_dict, strict=False)
            print("MobileNetV2模型权重加载成功！")
        except Exception as e:
            print(f"加载MobileNetV2模型权重时出错: {e}")
            raise RuntimeError(f"无法加载MobileNetV2模型权重: {e}")

# MobileNetV3特征提取器
class CustomMobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomMobileNetV3FeatureExtractor, self).__init__()
        # 加载预训练的MobileNetV3 Large模型
        self.mobilenet = models.mobilenet_v3_large(pretrained=False)
        
        # 获取原始分类器
        original_classifier = self.mobilenet.classifier
        
        # 只替换最后一层（输出层）
        if isinstance(original_classifier, nn.Sequential):
            layers = list(original_classifier.children())
            # 假设最后一层是线性层
            if isinstance(layers[-1], nn.Linear):
                in_features = layers[-1].in_features
                # 创建新的分类器，保留前面所有层，只替换最后一层
                new_layers = layers[:-1] + [nn.Linear(in_features, num_classes)]
                self.mobilenet.classifier = nn.Sequential(*new_layers)
            else:
                # 如果最后一层不是线性层，使用默认方法
                last_channel = 1280  # MobileNetV3 Large的默认通道数
                self.mobilenet.classifier = nn.Sequential(
                    nn.Linear(last_channel, 1024),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1024, num_classes)
                )
        else:
            # 如果分类器不是Sequential，使用默认方法
            last_channel = 1280  # MobileNetV3 Large的默认通道数
            self.mobilenet.classifier = nn.Sequential(
                nn.Linear(last_channel, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )
        
        # 提取特征部分
        self.features = self.mobilenet.features
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer='penultimate'):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('mobilenet.') for k in state_dict.keys()):
                # 移除'mobilenet.'前缀
                new_state_dict = {k[10:]: v for k, v in state_dict.items()}
                self.mobilenet.load_state_dict(new_state_dict, strict=False)
                print("MobileNetV3模型权重加载成功（已处理'mobilenet.'前缀）！")
            else:
                # 直接加载
                self.mobilenet.load_state_dict(state_dict, strict=False)
                print("MobileNetV3模型权重加载成功！")
        except Exception as e:
            print(f"加载MobileNetV3模型权重时出错: {e}")
            raise RuntimeError(f"无法加载MobileNetV3模型权重: {e}")

# RegNet特征提取器
class CustomRegNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomRegNetFeatureExtractor, self).__init__()
        # 1. 初始化模型
        self.regnet = models.regnet_y_400mf(pretrained=False)
        
        # 2. 修改分类层以匹配权重
        in_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重 (必须在替换 Identity 之前)
        self.load_weights(model_path)
        
        # 4. 【关键修改】分离分类头
        # RegNet forward: stem -> trunk -> avgpool -> flatten -> fc
        # 替换 fc 为 Identity 即可直接获取 flatten 后的特征
        self.classifier_head = self.regnet.fc
        self.regnet.fc = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        # 直接调用官方 forward，安全且准确
        features = self.regnet(x)
        
        if layer == 'penultimate':
            return features # [Batch, 440]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('regnet.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.regnet.load_state_dict(new_state_dict, strict=False)
            else:
                self.regnet.load_state_dict(state_dict, strict=False)
            print("RegNet模型权重加载成功！")
        except Exception as e:
            print(f"加载RegNet模型权重时出错: {e}")
            raise RuntimeError(f"无法加载RegNet模型权重: {e}")

# ShuffleNetV2特征提取器
class CustomShuffleNetV2FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomShuffleNetV2FeatureExtractor, self).__init__()
        # 1. 初始化模型
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
        
        # 2. 修改分类层
        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Linear(in_features, num_classes)
        
        # 3. 加载权重
        self.load_weights(model_path)
        
        # 4. 【关键修改】分离分类头
        # ShuffleNetV2 forward: ... -> conv5 -> mean([2,3]) -> fc
        # 这里的 mean([2,3]) 是函数式调用，不在 children() 里
        # 只有通过官方 forward 才能执行这一步。
        # 将 fc 替换为 Identity，输出的就是 mean 之后的 1024 维向量
        self.classifier_head = self.shufflenet.fc
        self.shufflenet.fc = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        features = self.shufflenet(x)
        
        if layer == 'penultimate':
            return features # [Batch, 1024]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if all(k.startswith('shufflenet.') for k in state_dict.keys()):
                new_state_dict = {k[11:]: v for k, v in state_dict.items()}
                self.shufflenet.load_state_dict(new_state_dict, strict=False)
            else:
                self.shufflenet.load_state_dict(state_dict, strict=False)
            print("ShuffleNetV2模型权重加载成功！")
        except Exception as e:
            print(f"加载ShuffleNetV2模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ShuffleNetV2模型权重: {e}")
        
# SqueezeNet特征提取器
class CustomSqueezeNetFeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomSqueezeNetFeatureExtractor, self).__init__()
        
        # 1. 预加载权重并自动识别版本
        state_dict, version = self._detect_version_and_load_dict(model_path)
        
        # 2. 根据识别到的版本实例化模型
        if version == '1.1':
            print("检测到权重版本为 SqueezeNet 1.1，正在初始化...")
            self.squeezenet = models.squeezenet1_1(weights=None)
        else:
            print("检测到权重版本为 SqueezeNet 1.0，正在初始化...")
            self.squeezenet = models.squeezenet1_0(weights=None)
        
        # 3. 修改分类器结构以匹配可能的训练结构 (防止加载权重时报错)
        # SqueezeNet 的分类器是卷积层而不是全连接层
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 4. 加载权重
        self.load_weights(model_path)
        
        # 5. 准备特征提取
        # SqueezeNet 的 features 部分输出 512 通道
        # 我们不需要 classifier 部分，因为那是用来降维到 num_classes 的
        self.features_block = self.squeezenet.features
        
        # 保存分类头用于 'final' 模式 (虽然 SqueezeNet 结构特殊，这里模拟一下)
        self.classifier_head = self.squeezenet.classifier

    # 检测是1.0还是1.1版本的辅助函数（可选）    
    def _detect_version_and_load_dict(self, model_path):
        """内部辅助函数：通过检查权重形状自动判定版本"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 处理可能存在的前缀
            if all(k.startswith('squeezenet.') for k in state_dict.keys()):
                state_dict = {k[11:]: v for k, v in state_dict.items()}
            
            # 核心判断逻辑：检查第一层卷积 (features.0.weight) 的形状
            # v1.0 形状为 [96, 3, 7, 7]
            # v1.1 形状为 [64, 3, 3, 3]
            first_layer_shape = state_dict['features.0.weight'].shape
            
            if first_layer_shape[2] == 3: # 卷积核大小为 3x3
                version = '1.1'
            else: # 卷积核大小为 7x7
                version = '1.0'
                
            return state_dict, version
            
        except Exception as e:
            raise RuntimeError(f"解析权重文件失败，请检查路径或文件格式: {e}")

    def forward(self, x, layer='penultimate'):
        # 1. 提取卷积特征
        x = self.features_block(x)
        
        # 2. 全局平均池化
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        # 3. 展平
        x = torch.flatten(x, 1)
        
        if layer == 'penultimate':
            # 返回 512 维特征
            return x
        else:
            # 如果需要分类结果，需要手动通过 classifier
            # 注意：SqueezeNet 的 classifier 包含 AvgPool，这里逻辑比较特殊
            # 简单起见，如果只是提取特征，上面返回 x 即可
            # 如果必须返回 logits，建议直接调用 self.squeezenet(original_input)
            return self.classifier_head(x.unsqueeze(2).unsqueeze(3)).flatten(1)
        
    
    def load_weights(self, model_path):
        """加载模型权重"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 处理前缀
            if all(k.startswith('squeezenet.') for k in state_dict.keys()):
                new_state_dict = {k[11:]: v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
            
            # 加载权重
            self.squeezenet.load_state_dict(new_state_dict, strict=False)
            print(f"SqueezeNet模型权重加载成功！")

        except Exception as e:
            print(f"加载SqueezeNet模型权重时出错: {e}")
            raise RuntimeError(f"无法加载SqueezeNet模型权重: {e}")
        
# ViT_B16特征提取器
class CustomViT_B16FeatureExtractor(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomViT_B16FeatureExtractor, self).__init__()
        self.vit = models.vit_b_16(pretrained=False)
        
        # 修改分类层
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
        self.load_weights(model_path)
        
        # 修正：直接替换 heads 模块为 Identity
        # Torchvision ViT forward: x -> encoder -> select_class_token -> heads
        # 替换 heads 后，直接输出 class token 的特征
        self.classifier_head = self.vit.heads
        self.vit.heads = nn.Identity()
    
    def forward(self, x, layer='penultimate'):
        features = self.vit(x)
        
        if layer == 'penultimate':
            return features # [Batch, 768]
        else:
            return self.classifier_head(features)
    
    def load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu') # 添加 map_location
            if all(k.startswith('vit.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vit.load_state_dict(new_state_dict, strict=False)
            else:
                self.vit.load_state_dict(state_dict, strict=False)
            print("ViT-B16模型权重加载成功！")
        except Exception as e:
            print(f"加载ViT-B16模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ViT-B16模型权重: {e}")

# ViT_L16特征提取器 (逻辑同上)
class CustomViT_L16FeatureExtractor(nn.Module):
    # def __init__(self, model_path, num_classes=2):
    #     super(CustomViT_L16FeatureExtractor, self).__init__()
    #     self.vit = models.vit_l_16(pretrained=False)
        
    #     in_features = self.vit.heads.head.in_features
    #     self.vit.heads.head = nn.Linear(in_features, num_classes)
        
    #     self.load_weights(model_path)
        
    #     self.classifier_head = self.vit.heads
    #     self.vit.heads = nn.Identity()
    
    # def forward(self, x, layer='penultimate'):
    #     features = self.vit(x)
    #     if layer == 'penultimate':
    #         return features # [Batch, 1024]
    #     else:
    #         return self.classifier_head(features)
    
    # def load_weights(self, model_path):
    #     try:
    #         state_dict = torch.load(model_path, map_location='cpu')
    #         if all(k.startswith('vit.') for k in state_dict.keys()):
    #             new_state_dict = {k[4:]: v for k, v in state_dict.items()}
    #             self.vit.load_state_dict(new_state_dict, strict=False)
    #         else:
    #             self.vit.load_state_dict(state_dict, strict=False)
    #         print("ViT-L16模型权重加载成功！")
    #     except Exception as e:
    #         print(f"加载ViT-L16模型权重时出错: {e}")
    #         raise RuntimeError(f"无法加载ViT-L16模型权重: {e}")
    def __init__(self, model_path, num_classes=2):
        super(CustomViT_L16FeatureExtractor, self).__init__()
        # 加载预训练的ViT-L16模型
        self.vit = models.vit_l_16(pretrained=False)
        
        # 修改最后的分类器以适应二分类任务
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
        # 加载训练好的权重
        self.load_weights(model_path)
    
    def forward(self, x, layer=None):
        """
        前向传播，提取最后一层transformer的class token特征
        
        参数:
        x: 输入张量
        layer: 保留参数，为了与其他特征提取器接口一致，但不使用
        """
        # 存储特征的变量
        class_token_feature = None
        
        def hook_fn(module, input, output):
            nonlocal class_token_feature
            # 获取最后一层transformer输出的class token
            class_token_feature = output[:, 0]
        
        # 找到encoder的最后一层
        last_layer = self.vit.encoder.layers[-1]
        
        # 临时注册一个前向钩子到最后一层
        hook = last_layer.register_forward_hook(hook_fn)
        
        # 执行前向传播
        _ = self.vit(x)
        
        # 移除钩子
        hook.remove()
        
        # 返回class token特征
        return class_token_feature
    
    def load_weights(self, model_path):
        """加载模型权重，处理键名前缀不匹配的问题"""
        try:
            state_dict = torch.load(model_path)
            
            # 检查是否需要处理键名前缀
            if all(k.startswith('vit.') for k in state_dict.keys()):
                # 移除'vit.'前缀
                new_state_dict = {k[4:]: v for k, v in state_dict.items()}
                self.vit.load_state_dict(new_state_dict)
                print("ViT模型权重加载成功（已处理'vit.'前缀）！")
            else:
                # 直接加载
                self.vit.load_state_dict(state_dict)
                print("ViT模型权重加载成功！")
        except Exception as e:
            print(f"加载ViT模型权重时出错: {e}")
            raise RuntimeError(f"无法加载ViT模型权重: {e}")
    
   
# 图片大小统一化（for Inception v3, 输入大小为 224x224）
def pad_to_square_transform(img):
    try:
        # 获取原始图像的大小
        width, height = img.size
        
        # 只有当图像不是正方形时才进行填充
        if width != height:
            # 计算填充后的目标大小
            max_side = max(width, height)
            
            # 创建一个新的白色背景图像
            new_image = Image.new('RGB', (max_side, max_side), (255, 255, 255))
            
            # 将原始图像粘贴到新图像的中心
            new_image.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
            
            # 调整到224x224大小
            return new_image.resize((224, 224), Image.LANCZOS)
        else:
            # 直接调整大小
            return img.resize((224, 224), Image.LANCZOS)
        
    except Exception as e:
        print(f"Error in pad_to_square_transform: {e}")
        # 返回一个默认图像
        return Image.new('RGB', (224, 224), (255, 255, 255))

# 图片大小统一化（for Inception v3, 输入大小为 299x299）
def pad_to_square_299_transform(img):
    try:
        # 获取原始图像的大小
        width, height = img.size
        
        # 计算填充后的目标大小
        max_side = max(width, height)
        
        # 创建一个新的白色背景图像
        new_image = Image.new('RGB', (max_side, max_side), (255, 255, 255))
        
        # 将原始图像粘贴到新图像的中心
        new_image.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
        
        # 调整到 299x299 大小，保持宽高比 (Inception V3 使用 299x299 输入)
        new_image = new_image.resize((299, 299), Image.LANCZOS)
        
        return new_image
    except Exception as e:
        print(f"Error in pad_to_square_299_transform: {e}")
        # 返回一个默认图像
        return Image.new('RGB', (299, 299), (255, 255, 255))

# 定义支持的模型映射
SUPPORTED_MODELS = {
    'AlexNet': {'class': CustomAlexNetFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNeXt': {'class': CustomResNeXtFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ConvNextLarge': {'class': CustomConvNextLargeFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'GoogLeNet': {'class': CustomGoogLeNetFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNet50': {'class': CustomResNet50FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ResNet101': {'class': CustomResNet101FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'DenseNet121': {'class': CustomDenseNet121FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'DenseNet161': {'class': CustomDenseNet161FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'VGG16': {'class': CustomVGG16FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'InceptionV3': {'class': CustomInceptionV3FeatureExtractor, 'input_size': 299, 'pad_func': pad_to_square_299_transform},
    'EfficientNetB0': {'class': CustomEfficientNetB0FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'EfficientNetV2Large': {'class': CustomEfficientNetV2LargeFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'MobileNetV2': {'class': CustomMobileNetV2FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'MobileNetV3': {'class': CustomMobileNetV3FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'RegNet': {'class': CustomRegNetFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ShuffleNetV2': {'class': CustomShuffleNetV2FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'SqueezeNet': {'class': CustomSqueezeNetFeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ViT_B16': {'class': CustomViT_B16FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform},
    'ViT_L16': {'class': CustomViT_L16FeatureExtractor, 'input_size': 224, 'pad_func': pad_to_square_transform}
}

def get_model_and_transform(model_type, device, model_path=None, mean=None, std=None, num_classes=2):
    """
    工厂函数：获取模型、转换函数和归一化参数

    参数:
        model_type  : 模型名称字符串，需在 SUPPORTED_MODELS 中
        device      : torch.device，cuda 或 cpu
        model_path  : 模型权重文件路径
        mean        : 归一化均值列表，默认 ImageNet 均值
        std         : 归一化标准差列表，默认 ImageNet 标准差
        num_classes : 【新增】类别数，默认 2，支持任意多分类
                      调用时传入 len(cancer_dirs) 即可自动适配
    """

    # 1. 定义均值和方差
    if model_path is None or (mean is None and std is None):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = mean
        std = std
        
    # 2. 检查模型类型是否支持
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的模型类型: {list(SUPPORTED_MODELS.keys())}")
    
    model_info = SUPPORTED_MODELS[model_type]
    model_class = model_info['class']
    input_size = model_info['input_size']
    pad_func = model_info['pad_func']
    
    # 3. 初始化模型
    # 【核心改动】原版未传 num_classes，导致多分类权重无法加载
    # 原: model = model_class(model_path=model_path)
    # 新: model = model_class(model_path=model_path, num_classes=num_classes)
    model = model_class(model_path=model_path, num_classes=num_classes)
    model = model.to(device)

    # 4. 定义Transform
    transform = transforms.Compose([
        transforms.Lambda(pad_func),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, transform, input_size
