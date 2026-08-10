import torch.nn as nn
from torchvision import models
import traceback

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomAlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.alexnet(x)


class CustomResNeXt(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNeXt, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=True)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnext(x)


class CustomConvNextLarge(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomConvNextLarge, self).__init__()
        self.convnext = models.convnext_large(pretrained=True)
        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.convnext(x)


class CustomGoogLeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomGoogLeNet, self).__init__()
        # 加载预训练的GoogLeNet模型
        self.googlenet = models.googlenet(pretrained=True,aux_logits=True)
        
        # 修改最后的分类器以适应二分类任务
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

        # 修改辅助分类器
        if hasattr(self.googlenet, 'aux1'):
            print("self.googlenet.aux1:",self.googlenet.aux1)
            in_features_aux1 = self.googlenet.aux1.fc2.in_features
            self.googlenet.aux1.fc2 = nn.Linear(in_features_aux1, num_classes)
        
        if hasattr(self.googlenet, 'aux2'):
            in_features_aux2 = self.googlenet.aux2.fc2.in_features
            self.googlenet.aux2.fc2 = nn.Linear(in_features_aux2, num_classes)
    
    def forward(self, x):
        return self.googlenet(x)  # 返回所有输出，包括辅助分类器


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class CustomResNet101(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomDenseNet121, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)


class CustomDenseNet161(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomDenseNet161, self).__init__()
        self.densenet = models.densenet161(pretrained=True)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)


class CustomVGG16(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomVGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vgg(x)


class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomInceptionV3, self).__init__()
        # 加载预训练的Inception V3模型
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        
        # 修改最后的分类器以适应二分类任务
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

        # 修改辅助分类器
        if hasattr(self.inception, 'AuxLogits'):
            print("self.inception.AuxLogits:",self.inception.AuxLogits)
            in_features_AuxLogits = self.inception.AuxLogits.fc.in_features
            self.inception.AuxLogits.fc = nn.Linear(in_features_AuxLogits, num_classes)
        
    
    def forward(self, x):
        return self.inception(x)  # 返回所有输出，包括辅助分类器


class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomEfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


class CustomEfficientNet_v2_l(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomEfficientNet_v2_l, self).__init__()
        self.efficientnet = models.efficientnet_v2_l(pretrained=True)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomMobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x)


class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomMobileNetV3, self).__init__()
        # 加载预训练的MobileNetV3 Large模型
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        
        # 获取原始分类器
        original_classifier = self.mobilenet.classifier
        
        # 只替换最后一层（输出层）
        # MobileNetV3的分类器通常是一个包含多个层的Sequential
        # 我们需要保留除了最后一层之外的所有层
        if isinstance(original_classifier, nn.Sequential):
            layers = list(original_classifier.children())
            # 假设最后一层是线性层
            if isinstance(layers[-1], nn.Linear):
                in_features = layers[-1].in_features
                # 创建新的分类器，保留前面所有层，只替换最后一层
                new_layers = layers[:-1] + [nn.Linear(in_features, num_classes)]
                self.mobilenet.classifier = nn.Sequential(*new_layers)
            else:
                # 如果最后一层不是线性层，打印警告并使用默认方法
                print("警告: 无法识别MobileNetV3分类器的最后一层，使用完全替换")
                last_channel = 1280  # MobileNetV3 Large的默认通道数
                self.mobilenet.classifier = nn.Sequential(
                    nn.Linear(last_channel, 1024),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1024, num_classes)
                )
        else:
            # 如果分类器不是Sequential，打印警告并使用完全替换
            print("警告: MobileNetV3分类器不是Sequential类型，使用完全替换")
            last_channel = 1280  # MobileNetV3 Large的默认通道数
            self.mobilenet.classifier = nn.Sequential(
                nn.Linear(last_channel, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )
    
    def forward(self, x):
        return self.mobilenet(x)


class CustomRegNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomRegNet, self).__init__()
        self.regnet = models.regnet_y_400mf(pretrained=True)
        in_features = self.regnet.fc.in_features
        self.regnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.regnet(x)


class CustomShuffleNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomShuffleNetV2, self).__init__()
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.shufflenet(x)


class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomSqueezeNet, self).__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=True)
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.squeezenet.num_classes = num_classes
    
    def forward(self, x):
        return self.squeezenet(x)


class CustomViT_L16(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomViT_L16, self).__init__()
        self.vit = models.vit_l_16(pretrained=True)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)


class CustomViT_B16(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomViT_B16, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# ==================== 模型创建函数 ====================
def create_model(model_name, num_classes=5):
    """
    根据模型名称创建对应的模型实例
    
    参数:
    model_name: 模型名称（字符串）
    num_classes: 类别数量
    
    返回:
    模型实例
    """
    model_name_lower = model_name.lower()
    
    if 'alexnet' in model_name_lower:
        return CustomAlexNet(num_classes)
    elif 'resnext' in model_name_lower:
        return CustomResNeXt(num_classes)
    elif 'convnext' in model_name_lower or 'convnextlarge' in model_name_lower:
        return CustomConvNextLarge(num_classes)
    elif 'googlenet' in model_name_lower:
        return CustomGoogLeNet(num_classes)
    elif 'resnet50' in model_name_lower:
        return CustomResNet50(num_classes)
    elif 'resnet101' in model_name_lower:
        return CustomResNet101(num_classes)
    elif 'densenet121' in model_name_lower:
        return CustomDenseNet121(num_classes)
    elif 'densenet161' in model_name_lower:
        return CustomDenseNet161(num_classes)
    elif 'vgg16' in model_name_lower or 'vgg' in model_name_lower:
        return CustomVGG16(num_classes)
    elif 'inceptionv3' in model_name_lower or ('inception' in model_name_lower and 'v3' in model_name_lower):
        return CustomInceptionV3(num_classes)
    elif 'efficientnetv2' in model_name_lower or 'efficientnet_v2' in model_name_lower:
        return CustomEfficientNet_v2_l(num_classes)
    elif 'efficientnetb0' in model_name_lower or 'efficientnet_b0' in model_name_lower:
        return CustomEfficientNetB0(num_classes)
    elif 'mobilenetv3' in model_name_lower or 'mobilenet_v3' in model_name_lower:
        return CustomMobileNetV3(num_classes)
    elif 'mobilenetv2' in model_name_lower or 'mobilenet_v2' in model_name_lower:
        return CustomMobileNetV2(num_classes)
    elif 'regnet' in model_name_lower:
        return CustomRegNet(num_classes)
    elif 'shufflenet' in model_name_lower:
        return CustomShuffleNetV2(num_classes)
    elif 'squeezenet' in model_name_lower:
        return CustomSqueezeNet(num_classes)
    elif 'vit_l' in model_name_lower or 'vitl' in model_name_lower:
        return CustomViT_L16(num_classes)
    elif 'vit_b' in model_name_lower or 'vitb' in model_name_lower:
        return CustomViT_B16(num_classes)
    else:
        print(f"警告: 未知的模型名称 '{model_name}'，使用默认的 ResNeXt")
        return CustomResNeXt(num_classes)


# ==================== 模型信息函数 ====================
def get_model_info(model_name):
    """
    获取模型的基本信息
    
    参数:
    model_name: 模型名称
    
    返回:
    包含模型信息的字典
    """
    model = create_model(model_name)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_size': 299 if 'Inception' in model_name else 224
    }

# 支持的模型列表
SUPPORTED_MODELS = [
    'AlexNet', 'ResNeXt', 'ConvNextLarge', 'GoogLeNet',
    'ResNet50', 'ResNet101', 'DenseNet121', 'DenseNet161',
    'VGG16', 'InceptionV3', 'EfficientNetB0', 'EfficientNetV2Large',
    'MobileNetV2', 'MobileNetV3', 'RegNet', 'ShuffleNetV2',
    'SqueezeNet', 'ViT_B16', 'ViT_L16'
]

# ==================== 测试函数 ====================
if __name__ == "__main__":
    # 测试所有模型是否能正常创建
    model_list = [
        'AlexNet', 'ResNeXt', 'ConvNextLarge', 'GoogLeNet',
        'ResNet50', 'ResNet101', 'DenseNet121', 'DenseNet161',
        'VGG16', 'InceptionV3', 'EfficientNetB0', 'EfficientNetV2Large',
        'MobileNetV2', 'MobileNetV3', 'RegNet', 'ShuffleNetV2',
        'SqueezeNet', 'ViT_B16', 'ViT_L16'
    ]
    
    print("测试所有模型创建...")
    for model_name in model_list:
        try:
            model = create_model(model_name)
            info = get_model_info(model_name)
            print(f"✅ {model_name}: {info['total_params']:,} 参数, 输入尺寸: {info['input_size']}x{info['input_size']}")
        except Exception as e:
            print(f"❌ {model_name}: {e}")