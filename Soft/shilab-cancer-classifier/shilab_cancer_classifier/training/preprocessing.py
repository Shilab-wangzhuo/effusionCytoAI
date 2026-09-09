"""Model-training utilities."""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import traceback


def pad_to_square(image, fill_color=(255, 255, 255)):
    """Pad to square."""
    try:
        width, height = image.size
        max_side = max(width, height)
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        new_image = new_image.resize((224, 224), Image.LANCZOS)
        return new_image
    except Exception as e:
        print(f"Error in pad_to_square: {e}")
        print(f"Image info: {image}")
        return Image.new('RGB', (224, 224), fill_color)


def pad_to_square_transform(img):
    """Pad to square transform."""
    try:
        return pad_to_square(img)
    except Exception as e:
        print(f"Error in pad_to_square_transform: {e}")
        return Image.new('RGB', (224, 224), (255, 255, 255))


def pad_to_square_299(image, fill_color=(255, 255, 255)):
    """Pad to square 299."""
    try:
        width, height = image.size
        max_side = max(width, height)
        new_image = Image.new('RGB', (max_side, max_side), fill_color)
        new_image.paste(image, ((max_side - width) // 2, (max_side - height) // 2))
        new_image = new_image.resize((299, 299), Image.LANCZOS)
        return new_image
    except Exception as e:
        print(f"Error in pad_to_square_299: {e}")
        print(f"Image info: {image}")
        return Image.new('RGB', (299, 299), fill_color)


def pad_to_square_299_transform(img):
    """Pad to square 299 transform."""
    try:
        return pad_to_square_299(img)
    except Exception as e:
        print(f"Error in pad_to_square_299_transform: {e}")
        return Image.new('RGB', (299, 299), (255, 255, 255))


def calculate_mean_std(data_dir, model_name=None):
    """Calculate dataset normalization statistics."""
    print(f"Calculating mean and std, data directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: directory {data_dir} does not exist")
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    
    try:
        if model_name and 'Inception' in model_name:
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_299_transform),
                transforms.ToTensor()
            ])
            print(f"Using 299x299 image size for mean/std calculation ({model_name})")
        else:
            transform = transforms.Compose([
                transforms.Lambda(pad_to_square_transform),
                transforms.ToTensor()
            ])
            print("Using 224x224 image size for mean/std calculation")
        
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"Dataset loaded successfully, {len(dataset)} images in total")
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_images = 0
        
        for i, (images, _) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
            
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_size
        
        mean /= total_images
        std /= total_images
        
        print(f"Mean and std calculation complete: mean={mean}, std={std}")
        return mean, std
    
    except Exception as e:
        print(f"Error calculating mean and std: {e}")
        traceback.print_exc()
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])