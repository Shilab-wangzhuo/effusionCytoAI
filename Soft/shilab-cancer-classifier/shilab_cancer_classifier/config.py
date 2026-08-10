"""Shared classification configuration."""

CANCER_CLASSES = ['Breast', 'Gastrointestinal', 'Gynecologic', 'Respiratory', 'Mesothelioma']
NUM_CLASSES = len(CANCER_CLASSES)
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CANCER_CLASSES)}
IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}

