# setup.py
import os
from setuptools import setup, find_packages

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as fh:
            return fh.read()
    return "Cross-domain cluster matching pipeline for cell image analysis."

setup(
    name="CrossClusterMatching_pipeline",  # 给你的包起个名字
    version="1.0.0",
    author="QRR",
    author_email="your.email@example.com",
    description="A cross-domain cluster matching pipeline for analyzing cell images",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CrossClusterMatching_pipeline",  # 替换为你的项目URL
    # packages=find_packages(where='cross_cluster_matching'),       # 自动寻找 src 目录下的所有包
    # package_dir={'': 'cross_cluster_matching'},
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'opencv-python>=4.5.0',
        'Pillow>=8.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'tqdm>=4.60.0',
        'umap-learn>=0.5.0',
        'scipy>=1.7.0',
    ],
)