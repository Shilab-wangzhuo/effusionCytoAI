from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shilab-cancer-classifier",
    version="0.3.0",
    author="ShiLab",
    description="Tools for multi-class cancer classification of cytology cell images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "shilab-cancer-infer=shilab_cancer_classifier.infer.cancer_infer:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
