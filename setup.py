"""
Setup configuration for CoVALend package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="covalend",
    version="1.0.0",
    author="Hongyi Zhou",
    author_email="h.zhou.23@abdn.ac.uk",
    description="A Value-Alignment Pipeline for Compliant Micro-Lending with Minimal-Intrusion Tabular-Text Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jackksonns/CoVALend",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "flash-attn": [
            "flash-attn>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "covalend-train=scripts.train:main",
            "covalend-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
