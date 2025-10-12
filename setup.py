#!/usr/bin/env python3
"""
Setup script for offload-ai package.

This script allows the offload-ai package to be installed using pip,
making it easy to distribute and install the professional file copying tools.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Professional file copying tools for DIT workflows"

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="offload-ai",
    version="1.0.0",
    author="offload-ai project",
    author_email="",
    description="Professional file copying tools for DIT workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/offload-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: System :: Archiving :: Backup",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "xxhash": [
            "xxhash>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pfndispatchcopy=pfndispatchcopy:main",
        ],
    },
    include_package_data=True,
    keywords="file-copy backup integrity verification DIT video professional",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/offload-ai/issues",
        "Source": "https://github.com/your-repo/offload-ai",
        "Documentation": "https://github.com/your-repo/offload-ai#readme",
    },
)
