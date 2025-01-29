from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="iiif-download",
    version="0.1.10",
    author="Segolene-Albouy",
    author_email="segolene.albouy@gmail.com",
    description="A Python package to download images from IIIF manifests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Segolene-Albouy/iiif-download",
    project_urls={
        "Bug Tracker": "https://github.com/Segolene-Albouy/iiif-download/issues",
        "Documentation": "https://github.com/Segolene-Albouy/iiif-download#readme",
        "Source Code": "https://github.com/Segolene-Albouy/iiif-download",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=11.0.0",
        "requests>=2.32.0",
        "tqdm>=4.66.0",
        "urllib3>=2.2.0",
        "aiohttp>=3.10",
        "aiofiles>=24.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "pytest-asyncio>=0.25.0",
            "flake8>=7.0.0",
            "black>=24.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iiif-download=iiif_download.cli:main",
        ],
    },
)
