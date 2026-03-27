from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    description="Ultra-efficient post-training quantization for LLMs",
    author="Kubenew",
    author_email="kubenew@example.com",
    url="https://github.com/Kubenew/TurboQuant-v3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "hf": [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
