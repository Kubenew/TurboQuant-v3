from setuptools import setup, find_packages
import os
import torch
from pathlib import Path

cuda_ext_include = os.path.join(os.path.dirname(__file__), "cuda", "include")
cuda_src = os.path.join(os.path.dirname(__file__), "cuda")

def get_extensions():
    extensions = []
    
    ext_source = [
        "cuda/int4_cuda.cpp",
        "cuda/int4_cuda_kernel.cuh",
    ]
    
    sources = [os.path.join(os.path.dirname(__file__), src) for src in ext_source]
    
    define_macros = []
    if torch.cuda.is_available():
        extension = torch.utils.cpp_extension.CUDAExtension(
            name="turboquant_cuda",
            sources=sources,
            include_dirs=[cuda_ext_include],
            define_macros=define_macros,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
        extensions.append(extension)
    else:
        extension = torch.utils.cpp_extension.CppExtension(
            name="turboquant_cuda",
            sources=sources,
            include_dirs=[cuda_ext_include],
            define_macros=define_macros,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        )
        extensions.append(extension)
    
    return extensions

setup(
    name="turboquant-cuda",
    version="0.1.0",
    description="CUDA kernels for TurboQuant-v3 INT4 quantization",
    author="Kubenew",
    author_email="kubenew@example.com",
    url="https://github.com/Kubenew/TurboQuant-v3",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    zip_safe=False,
)
