from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
from setuptools import find_packages
import os
import os.path as osp

ROOT = os.path.dirname(os.path.abspath(__file__))

_ext_src_root = "_ops"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# Load the version from the _version.py file
exec(open(osp.join(ROOT, "_version.py")).read())

# Specify requirements
requirements = ["torch>=1.4"]

# Optional: Uncomment this if you know the CUDA architecture of your GPU
# os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.7;8.9;9.0"

setup(
    name='pointnet22',
    ext_modules=[
        CUDAExtension(
            name='pointnet22._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
            },
            include_dirs=[osp.join(ROOT, _ext_src_root, "include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=requirements,  # Include your requirements here
    packages=find_packages()  # Automatically find all packages
)
