#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import os
import sys
from setuptools import setup
from torch.utils import cpp_extension
from torch import __version__ as torch_version

ver_major, ver_minor = torch_version.split(".")[:2]
ver_major = '-DTORCH_VER_MAJOR='+ver_major
ver_minor = '-DTORCH_VER_MINOR='+ver_minor

ucx_home = os.environ.get("UCX_HOME")
if ucx_home is None:
    ucx_home = os.environ.get("HPCX_UCX_DIR")
if ucx_home is None:
    print("Couldn't find UCX install dir, please set UCX_HOME env variable")
    sys.exit(1)

ucc_home = os.environ.get("UCC_HOME")
if ucc_home is None:
    print("Couldn't find UCC install dir, please set UCC_HOME env variable")
    sys.exit(1)

ucc_plugin_dir = os.path.dirname(os.path.abspath(__file__))

with_cuda = os.environ.get("WITH_CUDA")
if with_cuda is None or with_cuda == "no":
    print("CUDA support is disabled")
    module = cpp_extension.CppExtension(
        name = "torch_ucc",
        sources = ["src/torch_ucc.cpp",
                   "src/torch_ucc_sendrecv.cpp",
                   "src/torch_ucx_alltoall.cpp",
                   "src/torch_ucx_coll.cpp",
                   "src/torch_xccl.cpp"],
        include_dirs = ["{}/include/".format(ucc_plugin_dir),
                        "{}/include/".format(ucx_home),
                        "{}/include/".format(ucc_home)],
        library_dirs = ["{}/lib/".format(ucx_home),
                        "{}/lib/".format(ucc_home)],
        libraries = ["ucp", "uct", "ucm", "ucs", "xccl"],
        extra_compile_args=['-g', '-O0', ver_major, ver_minor]
    )
else:
    print("CUDA support is enabled")
    module = cpp_extension.CUDAExtension(
        name = "torch_ucc",
        sources = ["src/torch_ucc.cpp",
                   "src/torch_ucc_sendrecv.cpp",
                   "src/torch_ucx_alltoall.cpp",
                   "src/torch_ucx_coll.cpp",
                   "src/torch_xccl.cpp"],
        include_dirs = ["{}/include/".format(ucc_plugin_dir),
                        "{}/include/".format(ucx_home),
                        "{}/include/".format(ucc_home)],
        library_dirs = ["{}/lib/".format(ucx_home),
                        "{}/lib/".format(ucc_home)],
        libraries = ["ucp", "uct", "ucm", "ucs", "xccl"],
        extra_compile_args=['-g', '-O0', '-DUSE_CUDA', ver_major, ver_minor]
    )

setup(
    name = "torch-ucc",
    version = "0.1.0",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

