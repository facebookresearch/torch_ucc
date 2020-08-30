#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import os
import sys
from setuptools import setup
from torch.utils import cpp_extension

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

cuda_home = os.environ.get("CUDA_HOME")
if cuda_home is None:
    print("Couldn't find CUDA, please set CUDA_HOME env variable")
    sys.exit(1)

ucc_plugin_dir = os.path.dirname(os.path.abspath(__file__))

module = cpp_extension.CppExtension(
    name = "torch_ucc",
    sources = ["src/torch_ucc.cpp",
               "src/torch_ucc_sendrecv.cpp",
               "src/torch_ucx_alltoall.cpp",
               "src/torch_ucx_coll.cpp",
               "src/torch_xccl.cpp"],
    include_dirs = ["{}/include/".format(ucc_plugin_dir),
                    "{}/include/".format(ucx_home),
                    "{}/include/".format(ucc_home),
                    "{}/include/".format(cuda_home)],
    library_dirs = ["{}/lib/".format(ucx_home),
                    "{}/lib/".format(ucc_home),
                    "{}/lib64/".format(cuda_home)],
    libraries = ["ucp", "uct", "ucm", "ucs", "xccl", "cudart"],
    extra_compile_args=['-g', '-O0']
)

setup(
    name = "torch-ucc",
    version = "0.1.0",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

