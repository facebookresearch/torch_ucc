import os

from torch.utils.hipify import hipify_python

CUDA_TO_HIP_MAPPINGS = [
    ("UCS_MEMORY_TYPE_CUDA", "UCS_MEMORY_TYPE_ROCM"),
    ("UCC_MEMORY_TYPE_CUDA", "UCC_MEMORY_TYPE_ROCM"),
    ("UCC_EE_CUDA_STREAM", "UCC_EE_ROCM_STREAM"),
    ("nccl", "rccl"),
]

# TorchUCC specific hipification
def torch_ucc_hipify_file(src_path, dst_path, verbose=True):
    if verbose:
        print("Torch-UCC hipification applied to {} -> {}".format(src_path, dst_path))
    with open(src_path, "rt", encoding="utf-8") as fin:
        fin.seek(0)
        source = fin.read()
        for k, v in CUDA_TO_HIP_MAPPINGS:
            source = source.replace(k, v)
        fin.close()

    with open(dst_path, "wt", encoding="utf-8") as fout:
        fout.write(source)
        fout.close()


# Overwrite each source file for hipification
def torch_ucc_hipify(src_path_list, verbose=True):
    for src_path in src_path_list:
        torch_ucc_hipify_file(src_path, src_path)
