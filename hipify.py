import argparse
import os
import shutil
import sys

from torch.utils.hipify import hipify_python

CUDA_TO_HIP_MAPPINGS = [
    ("UCS_MEMORY_TYPE_CUDA", "UCS_MEMORY_TYPE_ROCM"),
    ("UCC_MEMORY_TYPE_CUDA", "UCC_MEMORY_TYPE_ROCM"),
    ("nccl", "rccl"),
]

# TorchUCC specific hipification
def torch_ucc_hipify_file(src_path, dst_path):
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
def torch_ucc_hipify(src_path_list):
    for src_path in src_path_list:
        torch_ucc_hipify_file(src_path, src_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir",
        type=str,
        default="",
        help="project source directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="output directory", required=True
    )
    parser.add_argument(
        "-s",
        "--src",
        type=str,
        default="",
        help="relative path to a source file",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dst",
        type=str,
        default="",
        help="relative path to the hipified output file",
        required=True,
    )

    args = parser.parse_args()

    # Hipify one file at a time so that we can move the final output to BUCK specified path (via -d)
    torch_output = hipify_python.hipify(
        project_directory=args.project_dir,
        output_directory=args.output_dir,
        includes=[args.src],
        is_pytorch_extension=True,
        show_detailed=True,
    )

    # Apply TorchUCC-specific hipification and generate output to BUCK specified path
    torch_hip_path = torch_output["{}/{}".format(args.output_dir, args.src)][
        "hipified_path"
    ]
    output_path = "{}/{}".format(args.output_dir, args.dst)
    torch_ucc_hipify_file(torch_hip_path, output_path)


if __name__ == "__main__":
    sys.exit(main())
