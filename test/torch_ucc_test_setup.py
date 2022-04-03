#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

# torch_ucc is required to enable ucc PG
import torch_ucc  # noqa: F401


def parse_test_args():
    parser = argparse.ArgumentParser(description="PG UCC Test")
    parser.add_argument("--backend", type=str, default="mpi")
    parser.add_argument("--use-cuda", default=False, action="store_true")
    parser.add_argument("--enable-prof", default=False, action="store_true")
    args = parser.parse_args()

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(0)

    # Tensor mem type support seems to rely on static definition at https://pytorch.org/docs/stable/distributed.html
    valid_bends = ["mpi", "ucc", "gloo"]
    if args.backend not in valid_bends:
        print(
            "The specified backend {} does not support CPU tensors for result validation. Please choose from {}".format(
                args.backend, ", ".join(valid_bends)
            )
        )
        sys.exit(0)

    return args


def get_tensor(count, is_cuda):
    dev = torch.device("cuda") if is_cuda else torch.device("cpu")
    t = torch.randint(0, 100, (count,), dtype=torch.int, device=dev)
    return t


def init_process_groups(bend, use_cuda, to=timedelta(seconds=60)):
    try:
        comm_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        comm_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    except:
        print("OMPI env variables are not found")
        sys.exit(1)

    if use_cuda:
        torch.cuda.set_device(local_rank)

    os.environ["MASTER_PORT"] = "32167"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["RANK"] = str(comm_rank)
    os.environ["WORLD_SIZE"] = str(comm_size)
    dist.init_process_group("ucc", rank=comm_rank, world_size=comm_size, timeout=to)
    pg = dist.new_group(backend=bend)

    return pg


# Compare UCC result tensor with the checking PG's result tensor.
# Return check status allocated on PG's device because the result is exchanged by PG
def check_tensor_equal(t_ucc, t_pg):
    # Copy to CPU before comparing with PG's resut which is always on CPU
    if t_ucc.is_cuda:
        t_ucc = t_ucc.cpu()
    if torch.all(torch.eq(t_ucc, t_pg)):
        return torch.tensor(1, device=t_pg.device)
    else:
        print("failed on rank {}".format(os.environ["RANK"]))
        return torch.tensor(0, device=t_pg.device)


# Compare UCC result tensor list with the checking PG's result tensor list.
# Return check status allocated on PG's device because the result is exchanged by PG
def check_tensor_list_equal(t_ucc, t_pg):
    num_tensors = len(t_ucc)
    for i in range(num_tensors):
        # Copy to CPU before comparing with PG's resut which is always on CPU
        if t_ucc[i].is_cuda:
            t_ucc[i] = t_ucc[i].cpu()
        if not torch.all(torch.eq(t_ucc[i], t_pg[i])):
            return torch.tensor(0, device=t_pg[i].device)
    return torch.tensor(1, device=t_pg[i].device)


def print_test_head(test_name, comm_rank):
    if comm_rank == 0:
        print("{} test".format(test_name))
        print("{0:20} {1}".format("count", "result"))


def print_test_result(status, count, comm_rank, comm_size):
    if comm_rank == 0:
        result = "OK" if status == comm_size else "Failed"
        print("{0:20} {1}".format(str(count), result))
    if status != comm_size:
        sys.exit(1)


def do_compute(t):
    return torch.topk(t, t.size()[0])[0]
