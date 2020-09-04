#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import argparse
import torch
import torch.distributed as dist
import torch_ucc
import sys
import os


def parse_test_args():
    parser = argparse.ArgumentParser(description="PG UCC Test")
    parser.add_argument("--backend", type=str, default='mpi')
    return parser.parse_args()


def get_tensor(count):
    t = torch.randint(0, 100, (count,), dtype=torch.int)
    return t

def init_process_groups(bend):
    try:
        comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    except:
        print('OMPI env variables are not found')
        sys.exit(1)

    os.environ['MASTER_PORT'] = '32167'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['RANK']        = str(comm_rank)
    os.environ['WORLD_SIZE']  = str(comm_size)
    dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
    pg = dist.new_group(backend=bend)

    return pg

def check_tensor_equal(test_name, t1, t2):
    if not torch.all(torch.eq(t1, t2)):
        print("Test {}: failed".format(test_name))
        sys.exit(1)
