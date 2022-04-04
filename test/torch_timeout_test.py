#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import time
from torch_ucc_test_setup import *
from datetime import timedelta

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda, timedelta(seconds=10))

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

dist.barrier()
if comm_rank == 0:
    time.sleep(20)

estr = ""
try:
    req = dist.barrier()
except Exception as e:
    estr = str(e)

if comm_rank != 0:
    if "Timeout expired" in estr:
        print("Test OK")
    else:
        print("Test Failed")
        sys.exit(1)
