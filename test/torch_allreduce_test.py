#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import numpy as np
from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(24)
for count in counts:
    tensor_ucc = get_tensor(count)
    tensor_test = tensor_ucc.clone()
    dist.all_reduce(tensor_ucc)
    dist.all_reduce(tensor_test, group=pg)
    check_tensor_equal("allreduce", tensor_ucc, tensor_test)

if comm_rank == 0:
    print("Test allreduce: succeeded")
