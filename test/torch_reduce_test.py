#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(24)
print_test_head("Reduce", comm_rank)
for count in counts:
    tensor_ucc = get_tensor(count, args.use_cuda)
    tensor_ucc = do_compute(tensor_ucc)
    tensor_test = tensor_ucc.cpu()
    dist.reduce(tensor_ucc, dst=0)
    dist.reduce(tensor_test, dst=0, group=pg)
    # only root (i.e., rank 0 here) need to check results
    if comm_rank == 0:
        status = check_tensor_equal(tensor_ucc, tensor_test)
    else:
        status = torch.tensor(1, device=tensor_ucc.device)
    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)

if comm_rank == 0:
    print("Test Reduce: succeeded")
