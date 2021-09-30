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
print_test_head("Point-to-point", comm_rank)

for count in counts:
    tensor_ucc = get_tensor(count, args.use_cuda)
    tensor_test = tensor_ucc.clone()
    tensor_ucc = do_compute(tensor_ucc)
    tensor_test = do_compute(tensor_test)
    # pt2pt-based bcast if more than 2 processes
    if comm_rank == 0:
        for dst in range(comm_size-1):
            dist.send(tensor_ucc, dst=dst+1, tag=0)
            dist.send(tensor_test, dst=dst+1, tag=0, group=pg)
        status = torch.tensor(1, device=tensor_ucc.device)
    else:
        dist.recv(tensor_ucc, src=0, tag=0)
        dist.recv(tensor_test, src=0, tag=0, group=pg)
        status = check_tensor_equal(tensor_ucc, tensor_test)
    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)

if comm_rank == 0:
    print("Test Point-to-point: succeeded")
