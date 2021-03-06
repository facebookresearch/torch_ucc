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
print_test_head("Gather", comm_rank)
for count in counts:
    tensor_input = get_tensor(count, args.use_cuda)
    tensors_out_ucc = None
    tensors_out_test = None
    if comm_rank == 0:
        tensors_out_ucc = []
        tensors_out_test = []
        for p in range(comm_size):
            tensors_out_ucc.append(get_tensor(count, args.use_cuda))
            tensors_out_test.append(get_tensor(count, is_cuda=False))

    dist.gather(gather_list=tensors_out_ucc, tensor=tensor_input, dst=0)
    dist.gather(gather_list=tensors_out_test, tensor=tensor_input.cpu(), dst=0, group=pg)

    if comm_rank == 0:
        status = check_tensor_list_equal(tensors_out_ucc, tensors_out_test)
    else:
        status = torch.tensor(1, device=tensor_ucc.device)

    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)
if comm_rank == 0:
    print("Test gather: succeeded")
