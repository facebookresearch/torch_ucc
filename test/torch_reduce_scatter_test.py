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
print_test_head("Reduce_scatter", comm_rank)
for count in counts:
    tensors_input = []
    for p in range(comm_size):
        tensors_input.append(get_tensor(count, args.use_cuda))
    tensor_ucc = get_tensor(count, args.use_cuda)
    tensor_test = tensor_ucc.clone()
    tensors_input[0] = do_compute(tensors_input[0])
    dist.reduce_scatter(tensor_ucc, tensors_input)
    dist.reduce_scatter(tensor_test, tensors_input, group=pg)
    status = check_tensor_equal(tensor_ucc, tensor_test)
    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)

if comm_rank == 0:
    print("Test reduce_scatter: succeeded")
