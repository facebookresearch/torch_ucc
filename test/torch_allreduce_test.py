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
print_test_head("Allreduce", comm_rank)
for count in counts:
    tensor_ucc = get_tensor(count, args.use_cuda)
    tensor_test = tensor_ucc.cpu()
    dist.all_reduce(tensor_ucc)
    dist.all_reduce(tensor_test, group=pg)
    status = check_tensor_equal(tensor_ucc, tensor_test)
    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)

if comm_rank == 0:
    print("Test allreduce: succeeded")
