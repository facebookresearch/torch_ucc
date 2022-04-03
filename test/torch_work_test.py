#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch_ucc_test_setup import *


def test_future(obj):
    print("Test WorkUCC: succeeded")


args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

print_test_head("WorkUCC", comm_rank)
count = 32
tensor_ucc = get_tensor(count, args.use_cuda)

work = dist.all_reduce(tensor_ucc, async_op=True)

# test future functionality
fut = work.get_future().then(test_future)

# test result functionality
work.wait()
opTensor = work.result()

# test future functionality
fut.wait()
