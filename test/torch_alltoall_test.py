#
# Copyright (C) Mellanox Technologies Ltd. 2001-2021.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch_ucc_test_setup import *
import numpy as np

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(4, 22)

print_test_head("Alltoall", comm_rank)
for count in counts:
    recv_tensor_test = get_tensor(count * comm_size, is_cuda=False)
    send_tensor_list = []
    recv_tensor_ucc = []
    for p in range(comm_size):
        recv_tensor_ucc.append(get_tensor(count, args.use_cuda))
        send_tensor_list.append(get_tensor(count, args.use_cuda))

    dist.all_to_all(
        recv_tensor_ucc,
        send_tensor_list,
    )
    # flatten the send_tensor_list and use all_to_all_single as not all PGs support all_to_all primitive
    dist.all_to_all_single(
        recv_tensor_test, torch.stack(send_tensor_list, dim=0).view(-1).cpu(), group=pg
    )

    status = check_tensor_equal(
        torch.stack(recv_tensor_ucc, dim=0).view(-1), recv_tensor_test
    )
    dist.all_reduce(status, group=pg)
    print_test_result(status, "{} x {}".format(count, comm_size), comm_rank, comm_size)

if comm_rank == 0:
    print("Test alltoall: succeeded")
