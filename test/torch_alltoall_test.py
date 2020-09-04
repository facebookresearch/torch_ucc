#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = [comm_size]
for i in range(20):
    counts.append(counts[-1] * 2)

for count in counts:
    send_tensor = get_tensor(count)
    recv_tensor_ucc = torch.zeros(count, dtype=torch.int)
    recv_tensor_test = torch.zeros(count, dtype=torch.int)
    dist.all_to_all_single(recv_tensor_ucc, send_tensor)
    dist.all_to_all_single(recv_tensor_test, send_tensor, group=pg)
    check_tensor_equal("alltoall", recv_tensor_ucc, recv_tensor_test)

if comm_rank == 0:
    print("Test alltoall: succeeded")
