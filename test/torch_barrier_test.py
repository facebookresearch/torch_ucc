#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import time
import random
from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

for i in range(comm_size):
    rand_sleep = random.randint(1, 1000)
    time.sleep(rand_sleep/1000)
    if i == comm_rank:
        print("rank {} checks in".format(comm_rank))
    dist.barrier()
if comm_rank == 0:
    print("Test barrier: succeeded")
