import os
from torch_ucc_test_setup import *

# create 2 UCC PGs
ucc_pg = init_process_groups("ucc", False)
dist.barrier()
ucc_pg.barrier()
