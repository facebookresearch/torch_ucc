#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

# torch-ucc
pytorch ucc plugin
## Build
Required packages:
* PyTorch
* UCX
* XCCL

```shell
UCX_HOME=<PATH_TO_UCX> UCC_HOME=<PATH_TO_XCCL> python setup.py clean --all install
```
## Run
```shell
export LD_LIBRARY_PATH=<PATH_TO_UCX>/lib:<PATH_TO_XCCL>/lib:$LD_LIBRARY_PATH
python example.py
```

```python
import torch
import torch.distributed as dist
import torch_ucc

....
dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
....
dist.all_to_all_single(recv_tensor, send_tensor)

```
