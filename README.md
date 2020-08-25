# Torch-UCC
The Torch-UCC plugin is a research prototype that enables collective communication over XCCL for distributed PyTorch applications that load the plugin at application runtime. The XCCL interface is a non-standard API with a corresponding [reference implementation](https://github.com/openucx/xccl) used to guide design decisions for the [UCC project](https://www.ucfconsortium.org/projects/ucc/).  

## Licenses
The torch-ucc plugin is licensed as:
* [BSD3](LICENSE)

## Contributor Agreement and Guidelines
In order to contribute to torch-ucc, please sign up with an appropriate
[Contributor Agreement](http://www.openucx.org/license/).
Follow these
[instructions](https://github.com/openucx/ucx/wiki/Guidance-for-contributors)
when submitting contributions and changes.


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
