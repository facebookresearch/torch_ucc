# PyTorch plugin for UCC

This repo implements PyTorch Process Group API for [UCC](https://www.ucfconsortium.org/projects/ucc/) as a third-party plugin.

## Requirements
* PyTorch
* [UCX](https://github.com/openucx/ucx)
* [UCC](https://github.com/openucx/ucc)

## License

This repo is released under the MIT license. Please see the [`LICENSE`](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for more info.
```shell
# Build
UCX_HOME=<PATH_TO_UCX> UCC_HOME=<PATH_TO_UCC> WITH_CUDA=<PATH_TO_CUDA> python setup.py install
```
UCX_HOME required, specifies path to UCX installation directory

UCC_HOME required, specifies path to UCC installation directory

WITH_CUDA optional, if WITH_CUDA=no is set then only CPU tensors are supported

## Run
Configuration variables
| Name                               | Values                    | Description                                                                                                   |
|------------------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------|
| TORCH_UCC_ALLGATHER_BLOCKING_WAIT | 0 or 1                     | Sets behavior of wait function for CUDA Allgather. [Async collective in PyTorch](https://pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations)|
| TORCH_UCC_ALLREDUCE_BLOCKING_WAIT | 0 or 1                     | Sets behavior of wait function for CUDA Allreduce.                                                            |
| TORCH_UCC_ALLTOALL_BLOCKING_WAIT  | 0 or 1                     | Sets behavior of wait function for CUDA Alltoall.                                                             |
| TORCH_UCC_BCAST_BLOCKING_WAIT     | 0 or 1                     | Sets behavior of wait function for CUDA Bcast.                                                                |

```shell
export LD_LIBRARY_PATH=<PATH_TO_UCX>/lib:<PATH_TO_UCC>/lib:$LD_LIBRARY_PATH
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
