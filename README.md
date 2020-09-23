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
# Build
UCX_HOME=<PATH_TO_UCX> WITH_XCCL=<PATH_TO_XCCL> WITH_CUDA=<PATH_TO_CUDA> python setup.py clean --all install
```
UCX_HOME required, specifies path to UCX installation directory

WITH_XCCL optional, if WITH_XCCL=no is specified then only point to point, alltoall and alltoallv operations are available

WITH_CUDA optional, if WITH_CUDA=no is specified then only CPU tensors are supported

## Run
Configuration variables
| Name                    | Values      | Description                                                                                                                                                                |
|-------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TORCH_UCC_COLL_BACKEND  | ucx or xccl | Set the backend that will be used for collective operations. UCX backend supports only alltoall and alltoallv collectives. |
| TORCH_UCC_THREAD_ENABLE | 0 or 1      | If not equal to zero then dedicated thread will be used to progress point to point and collective operations. |
| TORCH_UCC_UCX_REVERSE   | 0 or 1      | Determines order in which ranks traversed in alltoall pairwise exchange algorithm for UCX backend  |
| TORCH_UCC_UCX_CHUNK     | integer     | Maximum number of outstanding send/recv in alltoall pairwise exchange algorithm for UCX backend. If -1 is set then number of outstanding send/recv is equal to group size. |
 
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
