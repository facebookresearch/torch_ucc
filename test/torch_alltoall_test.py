import torch
import torch.distributed as dist
import torch_ucc
import sys
import os

def get_tensor(count):
    t = torch.randint(0, 100, (count,), dtype=torch.int)
    return t

try:
    comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    print('OMPI env variables are not found')
    sys.exit(1)

os.environ['MASTER_PORT'] = '32167'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK']        = str(comm_rank)
os.environ['WORLD_SIZE']  = str(comm_size)


dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
pg = dist.new_group(backend='mpi')

counts = [comm_size]
for i in range(20):
    counts.append(counts[-1] * 2)
for count in counts:
    send_tensor = get_tensor(count)
    recv_tensor_ucc = torch.zeros(count, dtype=torch.int)
    recv_tensor_mpi = torch.zeros(count, dtype=torch.int)
    dist.all_to_all_single(recv_tensor_ucc, send_tensor)
    dist.all_to_all_single(recv_tensor_mpi, send_tensor, group=pg)
    # if comm_rank == 0:
    #     print(recv_tensor_ucc)
    #     print(recv_tensor_mpi)
    # print(torch.all(torch.eq(recv_tensor_ucc, recv_tensor_mpi)))
    if not torch.all(torch.eq(recv_tensor_ucc, recv_tensor_mpi)):
        print("Test failed: ", count)
        sys.exit(1)

print("Test succeeded ", counts)
