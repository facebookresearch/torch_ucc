#include "torch_ucc.hpp"
#include "torch_ucc_sendrecv.hpp"
#include "torch_ucx_coll.hpp"
#include "torch_xccl.hpp"
#include <map>
#include <iostream>
#include <stdio.h>

namespace c10d {

std::map<ReduceOp, xccl_op_t> xccl_op_map = {
    {ReduceOp::MIN,     XCCL_OP_MIN},
    {ReduceOp::MAX,     XCCL_OP_MAX},
    {ReduceOp::SUM,     XCCL_OP_SUM},
    {ReduceOp::PRODUCT, XCCL_OP_PROD},
};

std::map<at::ScalarType, xccl_dt_t> xccl_type_map = {
    {at::kByte,   XCCL_DT_UINT8},
    {at::kChar,   XCCL_DT_INT8},
    {at::kHalf,   XCCL_DT_FLOAT16},
    {at::kDouble, XCCL_DT_FLOAT64},
    {at::kFloat,  XCCL_DT_FLOAT32},
    {at::kInt,    XCCL_DT_INT32},
    {at::kLong,   XCCL_DT_INT64},
};

void ProcessGroupUCC::check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupUCC takes 1 tensoe");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be contiguous");
  }
  if (tensors[0].is_sparse()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be dense");
  }
  //TODO: check cuda case
}

xccl_coll_req_h ProcessGroupUCC::launch_xccl_collective(xccl_collective_type_t coll,
                                                       const std::vector<at::Tensor>& tensors,
                                                       int root, xccl_op_t op)
{
  xccl_coll_op_args_t coll_args;
  xccl_coll_req_h     request;

  auto &tensor = tensors[0];
  coll_args.coll_type              = coll;
  coll_args.buffer_info.src_buffer = tensor.data_ptr();
  coll_args.buffer_info.dst_buffer = tensor.data_ptr();
  coll_args.buffer_info.len        = tensor.numel() * tensor.element_size();

  if ((coll == XCCL_BCAST) || (coll == XCCL_REDUCE)) {
      coll_args.root               = root;
  }

  if ((coll == XCCL_REDUCE) || (coll == XCCL_ALLREDUCE)) {
    coll_args.reduce_info.dt       = xccl_type_map.at(tensor.scalar_type());
    coll_args.reduce_info.op       = op;
    coll_args.reduce_info.count    = tensor.numel();
  }

  coll_args.alg.set_by_user        = 0;
  coll_args.tag                    = 123;

  xccl_collective_init(&coll_args, &request, xccl_comm->xccl_team);
  xccl_collective_post(request);
   return request;
}

ProcessGroupUCC::WorkUCX::~WorkUCX()
{
    if (req != NULL) {
        torch_ucx_request_free(req);
    }
}

bool ProcessGroupUCC::WorkUCX::isCompleted()
{
    torch_ucx_status_t st;

    st = torch_ucx_req_test(comm, &req, 1, NULL, 1, 1);
    return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCX::isSuccess() const
{
  //TODO
  return true;
}

bool ProcessGroupUCC::WorkUCX::wait()
{
    torch_ucx_req_test(comm, &req, 1, NULL, -1, 1);
    return true;
}

ProcessGroupUCC::WorkUCXColl::~WorkUCXColl()
{
    if (req != NULL) {
        delete req;
    }
}

bool ProcessGroupUCC::WorkUCXColl::isCompleted()
{
    torch_ucx_status_t st;

    if (no_progress) {
        st = req->status;
    } else {
        st = torch_ucx_coll_test(req);
    }

    return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCXColl::isSuccess() const
{
  //TODO
  return true;
}

bool ProcessGroupUCC::WorkUCXColl::wait()
{
    torch_ucx_status_t st;

    do {
        if (no_progress) {
            st = req->status;
        } else {
            st = torch_ucx_coll_test(req);
        }
    } while(st == TORCH_UCX_INPROGRESS);

    return true;
}

ProcessGroupUCC::WorkUCC::~WorkUCC()
{
  xccl_collective_finalize(req);
}

bool ProcessGroupUCC::WorkUCC::isCompleted()
{
  xccl_status_t st;

  st = xccl_collective_test(req);
  
  return st != XCCL_INPROGRESS;
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const
{
  return true;
}

bool ProcessGroupUCC::WorkUCC::wait()
{
  xccl_status_t st;

  st = xccl_collective_wait(req);

  if (args.coll_type == XCCL_ALLGATHER) {
    for (size_t i = 0; i < output_data_vec.size(); ++i) {
      (output_data_vec)[i].copy_(flat_tensor[i]);
    }

  }

  return st == XCCL_OK;
}

void ProcessGroupUCC::read_config()
{
    char *env;

    config.enable_xccl            = true;
    config.enable_ucx             = true;
    config.enable_progress_thread = true;
 
    env = std::getenv("TORCH_UCC_UCX_ENABLE");
    if (env) {
        config.enable_xccl = std::atoi(env);
    }
    env = std::getenv("TORCH_UCC_XCCL_ENABLE");
    if (env) {
        config.enable_ucx = std::atoi(env);
    }
    env = std::getenv("TORCH_UCC_THREAD_ENABLE");
    if (env) {
        config.enable_progress_thread = std::atoi(env);
    }

}

ProcessGroupUCC::ProcessGroupUCC(const std::shared_ptr<Store>& store,
                                 int rank,
                                 int size)
    : ProcessGroup(rank, size),
      store_(store), stop_progress_loop(false) {
    torch_ucx_status_t st;

    read_config();
    st = torch_ucx_comm_init(&ucx_comm, size, rank, store_);
    if (st != TORCH_UCX_OK) {
        throw std::runtime_error("ProcessGroupUCC init failed");
    }
    st = torch_ucx_coll_comm_init(ucx_comm, &ucx_coll_comm);
    if (st != TORCH_UCX_OK) {
        throw std::runtime_error("ProcessGroupUCC init failed");
    }
    st = torch_xccl_comm_init(ucx_comm, &xccl_comm);
    if (st != TORCH_UCX_OK) {
        throw std::runtime_error("ProcessGroupUCC init failed");
    }

    if (config.enable_progress_thread) {
        progress_thread = std::thread(&ProcessGroupUCC::progress_loop, this);
    }
}

void ProcessGroupUCC::progress_loop()
{
    std::unique_lock<std::mutex> lock(pg_mutex);
    torch_ucx_coll_request_t     *req;
    torch_ucx_status_t           st;
 
    while(!stop_progress_loop) {
        if (progress_queue.empty()) {
            queue_produce_cv.wait(lock);
            continue;
        }
        req = progress_queue.front();
        progress_queue.pop_front();
        lock.unlock();
        queue_consume_cv.notify_one();
        do {
            st = torch_ucx_coll_test(req);
        } while(st == TORCH_UCX_INPROGRESS);
        lock.lock();
    }
}

void ProcessGroupUCC::enqueue_request(torch_ucx_coll_request_t* req)
{
    std::unique_lock<std::mutex> lock(pg_mutex);
    progress_queue.push_back(req);
    lock.unlock();
    queue_produce_cv.notify_one();
}

ProcessGroupUCC::~ProcessGroupUCC()
{
    if (config.enable_progress_thread) {
        std::unique_lock<std::mutex> lock(pg_mutex);
        queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
        stop_progress_loop = true;
        lock.unlock();
        queue_produce_cv.notify_all();
        progress_thread.join();
    }

    torch_xccl_comm_close(xccl_comm);
    torch_ucx_coll_comm_close(ucx_coll_comm);
    torch_ucx_comm_close(ucx_comm, store_);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(std::vector<at::Tensor>& tensors,
                                                               const BroadcastOptions& opts)
{
   xccl_coll_req_h request;
  
//   request = launch_xccl_collective(XCCL_BCAST, tensors, opts.rootRank,
//                                    XCCL_OP_LAST_PREDEFINED);
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(std::vector<at::Tensor>& tensors,
                                                               const AllreduceOptions& opts)
{
   xccl_coll_req_h request;
  
  request = launch_xccl_collective(XCCL_ALLREDUCE, tensors, -1,
                                   xccl_op_map.at(opts.reduceOp));
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(std::vector<at::Tensor>& tensors,
                                                                         const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(std::vector<at::Tensor>& tensors,
                                                            const ReduceOptions& opts)
{
   xccl_coll_req_h request;
  
  request = launch_xccl_collective(XCCL_REDUCE, tensors, opts.rootRank,
                                   xccl_op_map.at(opts.reduceOp));
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                              std::vector<at::Tensor>& inputTensors,
                                                              const AllgatherOptions& opts)
{
  auto req     = std::make_shared<ProcessGroupUCC::WorkUCC>();
//   auto &tensor = inputTensors[0];
//   xccl_coll_op_args_t coll_args;
//   xccl_coll_req_h     request;

//   req->flat_tensor = newLikeFlat(outputTensors[0]);
//   req->output_data_vec = (outputTensors[0]);
//   coll_args.coll_type              = XCCL_ALLGATHER;
//   coll_args.buffer_info.src_buffer = tensor.data_ptr();
//   coll_args.buffer_info.dst_buffer = req->flat_tensor.data_ptr();
//   coll_args.buffer_info.len        = tensor.numel() * tensor.element_size() * size_;
//   coll_args.alg.set_by_user        = 0;
//   coll_args.tag                    = 123;

//   xccl_collective_init(&coll_args, &request, xccl_team);
//   xccl_collective_post(request);
//   req->args = coll_args;
//   req->req = request;

  return req;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(at::Tensor& outputBuffer,
                                                                    at::Tensor& inputBuffer,
                                                                    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& opts) {
  xccl_coll_req_h request;
  xccl_coll_op_args_t coll_args;

  coll_args.coll_type = XCCL_BARRIER;

  xccl_collective_init(&coll_args, &request, xccl_comm->xccl_team);
  xccl_collective_post(request);

  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<std::vector<at::Tensor>>& inputTensors,
                                                             const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(std::vector<at::Tensor>& outputTensors,
                                                                    std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}


int64_t computeLengthsAndOffsets(int group_size,
                                 const std::vector<int64_t>& split_sizes,
                                 const at::Tensor& tensor,
                                 uint32_t* lengths,
                                 uint32_t* offsets)
{
  bool equal_splits = false;
  int64_t dim0_size = tensor.size(0);
  int64_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  int64_t split_size = 0;
  int64_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }

  for (int i = 0; i < group_size; i++) {
    int64_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    lengths[i] = length;
    offsets[i] = offset;
    offset += length;
  }
  return offset;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(at::Tensor& outputTensor,
                                                                   at::Tensor& inputTensor,
                                                                   std::vector<int64_t>& outputSplitSizes,
                                                                   std::vector<int64_t>& inputSplitSizes,
                                                                   const AllToAllOptions& opts)
{
    if (config.enable_ucx) {
        auto request = std::make_shared<ProcessGroupUCC::WorkUCXColl>();

        if ((outputSplitSizes.size() == 0) || (inputSplitSizes.size() == 0)) {
            request->req->src_buffer = inputTensor.data_ptr();
            request->req->dst_buffer = outputTensor.data_ptr();
            request->req->len = inputTensor.element_size() * inputTensor.numel() / size_;
        } else {
            throw std::runtime_error("ProcessGroupUCC: ucx backend doesn't support alltoallv");
        }

        torch_ucx_alltoall_start(ucx_coll_comm, request->req);
        if (config.enable_progress_thread) {
            enqueue_request(request->req);
            request->no_progress = true;
        }
        return request;
    }
    if (config.enable_xccl) {
        auto req = std::make_shared<ProcessGroupUCC::WorkUCC>();
        xccl_coll_req_h     request;
        xccl_coll_op_args_t coll_args;

        if ((outputSplitSizes.size() == 0) || (inputSplitSizes.size() == 0)) {
            coll_args.coll_type              = XCCL_ALLTOALL;
            coll_args.buffer_info.src_buffer = inputTensor.data_ptr();
            coll_args.buffer_info.dst_buffer = outputTensor.data_ptr();
            coll_args.buffer_info.len        = inputTensor.element_size() * inputTensor.numel() / size_;
            coll_args.alg.set_by_user        = 0;
            coll_args.tag                    = 123;
        } else {
            req->scratch.resize(4 * size_);
            uint32_t *send_lengths = req->scratch.data();
            uint32_t *recv_lengths = (uint32_t*)((ptrdiff_t)send_lengths + 1*size_*sizeof(uint32_t));
            uint32_t *send_offsets = (uint32_t*)((ptrdiff_t)send_lengths + 2*size_*sizeof(uint32_t));
            uint32_t *recv_offsets = (uint32_t*)((ptrdiff_t)send_lengths + 3*size_*sizeof(uint32_t));

            computeLengthsAndOffsets(size_, inputSplitSizes, inputTensor, send_lengths, send_offsets);
            computeLengthsAndOffsets(size_, outputSplitSizes, outputTensor, recv_lengths, recv_offsets);

            coll_args.coll_type                     = XCCL_ALLTOALLV;
            coll_args.buffer_info.src_buffer        = inputTensor.data_ptr();
            coll_args.buffer_info.src_displacements = send_offsets;
            coll_args.buffer_info.src_counts        = send_lengths;
            coll_args.buffer_info.src_datatype      = xccl_type_map.at(inputTensor.scalar_type());
            coll_args.buffer_info.dst_buffer        = outputTensor.data_ptr();
            coll_args.buffer_info.dst_displacements = recv_offsets;
            coll_args.buffer_info.dst_counts        = recv_lengths;
            coll_args.buffer_info.dst_datatype      = xccl_type_map.at(outputTensor.scalar_type());
            coll_args.alg.set_by_user               = 0;
            coll_args.tag                           = 123;
        }

        xccl_collective_init(&coll_args, &request, xccl_comm->xccl_team);
        xccl_collective_post(request);

        req->args = coll_args;
        req->req  = request;
        return req;
    }

    throw std::runtime_error("ProcessGroupUCC: no collective backends");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(std::vector<at::Tensor>& outputTensors,
                                                              std::vector<at::Tensor>& inputTensors,
                                                              const AllToAllOptions& opts)
{
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::send(std::vector<at::Tensor>& tensors,
                                                          int dstRank,
                                                          int tag)
{
    //TODO: check tensor count and type, assume single dense tensor
    auto   &tensor = tensors[0];
    size_t size    = tensor.numel() * tensor.element_size();
    torch_ucx_request_t *req;
    torch_ucx_status_t  st;

    st = torch_ucx_send_nb(ucx_comm, tensor.data_ptr(), size, dstRank,
                           tag, &req, TORCH_UCX_P2P_TAG);
    if (st < 0) {
       throw std::runtime_error("TorchUCC: failed to send msg");
    }
  
    return std::make_shared<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(std::vector<at::Tensor>& tensors,
                                                          int srcRank,
                                                          int tag)
{
    auto   &tensor = tensors[0];
    size_t size    = tensor.numel() * tensor.element_size();
    torch_ucx_request_t *req;
    torch_ucx_status_t  st;

    st = torch_ucx_recv_nb(ucx_comm, tensor.data_ptr(), size, srcRank,
                           tag, &req, TORCH_UCX_P2P_TAG);
    if (st < 0) {
       throw std::runtime_error("TorchUCC: failed to recv msg");
    }

    return std::make_shared<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(std::vector<at::Tensor>& tensors,
                                                                   int tag)
{
    throw std::runtime_error("TorchUCC: recvAnysource is not supported");
}

std::shared_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const std::shared_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return std::make_shared<ProcessGroupUCC>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
