/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <torch/extension.h>

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <pybind11/chrono.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <ucp/api/ucp.h>
#include <api/xccl.h>

#include "torch_ucc_sendrecv.hpp"
#include "torch_ucx_coll.hpp"
#include "torch_xccl.hpp"

namespace c10d {

class ProcessGroupUCC : public ProcessGroup {
 public:
    class WorkUCX: public ProcessGroup::Work {
    public:
        WorkUCX(torch_ucx_request_t *request, torch_ucx_comm_t *ucx_comm):
            req(request), comm(ucx_comm) {}
        virtual ~WorkUCX();
        bool isCompleted() override;
        bool isSuccess() const override;
        bool wait() override;
    protected:
        torch_ucx_request_t *req;
        torch_ucx_comm_t    *comm;
        friend class ProcessGroupUCC;
    };

    class WorkUCXColl: public ProcessGroup::Work {
    public:
        WorkUCXColl() {
            req = new torch_ucx_coll_request_t;
            no_progress = false;
        }
        virtual ~WorkUCXColl();
        bool isCompleted() override;
        bool isSuccess() const override;
        bool wait() override;
    protected:
        bool                     no_progress;
        torch_ucx_coll_request_t *req;
        friend class ProcessGroupUCC;
    };

  class WorkUCC : public ProcessGroup::Work {
   public:
    WorkUCC(xccl_coll_req_h request): req(request){}
    WorkUCC(){}

    virtual ~WorkUCC();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait() override;

   protected:
    xccl_coll_req_h         req;
    xccl_coll_op_args_t     args;
    std::vector<uint32_t>   scratch;
    std::vector<at::Tensor> output_data_vec;
    at::Tensor              flat_tensor;
    friend class ProcessGroupUCC;
  };


  explicit ProcessGroupUCC(const std::shared_ptr<Store>& store,
                           int rank = -1,
                           int size = -1);
  virtual ~ProcessGroupUCC();

  std::shared_ptr<ProcessGroup::Work> broadcast(std::vector<at::Tensor>& data,
                                                const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(std::vector<at::Tensor>& tensors,
                                                const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(std::vector<at::Tensor>& tensors,
                                                          const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(std::vector<at::Tensor>& tensors,
                                             const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                std::vector<at::Tensor>& inputTensors,
                                                const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(at::Tensor& outputBuffer,
                                                     at::Tensor& inputBuffer,
                                                     const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> barrier(const BarrierOptions& opts = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                             std::vector<at::Tensor>& inputTensors,
                                             const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(std::vector<at::Tensor>& outputTensors,
                                              std::vector<std::vector<at::Tensor>>& inputTensors,
                                              const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(std::vector<at::Tensor>& outputTensors,
                                                     std::vector<std::vector<at::Tensor>>& inputTensors,
                                                     const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall_base(at::Tensor& outputTensor,
                                                    at::Tensor& inputTensor,
                                                    std::vector<int64_t>& outputSplitSizes,
                                                    std::vector<int64_t>& inputSplitSizes,
                                                    const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall(std::vector<at::Tensor>& outputTensors,
                                               std::vector<at::Tensor>& inputTensors,
                                               const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(std::vector<at::Tensor>& tensors,
                                           int dstRank,
                                           int tag);

  std::shared_ptr<ProcessGroup::Work> recv(std::vector<at::Tensor>& tensors,
                                           int srcRank,
                                           int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                                    int tag);

  static std::shared_ptr<ProcessGroup> createProcessGroupUCC(
      const std::shared_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupUCCConstructor() __attribute__((constructor)) {
      py::object module = py::module::import("torch.distributed");
      py::object register_backend = module.attr("Backend").attr("register_backend");
      register_backend("ucc", py::cpp_function(createProcessGroupUCC));
  }

protected:
    std::shared_ptr<Store>                store_;
    torch_ucx_comm_t                      *ucx_comm;
    torch_ucx_coll_comm_t                 *ucx_coll_comm;
    torch_xccl_comm_t                     *xccl_comm;
    std::mutex                            pg_mutex;
    std::thread                           progress_thread;
    bool                                  stop_progress_loop;
    std::deque<torch_ucx_coll_request_t*> progress_queue;
    std::condition_variable               queue_produce_cv;
    std::condition_variable               queue_consume_cv;

    void progress_loop();
    void enqueue_request(torch_ucx_coll_request_t* req);
private:
    struct ucc_config {
        bool enable_progress_thread;
        bool enable_xccl;
        bool enable_ucx;
    } config;
  
    void                 read_config();
    void                 check_tensor(const std::vector<at::Tensor>& tensors);
    xccl_coll_req_h      launch_xccl_collective(xccl_collective_type_t coll,
                                           const std::vector<at::Tensor>& tensors,
                                           int root, xccl_op_t op);
};

} // namespace c10d
