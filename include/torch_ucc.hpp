/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#pragma once

#include "torch_ucc_comm.hpp"
#include <torch/python.h>

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include <pybind11/chrono.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace c10d {

#define TORCH_UCC_DEVICE_NOT_SET -2

#define TORCH_UCX_MAKE_P2P_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_TAG_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_comm)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_OOB_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_OOB_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_rank)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_SEND_TAG(_ucp_tag, _tag, _rank, _comm)      \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_ANY_SOURCE (TORCH_UCX_MAX_RANK - 1)
#define TORCH_UCX_ANY_SOURCE_MASK (~TORCH_UCX_RANK_MASK)
#define TORCH_UCX_SPECIFIC_SOURCE_MASK ((uint64_t)-1)

#define TORCH_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank, _comm) \
  do {                                                                       \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm));           \
    if ((_rank) == TORCH_UCX_ANY_SOURCE) {                                   \
      (_ucp_tag_mask) = TORCH_UCX_ANY_SOURCE_MASK;                           \
    } else {                                                                 \
      (_ucp_tag_mask) = TORCH_UCX_SPECIFIC_SOURCE_MASK;                      \
    }                                                                        \
  } while (0)

#define TORCH_UCX_MAKE_OOB_SEND_TAG(_ucp_tag, _tag, _rank, _comm)  \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_MAKE_OOB_RECV_TAG(                               \
    _ucp_tag, _ucp_tag_mask, _tag, _rank, _comm)                   \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
    (_ucp_tag_mask) = (uint64_t)-1;                                \
  } while (0)

enum torch_ucx_tag_type_t { TORCH_UCX_P2P_TAG, TORCH_UCX_OOB_TAG };

struct event_pool_t {
#ifdef USE_CUDA
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool;
#endif
  std::mutex event_pool_mutex;
};

class CommPG;

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkData {
   public:
    std::vector<at::Tensor> src;
    std::vector<at::Tensor> dst;
    WorkData() {}
    virtual ~WorkData() = default;
  };
  class AlltoallWorkData : public WorkData {
   public:
    AlltoallWorkData(int size)
        : send_lengths(size),
          send_offsets(size),
          recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint32_t> send_lengths;
    std::vector<uint32_t> send_offsets;
    std::vector<uint32_t> recv_lengths;
    std::vector<uint32_t> recv_offsets;
  };

  class AllgatherWorkData : public WorkData {
   public:
    AllgatherWorkData(int size)
        : recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };

  class WorkUCC : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
    friend class CommPG;

   public:
    WorkUCC(
        OpType opType,
        ucc_status_t status,
        ucc_coll_req_h request,
        ucc_ee_h ee,
        CommBase* comm)
        : ProcessGroup::Work(-1, opType),
          status_(status),
          request_(request),
          comm_(comm) {}
    ~WorkUCC();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    void finalize();
    std::unique_ptr<WorkData> data;
#ifdef USE_CUDA
    std::unique_ptr<at::cuda::CUDAEvent> fence = nullptr;
    event_pool_t* ep = nullptr;
#endif
   protected:
    ucc_status_t status_;
    ucc_coll_req_h request_;
    CommBase* comm_;
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1);

  void initComm(c10::Device dev);

  ~ProcessGroupUCC() override;

  c10::intrusive_ptr<ProcessGroup::Work> collective_post(
      OpType opType,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::Device dev);

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  static c10::intrusive_ptr<ProcessGroup> createProcessGroupUCC(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupUCCConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("ucc", py::cpp_function(createProcessGroupUCC));
  }

 protected:
  torch_ucc_oob_coll_info_t oob;
  std::shared_ptr<CommPG> comm;
  uint32_t comm_id;
  std::vector<ucp_ep_h> eps;
  ucc_team_h team;
  ucc_ee_h cuda_ee;
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream = nullptr;
  event_pool_t ep;
#endif
};

class CommPG {
  CommUCX ucx_comm;
  CommUCC ucc_comm;
  c10::DeviceIndex device_index;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<c10::intrusive_ptr<ProcessGroupUCC::WorkUCC>> progress_queue;
  bool stop_progress_loop;

 public:
  c10::DeviceIndex cuda_device_index;
  CommPG(torch_ucc_oob_coll_info_t* oob_info,
      c10::Device dev);

  ~CommPG();

  void ucx_connect_eps(
      std::vector<ucp_ep_h>& eps,
      torch_ucc_oob_coll_info_t* oob);

  void ucx_disconnect_eps(
      std::vector<ucp_ep_h>& eps,
      torch_ucc_oob_coll_info_t* oob);

  void ucc_create_team(
      ucc_team_h& team,
      torch_ucc_oob_coll_info_t* oob_info);

  void ucc_destroy_team(ucc_team_h& team);

  c10::intrusive_ptr<ProcessGroup::Work> enqueue_p2p(
      OpType opType,
      ucc_coll_req_h request);

#ifdef USE_CUDA
  c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> enqueue_cuda_collective(
      OpType opType,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      ucc_team_h& team,
      ucc_ee_h ee,
      std::unique_ptr<at::cuda::CUDAEvent> cuda_ev,
      const at::cuda::CUDAStream& stream,
      event_pool_t* ep);
#endif

  c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> enqueue_collective(
      OpType opType,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      ucc_team_h& team);

  static std::shared_ptr<CommPG> get_comm(
      uint32_t& id,
      c10::Device dev,
      torch_ucc_oob_coll_info_t *oob);

  void progress_loop();

  ucc_coll_req_h send_nb(
      ucp_ep_h ep,
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag);

  ucc_coll_req_h recv_nb(
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag,
      ucp_tag_t ucp_tag_mask);
};

} // namespace c10d
