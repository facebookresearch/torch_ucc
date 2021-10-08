/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#include "torch_ucc_comm.hpp"
#include <torch/python.h>

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
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

#ifdef USE_CUDA
#define SAVE_TENSORS(_TENSORS, _DATA)                       \
  do {                                                      \
    if ((_TENSORS)[0].device().is_cuda()) {                 \
      for (const auto i : c10::irange((_TENSORS).size())) { \
        c10::cuda::CUDACachingAllocator::recordStream(      \
            (_TENSORS)[i].storage().data_ptr(), (*stream)); \
      }                                                     \
    } else {                                                \
      (_DATA) = (_TENSORS);                                 \
    }                                                       \
  } while (0)

#else
#define SAVE_TENSORS(_TENSORS, _DATA) (_DATA) = (_TENSORS);
#endif

constexpr const char* UCC_BACKEND_NAME = "UCC";

enum torch_ucx_tag_type_t { TORCH_UCX_P2P_TAG, TORCH_UCX_OOB_TAG };

struct event_pool_t {
#ifdef USE_CUDA
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool;
#endif
  std::mutex event_pool_mutex;
};

class CommPG;

class ProcessGroupUCC : public ProcessGroup {
 private:
  void set_timeout(ucc_coll_args_t &args);

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

  class ProgressEntry {
    friend class ProcessGroupUCC;
    friend class CommPG;

  public:
    ProgressEntry(
        CommBase* comm,
        ucc_coll_req_h request)
        : status_(UCC_INPROGRESS), comm_(comm), request_(request) {}
    void finalize(std::exception_ptr eptr = nullptr);
    ucc_status_t status_;
    CommBase* comm_;
    ucc_coll_req_h request_;
    std::unique_ptr<WorkData> data;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::exception_ptr eptr_;
  };

  class WorkUCCsingle : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
   public:
    WorkUCCsingle(
        OpType opType,
        const char* prof_title)
        : ProcessGroup::Work(-1, opType, prof_title) {}
    ~WorkUCCsingle() {}
    void setException() {}
    void setAndThrowException() {}
    bool isCompleted() override {return true;}
    bool isSuccess() const override {return true;}
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {return true;}
  };

  class WorkUCC : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
    friend class CommPG;

   public:
    WorkUCC(
        OpType opType,
        const char* prof_title)
        : ProcessGroup::Work(-1, opType, prof_title) {}
    ~WorkUCC();
    void setException();
    void setAndThrowException();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
#ifdef USE_CUDA
    std::unique_ptr<at::cuda::CUDAEvent> fence = nullptr;
    event_pool_t* ep = nullptr;
#endif
   protected:
    std::shared_ptr<ProgressEntry> entry_;
   private:
    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };


  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1,
      std::chrono::duration<float> timeout = kProcessGroupDefaultTimeout);

  void initComm(c10::Device dev);

  ~ProcessGroupUCC() override;

  const std::string getBackendName() const override {
      return std::string(UCC_BACKEND_NAME);
  }

  c10::intrusive_ptr<ProcessGroup::Work> collective_post(
      OpType opType,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::Device dev,
      std::vector<at::Tensor> &outputTensors,
      const char* prof_title);

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

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
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
  const std::chrono::duration<float> timeout_;
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
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};

class CommPG {
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
  CommUCX ucx_comm;
  CommUCC ucc_comm;
  c10::DeviceIndex device_index;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<std::shared_ptr<ProcessGroupUCC::ProgressEntry>> progress_queue;
  bool stop_progress_loop;
  bool collective_inprogress;

 public:
  c10::DeviceIndex cuda_device_index;
  CommPG(const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      torch_ucc_oob_coll_info_t* oob_info, c10::Device dev);

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
      ucc_coll_req_h request,
      const char* prof_title);

#ifdef USE_CUDA
  void enqueue_cuda_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team,
      ucc_ee_h ee);
#endif

  void enqueue_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team);

  static std::shared_ptr<CommPG> get_comm(
      uint32_t& id,
      c10::Device dev,
      torch_ucc_oob_coll_info_t *oob,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);

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
