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

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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

constexpr const char* UCC_BACKEND_NAME = "ucc";

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
    std::vector<at::Tensor> flat;
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

  class AllgathervWorkData : public WorkData {
   public:
    AllgathervWorkData(int size)
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
        ucc_coll_req_h request,
        uint64_t seq_num)
        : status_(UCC_INPROGRESS), comm_(comm), request_(request),
          seq_num_(seq_num) {}
    void finalize(std::exception_ptr eptr = nullptr);
    ucc_status_t status_;
    CommBase* comm_;
    ucc_coll_req_h request_;
    uint64_t seq_num_;
    std::unique_ptr<WorkData> data;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::exception_ptr eptr_;
    std::vector<ucp_ep_h> *eps;
    int rank;
    int comm_id;
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
    std::vector<at::Tensor> result() override;
#ifdef USE_CUDA
    std::unique_ptr<at::cuda::CUDAEvent> fence = nullptr;
    event_pool_t* ep = nullptr;
#endif
   protected:
    std::shared_ptr<ProgressEntry> entry_;
   private:
    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;
    // Store a reference to collective's outputs, used by result
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
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

#ifdef USE_CUDA
	std::unique_ptr<at::cuda::CUDAEvent> getPooledEvent();
#endif

template <typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<ProcessGroup::Work> collective_post(
      OpType opType,
      PreProcess preproc,
      PostProcess postproc,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::Device dev,
      std::vector<at::Tensor> &outputTensors,
      const char* prof_title
	);

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

 protected:
  const std::chrono::duration<float> timeout_;
  std::shared_ptr<torch_ucc_oob_coll_info_t> oob;
  std::shared_ptr<CommPG> comm = {nullptr};
  uint32_t comm_id;
  std::vector<ucp_ep_h> eps;
  ucc_team_h team {nullptr};
  ucc_ee_h cuda_ee {nullptr};
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream = nullptr;
  event_pool_t ep;
#endif
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};

class CommPG {
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
  std::vector<torch_ucc_rank_state_t> comm_state;
  std::shared_ptr<torch_ucc_oob_coll_info_t> oob;
  CommUCX ucx_comm;
  CommUCC ucc_comm;
  uint64_t seq_num;
  c10::DeviceIndex device_index;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<std::shared_ptr<ProcessGroupUCC::ProgressEntry>> progress_queue;
  bool stop_progress_loop;
  bool collective_inprogress;

  void check_communicator_status(int my_rank, int comm_id, uint64_t seq_num,
      std::vector<ucp_ep_h> *eps);
 public:
  c10::DeviceIndex cuda_device_index;
  CommPG(const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob, c10::Device dev);

  ~CommPG();

  void ucx_connect_eps(
      std::vector<ucp_ep_h>& eps,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob);

  void ucx_disconnect_eps(
      std::vector<ucp_ep_h>& eps,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob);

  void ucc_create_team(
      ucc_team_h& team,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob);

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
      ucc_ee_h ee,
      std::vector<ucp_ep_h> *eps,
      int rank,
      int comm_id);

#endif

  void enqueue_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team,
      std::vector<ucp_ep_h> *eps,
      int rank,
      int comm_id);

  static std::shared_ptr<CommPG> get_comm(
      uint32_t& id,
      c10::Device dev,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
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
  friend ucs_status_t torch_ucc_timeout_am_cb(
      void *arg,
      const void *header,
      size_t header_length,
      void *data,
      size_t length,
      const ucp_am_recv_param_t *param);
};

} // namespace c10d
