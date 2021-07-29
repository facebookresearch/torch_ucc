/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#include "torch_ucc.hpp"
#include <memory>

namespace c10d {

const std::map<c10::DeviceType, ucs_memory_type_t> ucs_mtype_map = {
    {c10::kCPU, UCS_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCS_MEMORY_TYPE_CUDA},
    {c10::kHIP, UCS_MEMORY_TYPE_ROCM},
    {c10::kFPGA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMSNPU, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kXLA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kVulkan, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMetal, UCS_MEMORY_TYPE_UNKNOWN},
};

const std::map<c10::DeviceType, ucc_memory_type_t> ucc_mtype_map = {
    {c10::kCPU, UCC_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCC_MEMORY_TYPE_CUDA},
    {c10::kHIP, UCC_MEMORY_TYPE_ROCM},
    {c10::kFPGA, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kMSNPU, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kXLA, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kVulkan, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kMetal, UCC_MEMORY_TYPE_UNKNOWN},
};

const std::map<at::ScalarType, ucc_datatype_t> ucc_dtype_map = {
    {at::kByte, UCC_DT_UINT8},
    {at::kChar, UCC_DT_INT8},
    {at::kHalf, UCC_DT_FLOAT16},
    {at::kDouble, UCC_DT_FLOAT64},
    {at::kFloat, UCC_DT_FLOAT32},
    {at::kInt, UCC_DT_INT32},
    {at::kLong, UCC_DT_INT64},
};

const std::map<ReduceOp, ucc_reduction_op_t> ucc_op_map = {
    {ReduceOp::SUM, UCC_OP_SUM},
    {ReduceOp::PRODUCT, UCC_OP_PROD},
    {ReduceOp::MIN, UCC_OP_MIN},
    {ReduceOp::MAX, UCC_OP_MAX},
    {ReduceOp::BAND, UCC_OP_BAND},
    {ReduceOp::BOR, UCC_OP_BOR},
    {ReduceOp::BXOR, UCC_OP_BXOR},
};

struct torch_ucc_config_t {
  std::once_flag flag;
  std::array<bool, 32> blocking_wait;
  bool enable_profiling;
  bool use_future;
} torch_ucc_config;

void read_confg() {
  char* env;

  torch_ucc_config.blocking_wait.fill(true);
  env = std::getenv("TORCH_UCC_ALLGATHER_BLOCKING_WAIT");
  if (env) {
    torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLGATHER] =
        std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_ALLREDUCE_BLOCKING_WAIT");
  if (env) {
    torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLREDUCE] =
        std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_ALLTOALL_BLOCKING_WAIT");
  if (env) {
    torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLTOALL_BASE] =
        std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_BCAST_BLOCKING_WAIT");
  if (env) {
    torch_ucc_config.blocking_wait[(std::uint8_t)OpType::BROADCAST] =
        std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_USE_FUTURE");
  if (env) {
    torch_ucc_config.use_future = !!std::atoi(env);
  } else {
    torch_ucc_config.use_future = true;
  }
  torch_ucc_config.enable_profiling = false;
  env = std::getenv("TORCH_UCC_PROFILING_ENABLE");
  if (env) {
    torch_ucc_config.enable_profiling = std::atoi(env);
  }
}

void check_device(c10::Device dev1, c10::Device dev2) {
  if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2) {
    throw std::runtime_error("ProcessGroupUCC multidevice is not supported");
  }
}

void check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupUCC takes 1 tensor");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
  if (tensors[0].is_sparse()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be dense");
  }
  // TODO: check cuda case
}

ProcessGroupUCC::WorkUCC::~WorkUCC() {
#ifdef USE_CUDA
  if (fence && ep) {
    std::lock_guard<std::mutex> lock(ep->event_pool_mutex);
    ep->event_pool.push(std::move(fence));
  }
#endif
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  return ((status_ == UCC_OK) || (status_ < 0));
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  return (status_ >= 0);
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
#ifdef USE_CUDA
  if (fence && !torch_ucc_config.blocking_wait[(int)opType_]) {
    // block user stream
    fence->block(at::cuda::getCurrentCUDAStream());
    return true;
  }
#endif
  // wait for complete
  while (!isCompleted())
    ;
  return true;
}

void ProcessGroupUCC::WorkUCC::finishWorkUCC(ucc_status_t status,
                                             std::exception_ptr eptr,
                                             std::vector<at::Tensor> output) {
  status_ = status;
  if (future_ && !future_->completed()) {
    if (eptr) {
      future_->setError(eptr);
    } else if (!early_future_complete) {
      future_->markCompleted(c10::IValue(output));
    }
  }
  finish(eptr);
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupUCC::WorkUCC::getFuture() {
  return future_;
}

void ProcessGroupUCC::ProgressEntry::finalize(std::exception_ptr eptr) {
  //TODO: is weak intrusive ptr lock thread safe?
  auto work = work_.lock();
  if (work) {
    work->finishWorkUCC(
        request_->status, eptr, data ? data->dst : std::vector<at::Tensor>());
  }
  comm_->free_request(request_);
}

CommPG::CommPG(
    uint32_t comm_id,
    int rank,
    int size,
    c10::intrusive_ptr<Store> str,
    c10::Device dev)
    : store(str),
      ucx_comm(size),
      ucc_comm(),
      cuda_device_index(TORCH_UCC_DEVICE_NOT_SET) {
  ucx_connect_eps(eps, rank, size, store);
  ucx_oob.comm_id = comm_id;
  ucx_oob.rank = rank;
  ucx_oob.size = size;
  ucx_oob.worker = ucx_comm.worker;
  ucx_oob.eps = eps;
  ucc_comm.create_context(&ucx_oob);
  if (dev.is_cuda()) {
    cuda_device_index = dev.index();
  }
  stop_progress_loop = false;
  progress_thread = std::thread(&CommPG::progress_loop, this);
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
}

CommPG::~CommPG() {
  ucc_comm.destroy_context();
  ucx_disconnect_eps(eps, ucx_oob.rank, ucx_oob.size, store);
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
}

void CommPG::store_barrier() {
  static int seq_num;
  std::string seq_num_str = std::to_string(seq_num);

  if ((size_t)store->add("epclosed" + seq_num_str, 1) == eps.size()) {
    store->add("epfinished" + seq_num_str, 1);
  } else {
    store->wait({"epfinished" + seq_num_str});
  }
  seq_num++;
}

std::shared_ptr<CommPG> CommPG::get_comm(
    uint32_t& id,
    c10::Device dev,
    int rank,
    int size,
    c10::intrusive_ptr<Store> store) {
  static std::mutex m;
  static std::weak_ptr<CommPG> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  id = (comm_id++ % TORCH_UCX_COMM_BITS);
  std::shared_ptr<CommPG> shared_comm = comm.lock();
  if (!shared_comm) {
    shared_comm = std::make_shared<CommPG>(id, rank, size, store, dev);
    comm = shared_comm;
  } else {
    if (dev.is_cuda()) {
      if ((shared_comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
          (shared_comm->cuda_device_index != dev.index())) {
        LOG(ERROR)
            << "ucc communicator was initialized with different cuda device,"
            << "multi device is not supported";
        throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      shared_comm->cuda_device_index = dev.index();
    }
  }
  return shared_comm;
}

void CommPG::ucx_get_eps(std::vector<ucp_ep_h>& ucx_eps, int size) {
  if ((size_t)size != eps.size()) {
    LOG(ERROR) << "creating subgroups is not supported";
    throw std::runtime_error("not supported");
  }
  ucx_eps = eps;
}

void CommPG::ucx_connect_eps(
    std::vector<ucp_ep_h>& eps,
    int rank,
    int size,
    c10::intrusive_ptr<Store> store) {
  ucs_status_t st;
  ucp_address_t* local_addr;
  size_t local_addr_len;
  std::vector<uint8_t> peer_addr;

  st = ucp_worker_get_address(ucx_comm.worker, &local_addr, &local_addr_len);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to get worker address";
    throw std::runtime_error(ucs_status_string(st));
  }
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store->set("wa" + std::to_string(rank), val);
  ucp_worker_release_address(ucx_comm.worker, local_addr);
  eps.resize(size);
  for (int i = 0; i < size; i++) {
    peer_addr = store->get("wa" + std::to_string(i));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    st = ucp_ep_create(ucx_comm.worker, &ep_params, &(eps[i]));
    if (st != UCS_OK) {
      LOG(ERROR) << "failed to create endpoint";
      throw std::runtime_error(ucs_status_string(st));
    }
  }
}

void CommPG::ucx_disconnect_eps(
    std::vector<ucp_ep_h>& eps,
    int rank,
    int size,
    c10::intrusive_ptr<Store> store) {
  ucs_status_t st;

  for (ucp_ep_h& ep : eps) {
    ucs_status_ptr_t close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      LOG(ERROR) << "failed to close endpoint";
      return;
    }
    if (UCS_PTR_IS_PTR(close_req)) {
      do {
        ucp_worker_progress(ucx_comm.worker);
        st = ucp_request_check_status(close_req);
      } while (st != UCS_OK);
      ucp_request_free(close_req);
    }
  }
  store_barrier();
}

ucc_coll_req_h CommPG::send_nb(
    ucp_ep_h ep,
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    ucp_tag_t ucp_tag) {
  ucs_status_ptr_t st;
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.memory_type = mtype;
  params.cb.send = [](void* request, ucs_status_t status, void* user_data) {
    static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
  };
  st = ucp_tag_send_nbx(ep, data, 1, ucp_tag, &params);
  if (UCS_PTR_IS_ERR(st)) {
    LOG(ERROR) << "failed to send message: "
               << ucs_status_string(UCS_PTR_STATUS(st));
    throw std::runtime_error(ucs_status_string(UCS_PTR_STATUS(st)));
  }
  return reinterpret_cast<ucc_coll_req_h>(st);
}

ucc_coll_req_h CommPG::recv_nb(
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    ucp_tag_t ucp_tag,
    ucp_tag_t ucp_tag_mask) {
  ucs_status_ptr_t st;
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.cb.recv = [](void* request,
                      ucs_status_t status,
                      const ucp_tag_recv_info_t* info,
                      void* user_data) {
    static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
  };
  params.memory_type = mtype;
  st = ucp_tag_recv_nbx(
      ucx_comm.worker, data, 1, ucp_tag, ucp_tag_mask, &params);
  if (UCS_PTR_IS_ERR(st)) {
    LOG(ERROR) << "failed to recv message: "
               << ucs_status_string(UCS_PTR_STATUS(st));
    throw std::runtime_error(ucs_status_string(UCS_PTR_STATUS(st)));
  }
  return reinterpret_cast<ucc_coll_req_h>(st);
}

void CommPG::ucc_create_team(ucc_team_h& team) {
  ucc_coll_args_t coll;
  ucc_status_t st;
  ucc_coll_req_h request;

  ucc_comm.create_team(&team, &ucx_oob);
  coll.mask = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  st = ucc_collective_init(&coll, &request, team);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init collective: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_collective_post(request);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to post collective: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  do {
    ucx_comm.progress();
    ucc_comm.progress();
    st = ucc_collective_test(request);
  } while (st == UCC_INPROGRESS);
  ucc_collective_finalize(request);
}

void CommPG::ucc_destroy_team(ucc_team_h& team) {
  ucc_comm.destroy_team(team);
}

c10::intrusive_ptr<ProcessGroup::Work> CommPG::enqueue_p2p(
    OpType opType,
    ucc_coll_req_h request,
    const char* prof_title) {
  if (request == nullptr) {
    // p2p2 request completed immediately don't save it to progress queue
    return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        opType, UCC_OK, prof_title);
  }
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
      opType, UCC_INPROGRESS, prof_title);
  auto entry = std::make_shared<ProcessGroupUCC::ProgressEntry>(
      &ucx_comm, request, work);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(entry);
  lock.unlock();
  queue_produce_cv.notify_one();
  return work;
}

void CommPG::enqueue_collective(
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
    ucc_coll_args_t& coll,
    ucc_team_h team) {
  ucc_coll_req_h request;
  ucc_status_t st;
  st = ucc_collective_init(&coll, &request, team);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init collective: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_collective_post(request);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to post collective: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  auto entry = std::make_shared<ProcessGroupUCC::ProgressEntry>(
      &ucc_comm, request, work);
  entry->data = std::move(data);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(entry);
  lock.unlock();
  queue_produce_cv.notify_one();
}

#ifdef USE_CUDA
void CommPG::enqueue_cuda_collective(
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
    ucc_coll_args_t& coll,
    ucc_team_h team,
    ucc_ee_h ee) {
  ucc_coll_req_h request;
  ucc_status_t st;
  st = ucc_collective_init(&coll, &request, team);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init collective: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  ucc_ev_t comp_ev, *post_ev;
  comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
  comp_ev.ev_context = nullptr;
  comp_ev.ev_context_size = 0;
  comp_ev.req = request;
  st = ucc_collective_triggered_post(ee, &comp_ev);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to post triggered collective: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_ee_get_event(ee, &post_ev);
  TORCH_CHECK(st == UCC_OK && post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
  ucc_ee_ack_event(ee, post_ev);
  auto entry = std::make_shared<ProcessGroupUCC::ProgressEntry>(
      &ucc_comm, request, work);
  entry->data = std::move(data);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(entry);
  lock.unlock();
  queue_produce_cv.notify_one();
}
#endif

void CommPG::progress_loop() {
  std::unique_lock<std::mutex> lock(mutex);
#ifdef USE_CUDA
  bool device_set = false;
#endif
  while (!stop_progress_loop) {
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    auto work = progress_queue.front();
    progress_queue.pop_front();
    lock.unlock();
    queue_consume_cv.notify_one();
#ifdef USE_CUDA
    if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
      c10::cuda::set_device(cuda_device_index);
      device_set = true;
    }
#endif
    std::exception_ptr eptr;
    try {
      while (work->request_->status > 0) {
        work->comm_->progress();
      }
    } catch (...) {
      eptr = std::current_exception();
    }
    work->finalize(eptr);
    lock.lock();
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store) {
  std::call_once(torch_ucc_config.flag, read_confg);
  comm = nullptr;
  cuda_ee = nullptr;
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (comm) {
    comm->ucc_destroy_team(team);
    if (cuda_ee) {
      ucc_ee_destroy(cuda_ee);
    }
    comm = nullptr;
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::collective_post(
    OpType opType,
    ucc_coll_args_t& coll,
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::Device dev,
    std::vector<at::Tensor> &future_tensor,
    const char* prof_title) {
  auto work =
      c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
          opType,
          UCC_INPROGRESS,
          torch_ucc_config.enable_profiling ? prof_title : nullptr);

  switch (dev.type()) {
    case c10::DeviceType::CPU: {
      if (torch_ucc_config.use_future) {
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
      }
      comm->enqueue_collective(std::move(data), work, coll, team);
      return work;
    }
#ifdef USE_CUDA
    case c10::DeviceType::CUDA: {
      std::unique_ptr<at::cuda::CUDAEvent> cuda_ev;
      {
        std::lock_guard<std::mutex> lock(ep.event_pool_mutex);
        if (ep.event_pool.empty()) {
          cuda_ev = std::make_unique<at::cuda::CUDAEvent>();
        } else {
          cuda_ev = std::move(ep.event_pool.front());
          ep.event_pool.pop();
        }
      }
      cuda_ev->record(at::cuda::getCurrentCUDAStream(dev.index()));
      cuda_ev->block(*stream);
      work->early_future_complete = true;
      if (torch_ucc_config.use_future) {
        c10::cuda::CUDAMultiStreamGuard streamGuard(*stream);
        std::vector<c10::Device> devList{dev};
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()), devList);
      }
      comm->enqueue_cuda_collective(std::move(data), work, coll, team, cuda_ee);
      cuda_ev->record(*stream);
      work->fence = std::move(cuda_ev);
      work->ep = &ep;
      if (torch_ucc_config.use_future) {
        work->future_->markCompleted(c10::IValue(future_tensor));
      }
      return work;
    }
#endif // #ifdef USE_CUDA
    default: {
      LOG(ERROR) << "unsupported device type " << dev;
      throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
    }
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  auto& tensor = inputTensors[0];
  check_device(tensor.device(), outputTensors[0][0].device());
  initComm(tensor.device());

  AllgatherWorkData* data = new AllgatherWorkData(size_);
  for (int i = 0; i < size_; i++) {
    data->recv_lengths[i] = tensor.element_size() * tensor.numel();
    data->recv_offsets[i] = (uint64_t)outputTensors[0][i].data_ptr();
  }
  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_ALLGATHERV;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.element_size() * tensor.numel();
  coll.src.info.datatype = UCC_DT_UINT8;
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.dst.info_v.buffer = nullptr;
  coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
  coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
  coll.dst.info_v.datatype = UCC_DT_UINT8;
  coll.dst.info_v.mem_type =
      ucc_mtype_map.at(outputTensors[0][0].device().type());
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors[0], data->dst);

  return collective_post(
      OpType::ALLGATHER,
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      outputTensors[0],
      "ucc:allgather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask =
      UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS | UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_ALLREDUCE;
  coll.reduce.predefined_op = ucc_op_map.at(opts.reduceOp);
  coll.src.info.buffer = nullptr;
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.dst.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  SAVE_TENSORS(tensors, data->dst);

  return collective_post(
      OpType::ALLREDUCE,
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      "ucc:allreduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_device(inputTensor.device(), outputTensor.device());
  initComm(inputTensor.device());
  ucc_coll_args_t coll;
  AlltoallWorkData* data;

  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    data = new AlltoallWorkData(0);
    TORCH_CHECK(
        (outputTensor.size(0) % size_ == 0) &&
            (inputTensor.size(0) % size_ == 0),
        "Tensor's dim 0 does not divide equally across group size");
    coll.mask = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALL;
    coll.src.info.buffer = inputTensor.data_ptr();
    coll.src.info.count =
        inputTensor.element_size() * inputTensor.numel() / size_;
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = ucc_mtype_map.at(inputTensor.device().type());
    coll.dst.info.buffer = outputTensor.data_ptr();
    coll.dst.info.count =
        outputTensor.element_size() * outputTensor.numel() / size_;
    coll.dst.info.datatype = UCC_DT_UINT8;
    coll.dst.info.mem_type = ucc_mtype_map.at(outputTensor.device().type());
  } else {
    data = new AlltoallWorkData(size_);
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    computeLengthsAndOffsets(
        outputSplitSizes,
        outputTensor,
        &data->recv_lengths,
        &data->recv_offsets);
    computeLengthsAndOffsets(
        inputSplitSizes, inputTensor, &data->send_lengths, &data->send_offsets);
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    coll.src.info_v.buffer = inputTensor.data_ptr();
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = ucc_dtype_map.at(inputTensor.scalar_type());
    coll.src.info_v.mem_type = ucc_mtype_map.at(inputTensor.device().type());
    coll.dst.info_v.buffer = outputTensor.data_ptr();
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = ucc_dtype_map.at(outputTensor.scalar_type());
    coll.dst.info_v.mem_type = ucc_mtype_map.at(outputTensor.device().type());
    coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
        UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
  }
  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::ALLTOALL_BASE,
      coll,
      std::unique_ptr<WorkData>(data),
      inputTensor.device(),
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& /* unused */) {
  initComm(c10::DeviceType::CPU);

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  auto dummy_tensor = std::vector<at::Tensor>();
  return collective_post(
      OpType::BARRIER, coll, nullptr, c10::DeviceType::CPU, dummy_tensor, "ucc:barrier");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.root = opts.rootRank;
  SAVE_TENSORS(tensors, data->dst);

  return collective_post(
      OpType::BROADCAST,
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      "ucc:broadcast");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag;
  TORCH_UCX_MAKE_SEND_TAG(ucp_tag, tag, rank_, comm_id);
  ucc_coll_req_h request = comm->send_nb(
      eps[dstRank],
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag);
  return comm->enqueue_p2p(OpType::SEND, request, "ucc:send");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag, ucp_tag_mask;
  TORCH_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, srcRank, comm_id);
  ucc_coll_req_h request = comm->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return comm->enqueue_p2p(OpType::RECV, request, "ucc:recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag, ucp_tag_mask;
  TORCH_UCX_MAKE_RECV_TAG(
      ucp_tag, ucp_tag_mask, tag, TORCH_UCX_ANY_SOURCE, comm_id);
  ucc_coll_req_h request = comm->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return comm->enqueue_p2p(OpType::RECVANYSOURCE, request, "ucc:recv");
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size);
}

void ProcessGroupUCC::initComm(c10::Device dev) {
  if (!comm) {
#ifdef USE_CUDA
    if (dev.is_cuda()) {
      c10::cuda::set_device(dev.index());
    }
#endif
    comm = CommPG::get_comm(comm_id, dev, rank_, size_, store_);
    comm->ucx_get_eps(eps, size_);
    comm->ucc_create_team(team);
  } else {
    if (dev.is_cuda()) {
      if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
          (comm->cuda_device_index != dev.index())) {
        LOG(ERROR)
            << "ucc communicator was initialized with different cuda device, "
            << "multi device is not supported";
        throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      comm->cuda_device_index = dev.index();
    }
  }
#ifdef USE_CUDA
  if (!cuda_ee && dev.is_cuda()) {
    ucc_status_t st;
    stream = std::make_unique<at::cuda::CUDAStream>(
        at::cuda::getStreamFromPool(true, dev.index()));
    ucc_ee_params_t params;
    params.ee_type = UCC_EE_CUDA_STREAM;
    params.ee_context = (void*)stream->stream();
    params.ee_context_size = sizeof(cudaStream_t);
    st = ucc_ee_create(team, &params, &cuda_ee);
    if (st != UCC_OK) {
      LOG(ERROR) << "failed to create UCC EE: " << ucc_status_string(st);
      throw std::runtime_error(ucc_status_string(st));
    }
  }
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
