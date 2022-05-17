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
#include "torch_ucc_comm.hpp"
#include <memory>

namespace c10d {

namespace {
constexpr int64_t kBusyWaitMillis = 10;

const std::map<c10::DeviceType, ucs_memory_type_t> ucs_mtype_map = {
    {c10::kCPU, UCS_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCS_MEMORY_TYPE_CUDA},
};

ucs_memory_type_t to_ucs_memType(c10::DeviceType _c10_type) {
  if (ucs_mtype_map.find(_c10_type) != ucs_mtype_map.end())
    return ucs_mtype_map.at(_c10_type);
  else
    return UCS_MEMORY_TYPE_UNKNOWN;
}

const std::map<c10::DeviceType, ucc_memory_type_t> ucc_mtype_map = {
    {c10::kCPU, UCC_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCC_MEMORY_TYPE_CUDA},
};

ucc_memory_type_t to_ucc_memType(c10::DeviceType _c10_type) {
  if (ucc_mtype_map.find(_c10_type) != ucc_mtype_map.end())
    return ucc_mtype_map.at(_c10_type);
  else
    return UCC_MEMORY_TYPE_UNKNOWN;
}

const std::map<at::ScalarType, ucc_datatype_t> ucc_dtype_map = {
    {at::kByte, UCC_DT_UINT8},
    {at::kChar, UCC_DT_INT8},
    {at::kHalf, UCC_DT_FLOAT16},
    {at::kBFloat16, UCC_DT_BFLOAT16},
    {at::kDouble, UCC_DT_FLOAT64},
    {at::kFloat, UCC_DT_FLOAT32},
    {at::kInt, UCC_DT_INT32},
    {at::kLong, UCC_DT_INT64},
    {at::kBool, UCC_DT_UINT8},
};

ucc_datatype_t to_ucc_dType(at::Tensor _tensor) {
  if (_tensor.scalar_type() == at::kBool && _tensor.element_size() != 1) {
    TORCH_CHECK(
        false, "Size of Boolean type larger than 1 is not supported in UCC");
  }
  try {
    return ucc_dtype_map.at(_tensor.scalar_type());
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(false, "Not supported data type for UCC");
  }
}

const std::map<ReduceOp, ucc_reduction_op_t> ucc_op_map = {
    {ReduceOp::SUM, UCC_OP_SUM},
    {ReduceOp::PRODUCT, UCC_OP_PROD},
    {ReduceOp::MIN, UCC_OP_MIN},
    {ReduceOp::MAX, UCC_OP_MAX},
    {ReduceOp::BAND, UCC_OP_BAND},
    {ReduceOp::BOR, UCC_OP_BOR},
    {ReduceOp::BXOR, UCC_OP_BXOR},
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 11)
    {ReduceOp::AVG, UCC_OP_AVG},
#endif
};

ucc_reduction_op_t to_ucc_reduceOp(
    const ReduceOp _op,
    const at::ScalarType _dt) {
  if (_dt == at::kBool) {
    if (_op == ReduceOp::SUM) {
      // bitwise or
      return UCC_OP_MAX;
    } else if (_op == ReduceOp::PRODUCT) {
      // bitwise and
      return UCC_OP_MIN;
    }
#if TORCH_VERSION_MAJOR > 1 || \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 11)
    else if (_op == ReduceOp::AVG) {
      TORCH_CHECK(
          false, "Cannot use ReduceOp.AVG with boolean inputs");
    }
#endif
  }

  try {
    return ucc_op_map.at(_op);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(
        false, "Not supported ReduceOp for UCC");
  }
}

struct torch_ucc_config_t {
  std::once_flag flag;
  std::array<bool, 32> blocking_wait;
  bool enable_profiling;
  bool use_future;
  bool shared_comm;
  bool use_allgatherv;
} torch_ucc_config;

std::map<std::string, std::string> torch_ucc_envs_map = {
    {"TORCH_UCC_ALLGATHER_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_ALLGATHER_BASE_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_ALLREDUCE_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_ALLTOALL_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_BCAST_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_GATHER_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_REDUCE_SCATTER_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_SCATTER_BLOCKING_WAIT", "1"},
    {"TORCH_UCC_USE_FUTURE", "1"},
    {"TORCH_UCC_PROFILING_ENABLE", "0"},
    {"TORCH_UCC_TLS", "nccl,ucp"},
    {"TORCH_UCC_SHARED_COMM", "1"},
    {"TORCH_UCC_USE_ALLGATHERV", "0"},
};

} // namespace

void read_confg() {
  // default configuration
  torch_ucc_config.blocking_wait.fill(true);
  torch_ucc_config.enable_profiling = false;
  torch_ucc_config.use_future = true;
  torch_ucc_config.shared_comm = false;
  torch_ucc_config.use_allgatherv = false;

  // read all torch_ucc env. variables and update the map
  char* env;
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    env = std::getenv(torch_ucc_env.first.c_str());
    if (env) {
      torch_ucc_envs_map[torch_ucc_env.first] = std::string(env);
    }
  }
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLGATHER] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ALLGATHER_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::_ALLGATHER_BASE] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ALLGATHER_BASE_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLREDUCE] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ALLREDUCE_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::ALLTOALL_BASE] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ALLTOALL_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::BROADCAST] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_BCAST_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::REDUCE_SCATTER] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_REDUCE_SCATTER_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::SCATTER] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_SCATTER_BLOCKING_WAIT"));
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::GATHER] =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_GATHER_BLOCKING_WAIT"));
  torch_ucc_config.use_future =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_FUTURE"));
  torch_ucc_config.enable_profiling =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_PROFILING_ENABLE"));
  torch_ucc_config.shared_comm =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_SHARED_COMM"));
  torch_ucc_config.use_allgatherv =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_ALLGATHERV"));
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

void ProcessGroupUCC::WorkUCC::setException() {
  if (exception() || !entry_) {
    return;
  }
  exception_ = entry_->eptr_;
}

void ProcessGroupUCC::WorkUCC::setAndThrowException() {
  setException();
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  if (!entry_) {
    return true;
  }
  setException();
  return exception() || entry_->status_ <= 0;
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  if (!entry_) {
    return true;
  }
  return !exception() && entry_->status_ == 0;
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
#ifdef USE_CUDA
  if (fence && !torch_ucc_config.blocking_wait[(int)opType_]) {
    // block user stream
    setAndThrowException();
    fence->block(at::cuda::getCurrentCUDAStream());
    return true;
  }
#endif
  // wait for complete
  while (!isCompleted())
    ;
  setAndThrowException();
  // manually call profiling end callbacks if they are set,
  // since progress thread does not own WorkUCC
  if (ProcessGroup::Work::recordFunctionEndCallback_) {
    ProcessGroup::Work::recordFunctionEndCallback_();
    ProcessGroup::Work::recordFunctionEndCallback_ = nullptr;
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupUCC::WorkUCC::getFuture() {
  return future_;
}

std::vector<at::Tensor> ProcessGroupUCC::WorkUCC::result() {
  return *outputs_;
}

void ProcessGroupUCC::ProgressEntry::finalize(std::exception_ptr eptr) {
  ucc_status_t status = UCC_OK;

  if (request_ != nullptr) {
    status = request_->status;
    comm_->free_request(request_);
  }
  if (eptr) {
    eptr_ = eptr;
  } else {
    status_ = status;
  }
  if (future_) {
    if (eptr) {
      future_->setError(eptr);
    } else {
      future_->markCompleted(
          c10::IValue(data ? data->dst : std::vector<at::Tensor>()));
    }
  }
}

CommPG::CommPG(
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob_,
    c10::Device dev)
    : logger(logger_),
      oob(oob_),
      ucx_comm(oob->size, logger),
      ucc_comm(oob, logger),
      cuda_device_index(TORCH_UCC_DEVICE_NOT_SET) {
  if (dev.is_cuda()) {
    cuda_device_index = dev.index();
  }
  stop_progress_loop = false;
  collective_inprogress = false;
  progress_thread = std::thread(&CommPG::progress_loop, this);
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
}

CommPG::~CommPG() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
}

std::shared_ptr<CommPG> CommPG::get_comm(
    uint32_t& id,
    c10::Device dev,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger) {
  static std::mutex m;
  static std::weak_ptr<CommPG> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  id = (comm_id % TORCH_UCX_MAX_COMM);

  std::vector<uint8_t> remote_comm_id;
  if (oob->rank != 0) {
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set("group_id" + std::to_string(oob->rank), val);
  } else {
    for (int i = 1; i < oob->size; i++) {
      remote_comm_id = oob->store->get("group_id" + std::to_string(i));
      id = std::max(id, *(reinterpret_cast<uint32_t*>(remote_comm_id.data())));
    }
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set("group_id" + std::to_string(oob->rank), val);
  }
  remote_comm_id = oob->store->get("group_id" + std::to_string(0));
  oob->comm_id = *(reinterpret_cast<uint32_t*>(remote_comm_id.data()));
  comm_id = oob->comm_id + 1;

  if (torch_ucc_config.shared_comm) {
    std::shared_ptr<CommPG> shared_comm = comm.lock();
    if (!shared_comm) {
      shared_comm = std::make_shared<CommPG>(
          logger, oob, dev);
      comm = shared_comm;
    } else {
      if (dev.is_cuda()) {
        if ((shared_comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
            (shared_comm->cuda_device_index != dev.index())) {
          TORCH_UCC_LOG_ERROR(
              TORCH_UCC_INIT,
              "ucc communicator was initialized with different cuda device,"
              "multi device is not supported");
          throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        shared_comm->cuda_device_index = dev.index();
      }
    }
    return shared_comm;
  } else {
    return std::make_shared<CommPG>(logger, oob, dev);
  }
}

void CommPG::ucx_connect_eps(
    std::vector<ucp_ep_h>& eps,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob) {
  ucp_address_t* local_addr;
  size_t local_addr_len;
  std::vector<uint8_t> peer_addr;

  TORCH_UCX_CHECK(
      ucp_worker_get_address(ucx_comm.worker, &local_addr, &local_addr_len),
      "failed to get worker address");

  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  oob->store->set(oob->getKey("wa" + std::to_string(oob->rank)), val);
  ucp_worker_release_address(ucx_comm.worker, local_addr);
  eps.resize(oob->size);
  for (int i = 0; i < oob->size; i++) {
    peer_addr = oob->store->get(oob->getKey("wa" + std::to_string(i)));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    TORCH_UCX_CHECK(
        ucp_ep_create(ucx_comm.worker, &ep_params, &(eps[i])),
        c10::str("failed to create endpoint with rank ", i));
  }
}

void CommPG::ucx_disconnect_eps(
    std::vector<ucp_ep_h>& eps,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob) {
  ucs_status_t st;

  for (ucp_ep_h& ep : eps) {
    ucs_status_ptr_t close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_FINALIZE,
          "failed to close endpoint, ignore and continue...");
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
  if (!eps.size()) {
    return;
  }
  try {
    auto sz = (size_t)oob->store->add(oob->getKey("epclosed"), 1);
    while (sz != eps.size()) {
      ucp_worker_progress(ucx_comm.worker);
      std::this_thread::sleep_for(std::chrono::milliseconds(kBusyWaitMillis));
      sz = (size_t)oob->store->add(oob->getKey("epclosed"), 0);
    }
  } catch (std::exception& ex) {
    LOG(ERROR) << "(disconnect_eps) Caught error in Store Operation .. "
               << "[" << ex.what() << "]";
  }
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
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_COLL_POST,
        c10::str(
            "failed to send message: ", ucs_status_string(UCS_PTR_STATUS(st))));
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
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_COLL_POST,
        c10::str(
            "failed to recv message: ", ucs_status_string(UCS_PTR_STATUS(st))));
    throw std::runtime_error(ucs_status_string(UCS_PTR_STATUS(st)));
  }
  return reinterpret_cast<ucc_coll_req_h>(st);
}

void CommPG::ucc_create_team(
    ucc_team_h& team,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
      UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = oob.get();
  team_params.oob.n_oob_eps = oob->size;
  team_params.oob.oob_ep = oob->rank;
  team_params.ep = oob->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  TORCH_UCC_CHECK(
      ucc_team_create_post(&ucc_comm.context, 1, &team_params, &team),
      "failed to post team create");
  do {
    st = ucc_team_create_test(team);
    ucc_context_progress(ucc_comm.context);
  } while (st == UCC_INPROGRESS);
  TORCH_UCC_CHECK(st, "failed to create UCC team");
}

void CommPG::ucc_destroy_team(ucc_team_h& team) {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });

  ucc_status_t status;
  while (UCC_INPROGRESS == (status = ucc_team_destroy(team))) {
    if (UCC_OK != status) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_FINALIZE,
          c10::str("ucc team destroy error: ", ucc_status_string(status)));
      break;
    }
  }

  lock.unlock();
}

c10::intrusive_ptr<ProcessGroup::Work> CommPG::enqueue_p2p(
    OpType opType,
    ucc_coll_req_h request,
    const char* prof_title) {
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(opType, prof_title);
  if (torch_ucc_config.use_future) {
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
  }
  if (request == nullptr) {
    // p2p2 request completed immediately don't save it to progress queue
    // and mark future completed immediately
    if (torch_ucc_config.use_future) {
      work->future_->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    }
    return work;
  }
  auto entry =
      std::make_shared<ProcessGroupUCC::ProgressEntry>(&ucx_comm, request);
  work->entry_ = entry;
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
  TORCH_UCC_CHECK(
      ucc_collective_init(&coll, &request, team), "failed to init collective");
  TORCH_UCC_CHECK(ucc_collective_post(request), "failed to post collective");

  auto entry =
      std::make_shared<ProcessGroupUCC::ProgressEntry>(&ucc_comm, request);
  entry->data = std::move(data);
  entry->future_ = work->getFuture();
  work->entry_ = entry;
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
  TORCH_UCC_CHECK(
      ucc_collective_init(&coll, &request, team),
      "failed to init cuda collective");
  ucc_ev_t comp_ev, *post_ev;
  comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
  comp_ev.ev_context = nullptr;
  comp_ev.ev_context_size = 0;
  comp_ev.req = request;
  TORCH_UCC_CHECK(
      ucc_collective_triggered_post(ee, &comp_ev),
      "failed to post triggered collective");
  ucc_status_t st = ucc_ee_get_event(ee, &post_ev);
  TORCH_CHECK(st == UCC_OK && post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST);
  ucc_ee_ack_event(ee, post_ev);
  auto entry =
      std::make_shared<ProcessGroupUCC::ProgressEntry>(&ucc_comm, request);
  entry->data = std::move(data);
  work->entry_ = entry;
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
    collective_inprogress = true;
    auto work = progress_queue.front();
    progress_queue.pop_front();
    lock.unlock();
#ifdef USE_CUDA
    if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
      c10::cuda::set_device(cuda_device_index);
      device_set = true;
    }
#endif
    std::exception_ptr eptr;
    try {
      while (work->request_->status > 0) {
        ucc_comm.progress();
        ucx_comm.progress();
      }
      if (work->request_->status < 0) {
        eptr = std::make_exception_ptr(
            std::runtime_error(ucc_status_string(work->request_->status)));
        std::string err_log = c10::str(
            "Failed to progress communication", // TODO: report exact op type or
                                                // id?
            ucc_status_string(work->request_->status));
        TORCH_UCC_LOG_ERROR(TORCH_UCC_COLL_PROGRESS, err_log);
      }
    } catch (...) {
      eptr = std::current_exception();
    }
    work->finalize(eptr);
    work = nullptr;
    collective_inprogress = false;
    queue_consume_cv.notify_one();
    lock.lock();
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::chrono::duration<float> timeout)
    : ProcessGroup(rank, size), timeout_(timeout) {
  std::call_once(torch_ucc_config.flag, read_confg);
  oob = std::make_shared<torch_ucc_oob_coll_info_t>();
  oob->rank = rank;
  oob->size = size;
  oob->store = store;
  comm = nullptr;
  cuda_ee = nullptr;
  static uint32_t id = 0;
  uint32_t pg_id = (id++ % TORCH_UCX_MAX_COMM);

  logger = c10::make_intrusive<ProcessGroupUCCLogger>(
      c10::str("[Rank ", rank_, "]", "[ProcessGroupUCC-", pg_id, "]"),
      TORCH_UCC_INIT);
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str("Created ProcessGroupUCC with ", size, " ranks, with timeout ", timeout_.count(), " secs"));
  std::string envs = "";
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    envs += ("\n\t" + torch_ucc_env.first + "=" + torch_ucc_env.second);
  }
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str(
          "Successfully read and set ProcessGroupUCC env. variables as followings",
          envs));
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (comm) {
    logger->setPhase(TORCH_UCC_FINALIZE);
    comm->ucc_destroy_team(team);
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_FINALIZE, "Successfully destroyed UCC library");
    comm->ucx_disconnect_eps(eps, oob);
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_FINALIZE, "Successfully destroyed UCX library");
    try {
      if (cuda_ee) {
        ucc_ee_destroy(cuda_ee);
      }
      if ((size_t)oob->store->add(oob->getKey("ucc_pg_closed"), 1) ==
          eps.size()) {
        std::vector<uint8_t> val = {1};
        oob->store->set(oob->getKey("ucc_pg_finished"), val);
      } else {
        oob->store->wait({oob->getKey("ucc_pg_finished")});
      }
    } catch (std::exception& ex) {
      TORCH_UCC_LOG_INFO(
        TORCH_UCC_FINALIZE,
        c10::str(
          "(~ProcessGroupUCC) Caught error in Store Operation .. ",
          "[",
          ex.what(),
          "]"));
    }
    comm = nullptr;
  }
}

void ProcessGroupUCC::set_timeout(ucc_coll_args_t& args) {
  args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
  args.flags |= UCC_COLL_ARGS_FLAG_TIMEOUT;
  args.timeout = timeout_.count();
}

#ifdef USE_CUDA
std::unique_ptr<at::cuda::CUDAEvent> ProcessGroupUCC::getPooledEvent() {
	std::unique_ptr<at::cuda::CUDAEvent> ev;
	std::lock_guard<std::mutex> lock(ep.event_pool_mutex);
	if (ep.event_pool.empty()) {
		ev = std::make_unique<at::cuda::CUDAEvent>();
	} else {
		ev = std::move(ep.event_pool.front());
		ep.event_pool.pop();
	}
	return ev;
}
#endif

template <typename PreProcess, typename PostProcess>
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::collective_post(
    OpType opType,
    PreProcess preproc,
    PostProcess postproc,
    ucc_coll_args_t& coll,
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::Device dev,
    std::vector<at::Tensor> &outputTensors,
    const char* prof_title) {
  set_timeout(coll);
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
      opType, torch_ucc_config.enable_profiling ? prof_title : nullptr);

  // Store references to outputs to be used by result
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputTensors);
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
      auto cuda_ev = getPooledEvent();
      cuda_ev->record(at::cuda::getCurrentCUDAStream(dev.index()));
      cuda_ev->block(*stream);
      at::cuda::CUDAStreamGuard guard(*stream);
      preproc();
      comm->enqueue_cuda_collective(std::move(data), work, coll, team, cuda_ee);
      postproc();
      cuda_ev->record(*stream);
      work->fence = std::move(cuda_ev);
      work->ep = &ep;
      if (torch_ucc_config.use_future) {
        c10::cuda::CUDAMultiStreamGuard streamGuard(*stream);
        std::vector<c10::Device> devList{dev};
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()), devList);
        // Add a callback that runs profiling end callbacks
        if (work->recordFunctionEndCallback_) {
          work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          });
        }

        work->future_->markCompleted(c10::IValue(outputTensors));
      }
      return work;
    }
#endif // #ifdef USE_CUDA
    default: {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, c10::str("unsupported device type ", dev.str()));
      throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
    }
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  if (size_ == 1) {
      outputTensors[0][0].copy_(inputTensors[0]);
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::ALLGATHER,
                torch_ucc_config.enable_profiling ? "ucc:allgather" : nullptr);
  }
  auto& tensor = inputTensors[0];
  check_device(tensor.device(), outputTensors[0][0].device());
  initComm(tensor.device());

  if (tensor.device().is_cpu() || torch_ucc_config.use_allgatherv) {
    AllgathervWorkData* data = new AllgathervWorkData(size_);
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
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info_v.buffer = nullptr;
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    coll.dst.info_v.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());
    SAVE_TENSORS(inputTensors, data->src);
    SAVE_TENSORS(outputTensors[0], data->dst);

    return collective_post(
        OpType::ALLGATHER,
        []() {},
        []() {},
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        outputTensors[0],
        "ucc:allgatherv");
  } else {
    WorkData* data = new WorkData();
    std::vector<at::Tensor> flat_output(outputTensors.size());
    for (size_t i = 0; i < outputTensors.size(); i++) {
      TORCH_CHECK(outputTensors[i].size() == outputTensors.size() * size_,
        "Tensor output list is not valid for the number of participants");
        flat_output[i] = c10d::newLikeFlat(outputTensors, i);
    }
    SAVE_TENSORS(flat_output, data->flat);
    ucc_coll_args_t coll;
    coll.mask = 0;
    coll.flags = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
    coll.src.info.buffer = tensor.data_ptr();
    coll.src.info.count = tensor.numel();
    coll.src.info.datatype = to_ucc_dType(tensor);
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info.buffer = flat_output[0].data_ptr();
    coll.dst.info.count = flat_output[0].numel();
    coll.dst.info.datatype = to_ucc_dType(flat_output[0]);
    coll.dst.info.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());

    auto copy_from_flat = [&] {
      bool asyncCopy = false;
  #ifdef USE_CUDA
      bool isCuda = outputTensors[0][0].device().is_cuda();;
  #endif
      for (size_t i = 0; i < outputTensors.size(); i++) {
        auto inumel = inputTensors[i].numel();
        for (size_t j = 0; j < outputTensors[i].size(); j++) {
          TORCH_CHECK(
            (outputTensors[i][j].numel() == inumel),
            "Tensor operand counts must be same");
  #ifdef USE_CUDA
          if (isCuda) {
            c10::cuda::CUDACachingAllocator::recordStream(
              outputTensors[i][j].storage().data_ptr(), (*stream));
            asyncCopy = true;
          }
  #endif
          outputTensors[i][j].copy_(flat_output[i][j], asyncCopy);
        }
      }
    };
    return collective_post(
        OpType::ALLGATHER,
        []() {},
        copy_from_flat,
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        outputTensors[0],
        "ucc:allgather");
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts) {
  if (size_ == 1) {
    outputTensor.copy_(inputTensor);
    return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
              OpType::_ALLGATHER_BASE,
              torch_ucc_config.enable_profiling ? "ucc:allgather_base" : nullptr);
  }
  check_tensor({outputTensor});
  check_tensor({inputTensor});
  initComm(outputTensor.device());

  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
  coll.src.info.buffer = inputTensor.data_ptr();
  coll.src.info.count = inputTensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(inputTensor.scalar_type());
  coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
  coll.dst.info.buffer = outputTensor.data_ptr();
  coll.dst.info.count = outputTensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(outputTensor.scalar_type());
  coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());

  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::_ALLGATHER_BASE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      outputTensor.device(),
      outputTensors,
      "ucc:allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  if (size_ == 1) {
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::ALLREDUCE,
                torch_ucc_config.enable_profiling ? "ucc:allreduce" : nullptr);
  }
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_ALLREDUCE;
  coll.op = to_ucc_reduceOp(opts.reduceOp, tensor.scalar_type());
  coll.src.info.buffer = nullptr;
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = to_ucc_dType(tensor);
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(tensors, data->dst);
  return collective_post(
      OpType::ALLREDUCE,
      []() {},
      []() {},
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
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }
  if (size_ == 1) {
    for (const auto r : c10::irange(outputTensors.size())) {
      outputTensors[r].copy_(inputTensors[r]);
    }
    return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        OpType::ALLTOALL,
        torch_ucc_config.enable_profiling ? "ucc:alltoall" : nullptr);
  }

  initComm(device);
  ucc_coll_args_t coll;
  AlltoallWorkData* data;
  data = new AlltoallWorkData(size_);

  /* to avoid flatten the tensors, we use alltoallv to achieve Alltoall as
     follow.
      1. store addresses of each tensor directly in displacements, keep buffer
     to nullptr, i.e., 0
      2. convert datatype to UINT8, which is always 1 bytes, to avoid wrong size
     calculation in UCC layer
      3. post Alltoallv
  */
  for (const auto i : c10::irange(size_)) {
    data->send_lengths[i] =
        (uint64_t)(inputTensors[i].element_size() * inputTensors[i].numel());
    data->send_offsets[i] = (uint64_t)inputTensors[i].data_ptr();
    data->recv_lengths[i] =
        (uint64_t)(outputTensors[i].element_size() * outputTensors[i].numel());
    data->recv_offsets[i] = (uint64_t)outputTensors[i].data_ptr();
  }

  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
  coll.src.info_v.buffer = 0;
  coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
  coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
  coll.src.info_v.datatype = UCC_DT_UINT8;
  coll.src.info_v.mem_type = to_ucc_memType(inputTensors[0].device().type());
  coll.dst.info_v.buffer = 0;
  coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
  coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
  coll.dst.info_v.datatype = UCC_DT_UINT8;
  coll.dst.info_v.mem_type = to_ucc_memType(outputTensors[0].device().type());

  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::ALLTOALL,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      device,
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  if (size_ == 1) {
      outputTensor.copy_(inputTensor);
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::ALLTOALL_BASE,
                torch_ucc_config.enable_profiling ? "ucc:alltoall" : nullptr);
  }
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
    coll.src.info.count = inputTensor.element_size() * inputTensor.numel();
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
    coll.dst.info.buffer = outputTensor.data_ptr();
    coll.dst.info.count = outputTensor.element_size() * outputTensor.numel();
    coll.dst.info.datatype = UCC_DT_UINT8;
    coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());
    coll.flags = 0;
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
    coll.src.info_v.datatype = to_ucc_dType(inputTensor);
    coll.src.info_v.mem_type = to_ucc_memType(inputTensor.device().type());
    coll.dst.info_v.buffer = outputTensor.data_ptr();
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = to_ucc_dType(outputTensor);
    coll.dst.info_v.mem_type = to_ucc_memType(outputTensor.device().type());
    coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
        UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER |
        UCC_COLL_ARGS_FLAG_COUNT_64BIT |
        UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  }
  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::ALLTOALL_BASE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      inputTensor.device(),
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& opts) {
  if (size_ == 1) {
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::BARRIER,
                torch_ucc_config.enable_profiling ? "ucc:barrier" : nullptr);
  }
  c10::Device device = c10::Device(c10::DeviceType::CPU);
#ifdef USE_CUDA
  auto numGPUs = at::cuda::getNumGPUs();
  if (!opts.device_ids.empty()) {
    device = c10::Device(c10::DeviceType::CUDA, opts.device_ids.front());
  } else if (comm && comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) {
    device = c10::Device(c10::DeviceType::CUDA, comm->cuda_device_index);
  } else if (numGPUs > 0) {
    int8_t deviceIdx = static_cast<int8_t>(c10::cuda::current_device());
    // if current device is 0, likely the device is not set, use the best guess
    if (0 == (int)deviceIdx) {
      deviceIdx = static_cast<int8_t>(this->getRank() % numGPUs);
    }
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_COLL_POST,
        c10::str(
            "post barrier before specifying any GPU while there are ",
            numGPUs,
            " GPUs available. ",
            "Not clear if GPU barrier is required, using GPU ",
            (int)deviceIdx,
            " to perform barrier. ",
            "Specify device_ids option in barrier() to force ",
            "use of a particular device"));
    device = c10::Device(c10::DeviceType::CUDA, deviceIdx);
  }
#endif
  initComm(device);

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  auto dummy_tensor = std::vector<at::Tensor>();
  return collective_post(
      OpType::BARRIER,
      []() {},
      []() {},
      coll,
      nullptr,
      device,
      dummy_tensor,
      "ucc:barrier");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  if (size_ == 1) {
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::BROADCAST,
                torch_ucc_config.enable_profiling ? "ucc:broadcast" : nullptr);
  }
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.root = opts.rootRank;
  SAVE_TENSORS(tensors, data->dst);

  return collective_post(
      OpType::BROADCAST,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      "ucc:broadcast");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  if (size_ == 1) {
    outputTensors[0][0].copy_(inputTensors[0]);
    return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        OpType::GATHER,
        torch_ucc_config.enable_profiling ? "ucc:gather" : nullptr);
  }
  std::vector<at::Tensor> outputs;
  auto& input = inputTensors[0];
  initComm(input.device());

  AllgathervWorkData* data = new AllgathervWorkData(size_);
  ucc_coll_args_t coll;
  coll.root = opts.rootRank;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_GATHERV;

  /* for non-root ranks, only src is valid */
  coll.src.info.buffer = input.data_ptr();
  coll.src.info.count = (uint64_t)(input.element_size() * input.numel());
  coll.src.info.datatype = UCC_DT_UINT8;
  coll.src.info.mem_type = to_ucc_memType(input.device().type());

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "Incorrect output list size ",
              outputTensors[0].size(),
              ". Output list size should be ",
              getSize(),
              ", same as size of the process group."));
    }
    outputs = outputTensors[0];

    for (int i = 0; i < size_; i++) {
      data->recv_lengths[i] =
          (uint64_t)(outputs[i].element_size() * outputs[i].numel());
      data->recv_offsets[i] = (uint64_t)outputs[i].data_ptr();
    }
    /* use gatherv and store non-contiguous addresses in displacements to avoid
     * flatten outputTensors */
    coll.dst.info_v.buffer = nullptr;
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    coll.dst.info_v.mem_type = to_ucc_memType(outputs[0].device().type());

    SAVE_TENSORS(outputs, data->dst);
  } else {
    // for non-root ranks, outputTensors should be an empty list
    if (outputTensors.size() != 0) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list to be used by future mark
    outputs.emplace_back();
  }

  SAVE_TENSORS(inputTensors, data->src);

  return collective_post(
      OpType::GATHER,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      input.device(),
      outputs,
      "ucc:gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  if (size_ == 1) {
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
                OpType::REDUCE,
                torch_ucc_config.enable_profiling ? "ucc:reduce" : nullptr);
  }
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());
  WorkData* data = new WorkData();

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_REDUCE;
  coll.op = ucc_op_map.at(opts.reduceOp);
  coll.root = opts.rootRank;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(tensors, data->dst);
  return collective_post(
      OpType::REDUCE,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      "ucc:reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
	TORCH_CHECK(
		(outputTensors.size() == inputTensors.size()),
		"Tensor input/output list for reduce_scatter must have same size");
	check_tensor(outputTensors);
	check_device(inputTensors[0][0].device(), outputTensors[0].device());
	initComm(inputTensors[0][0].device());
  auto data = std::make_unique<WorkData>();
	std::vector<at::Tensor> flat_input(inputTensors.size());
  for (size_t i = 0; i < inputTensors.size(); i++) {
    TORCH_CHECK(inputTensors[i].size() == inputTensors.size() * size_,
      "Tensor input list is not valid for the number of participants");
      flat_input[i] = c10d::newLikeFlat(inputTensors, i);
  }
  SAVE_TENSORS(flat_input, data->flat);
	check_tensor(flat_input);
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
	coll.op = to_ucc_reduceOp(opts.reduceOp, flat_input[0].scalar_type());

  coll.src.info.buffer = flat_input[0].data_ptr();
  coll.src.info.count = flat_input[0].numel();
  coll.src.info.datatype = to_ucc_dType(flat_input[0]);
  coll.src.info.mem_type = to_ucc_memType(flat_input[0].device().type());
  coll.dst.info.buffer = outputTensors[0].data_ptr();
  coll.dst.info.count = outputTensors[0].numel();
  coll.dst.info.datatype = to_ucc_dType(outputTensors[0]);
  coll.dst.info.mem_type = to_ucc_memType(outputTensors[0].device().type());

  SAVE_TENSORS(inputTensors[0], data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  auto copy_to_flat = [&] {
    bool asyncCopy = false;
    auto isize = inputTensors.size();
#ifdef USE_CUDA
    bool isCuda = inputTensors[0][0].device().is_cuda();
#endif
    for (size_t i = 0; i < isize; i++) {
      auto onumel = outputTensors[i].numel();
      for (size_t j = 0; j < inputTensors[i].size(); j++) {
        TORCH_CHECK(
          (inputTensors[i][j].numel() == onumel),
          "Tensor operand counts must be same");
#ifdef USE_CUDA
        if (isCuda) {
          c10::cuda::CUDACachingAllocator::recordStream(
            inputTensors[i][j].storage().data_ptr(), (*stream));
          asyncCopy = true;
        }
#endif
        flat_input[i][j].copy_(inputTensors[i][j], asyncCopy);
      }
    }
  };

 return collective_post(
    	OpType::REDUCE_SCATTER,
      copy_to_flat,
      []() {},
    	coll,
    	std::move(data),
    	inputTensors[0][0].device(),
    	outputTensors,
    	"ucc:reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  if (size_ == 1) {
    outputTensors[0].copy_(inputTensors[0][0]);
    return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        OpType::SCATTER,
        torch_ucc_config.enable_profiling ? "ucc:scatter" : nullptr);
  }
  auto& tensor = outputTensors[0];
  initComm(tensor.device());

  ScattervWorkData* data = new ScattervWorkData(size_);
  ucc_coll_args_t coll;
  coll.root = opts.rootRank;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_SCATTERV;

  if (getRank() == opts.rootRank) {
    /* src is only valid at non-root rank */
    if (inputTensors.size() != 1) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      TORCH_UCC_LOG_ERROR(TORCH_UCC_COLL_POST,
        c10::str(
          "Incorrect output list size ", inputTensors[0].size(),
          ". Output list size should be ", getSize(),
          ", same as size of the process group."));
    }

    for (int i = 0; i < size_; i++) {
      data->send_lengths[i] = (uint64_t) tensor.element_size() * tensor.numel();
      data->send_offsets[i] = (uint64_t)inputTensors[0][i].data_ptr();
    }
    /* use scatter and store non-contiguous addresses in displacements to avoid
     * flatten inputTensors */
    coll.src.info_v.buffer = nullptr;
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = UCC_DT_UINT8;
    coll.src.info_v.mem_type =
        to_ucc_memType(inputTensors[0][0].device().type());

    SAVE_TENSORS(inputTensors[0], data->src);
  } else {
    // for non-root ranks, inputTensors should be an empty list
    if (inputTensors.size() != 0) {
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
    }
  }

  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = (uint64_t) tensor.element_size() * tensor.numel();
  coll.dst.info.datatype = UCC_DT_UINT8;
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
  SAVE_TENSORS(outputTensors, data->dst);

  return collective_post(
      OpType::SCATTER,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      outputTensors,
      "ucc:scatter");
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
      to_ucs_memType(tensor.device().type()),
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
      to_ucs_memType(tensor.device().type()),
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
      to_ucs_memType(tensor.device().type()),
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
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size, timeout);
}

void ProcessGroupUCC::initComm(c10::Device dev) {
  if (!comm) {
#ifdef USE_CUDA
    if (dev.is_cuda()) {
      c10::cuda::set_device(dev.index());
    }
#endif
    comm = CommPG::get_comm(comm_id, dev, oob, logger);
    comm->ucx_connect_eps(eps, oob);
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCX library");
    comm->ucc_create_team(team, oob);
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCC library");
    logger->setPhase(TORCH_UCC_READY);
  } else {
    if (dev.is_cuda()) {
      if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
          (comm->cuda_device_index != dev.index())) {
        TORCH_UCC_LOG_ERROR(
            TORCH_UCC_INIT,
            "ucc communicator was initialized with different cuda device,"
            "multi device is not supported");
        throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      comm->cuda_device_index = dev.index();
    }
  }
#ifdef USE_CUDA
  if (!cuda_ee && dev.is_cuda()) {
    stream = std::make_unique<at::cuda::CUDAStream>(
        at::cuda::getStreamFromPool(true, dev.index()));
    ucc_ee_params_t params;
    params.ee_type = UCC_EE_CUDA_STREAM;
    params.ee_context = (void*)stream->stream();
    params.ee_context_size = sizeof(cudaStream_t);
    TORCH_UCC_CHECK(
        ucc_ee_create(team, &params, &cuda_ee),
        "failed to create UCC execution engine");
  }
#endif
}

} // namespace c10d
