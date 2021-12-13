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

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_TAG_BITS 32
#define TORCH_UCX_OOB_BITS 1

#define TORCH_UCX_COMM_BITS_OFFSET 0
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_TAG_BITS_OFFSET (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS)
#define TORCH_UCX_OOB_BITS_OFFSET \
  (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS + TORCH_UCX_TAG_BITS)

#define TORCH_UCX_MAX_COMM ((((uint64_t)1) << TORCH_UCX_COMM_BITS) - 1)
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_MAX_TAG ((((uint64_t)1) << TORCH_UCX_TAG_BITS) - 1)
#define TORCH_UCX_MAX_OOB ((((uint64_t)1) << TORCH_UCX_OOB_BITS) - 1)

#define TORCH_UCX_COMM_MASK (TORCH_UCX_MAX_COMM << TORCH_UCX_COMM_BITS_OFFSET)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)
#define TORCH_UCX_TAG_MASK (TORCH_UCX_MAX_TAG << TORCH_UCX_TAG_BITS_OFFSET)
#define TORCH_UCX_OOB_MASK (TORCH_UCX_MAX_OOB << TORCH_UCX_OOB_BITS_OFFSET)

namespace c10d {

// Macro to throw on a non-successful UCC return value.
#define TORCH_UCC_CHECK(_cmd, _error_msg) \
  do {                                    \
    ucc_status_t result = _cmd;           \
    if (result != UCC_OK) {               \
      std::string err = c10::str(         \
          logger->getLogPrefix(),         \
          "UCC error: ",                  \
          _error_msg,                     \
          "\n in: ",                      \
          std::string(__FILE__),          \
          ":",                            \
          std::to_string(__LINE__),       \
          ":\n error code ",              \
          result,                         \
          "(",                            \
          ucc_status_string(result),      \
          ")");                           \
      TORCH_CHECK(false, err);            \
    }                                     \
  } while (0)

// Macro to throw on a non-successful UCX return value.
#define TORCH_UCX_CHECK(_cmd, _error_msg) \
  do {                                    \
    ucs_status_t result = _cmd;           \
    if (result != UCS_OK) {               \
      std::string err = c10::str(         \
          logger->getLogPrefix(),         \
          "UCX error: ",                  \
          _error_msg,                     \
          "\n in: ",                      \
          std::string(__FILE__),          \
          ":",                            \
          std::to_string(__LINE__),       \
          ":\n error code ",              \
          result,                         \
          "(",                            \
          ucs_status_string(result),      \
          ")");                           \
      TORCH_CHECK(false, err);            \
    }                                     \
  } while (0)

// Macros to print logs with unified format
#define TORCH_UCC_LOG_ERROR(_phase, _msg) \
  LOG(ERROR) << logger->getLogPrefix(_phase) << "[ERROR] " << _msg;
#define TORCH_UCC_LOG_INFO(_phase, _msg) \
  LOG(INFO) << logger->getLogPrefix(_phase) << "[INFO] " << _msg;
#define TORCH_UCC_LOG_DEBUG(_phase, _msg)  \
  VLOG(1) << logger->getLogPrefix(_phase) << "[DEBUG] " << _msg;

enum torch_ucc_phase_t {
  TORCH_UCC_UNKNOWN,
  TORCH_UCC_INIT,
  TORCH_UCC_READY,
  TORCH_UCC_COLL_POST,
  TORCH_UCC_COLL_PROGRESS,
  TORCH_UCC_FINALIZE
};

const std::map<torch_ucc_phase_t, std::string> ucc_phase_map = {
    {TORCH_UCC_UNKNOWN, "UNKNOWN"},
    {TORCH_UCC_INIT, "INIT"},
    {TORCH_UCC_READY, "READY"},
    {TORCH_UCC_COLL_POST, "COLL_POST"},
    {TORCH_UCC_COLL_PROGRESS, "COLL_PROGRESS"},
    {TORCH_UCC_FINALIZE, "FINALIZE"},
};

class TORCH_API ProcessGroupUCCLogger : public torch::CustomClassHolder {
 public:
  ProcessGroupUCCLogger();
  ProcessGroupUCCLogger(std::string log_prefix, torch_ucc_phase_t phase);

  std::string getLogPrefix(torch_ucc_phase_t phase = TORCH_UCC_UNKNOWN);
  void setLogPrefix(std::string log_prefix);
  inline void setPhase(torch_ucc_phase_t phase) {
    local_phase = phase;
  }

 protected:
  std::string log_prefix;
  torch_ucc_phase_t local_phase = TORCH_UCC_UNKNOWN;
};

struct torch_ucc_oob_coll_info_t {
  c10::intrusive_ptr<Store> store;
  uint32_t comm_id;
  int rank;
  int size;
  void* rbuf;
  size_t msglen;
  std::string getKey(std::string key) {
    return std::to_string(comm_id) + key;
  }
};

class CommBase {
 public:
  CommBase(const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_)
      : logger(logger_) {}
  virtual void progress() = 0;
  virtual void free_request(ucc_coll_req_h request) = 0;
  virtual ~CommBase() {}
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};

class CommUCX : public CommBase {
 public:
  ucp_context_h context{nullptr};
  ucp_worker_h worker{nullptr};

 public:
  void progress() override;
  void free_request(ucc_coll_req_h request) override;
  CommUCX(
      int comm_size,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  ~CommUCX();
};

class CommUCC : public CommBase {
 public:
  ucc_lib_h lib{nullptr};
  ucc_context_h context{nullptr};

 public:
  void progress() override;
  CommUCC(
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  void free_request(ucc_coll_req_h request) override;
  ~CommUCC();
};

ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req);

ucc_status_t oob_allgather_test(void* req);

ucc_status_t oob_allgather_free(void* req);

} // namespace c10d
