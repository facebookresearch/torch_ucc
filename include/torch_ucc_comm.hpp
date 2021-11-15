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

#define TORCH_UCC_TIMEOUT_AM_ID 0

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
  TORCH_UCC_FINALIZE,
  TORCH_UCC_COMM_CHECK
};

const std::map<torch_ucc_phase_t, std::string> ucc_phase_map = {
    {TORCH_UCC_UNKNOWN, "UNKNOWN"},
    {TORCH_UCC_INIT, "INIT"},
    {TORCH_UCC_READY, "READY"},
    {TORCH_UCC_COLL_POST, "COLL_POST"},
    {TORCH_UCC_COLL_PROGRESS, "COLL_PROGRESS"},
    {TORCH_UCC_FINALIZE, "FINALIZE"},
    {TORCH_UCC_COMM_CHECK, "COMM_CHECK"}
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

enum torch_ucc_rank_state_t {
  TORCH_UCC_RANK_STATE_NOT_RESPONDIG,
  TORCH_UCC_RANK_STATE_COLLECTIVE_NOT_POSTED,
  TORCH_UCC_RANK_STATE_COLLECTIVE_INPROGRESS,
  TORCH_UCC_RANK_STATE_COLLECTIVE_TIMEOUT,
  TORCH_UCC_RANK_STATE_DEVICE_ERROR,
  TORCH_UCC_RANK_STATE_COLLECTIVE_DONE,
  TORCH_UCC_RANK_STATE_UNKNOWN
};

const char *torch_ucc_rank_state_string(torch_ucc_rank_state_t state);

struct torch_ucc_timeout_desc_t {
  int rank;
  int comm_id;
  uint64_t seq_num;
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
  CommUCX(
      int comm_size,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  void progress() override;
  void free_request(ucc_coll_req_h request) override;
  void set_am_recv_handler(const ucp_am_handler_param_t *params);
  ~CommUCX();
};

class CommUCC : public CommBase {
 public:
  ucc_lib_h lib{nullptr};
  ucc_context_h context{nullptr};

 public:
  CommUCC(
      torch_ucc_oob_coll_info_t* oob_info,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  void progress() override;
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
