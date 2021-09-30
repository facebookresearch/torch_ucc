/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#include "torch_ucc_comm.hpp"

namespace c10d {

CommUCX::CommUCX(int comm_size, const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger) {
  ucp_params_t params;
  ucp_config_t* config;
  ucs_status_t st;
  ucp_worker_params_t worker_params;

  TORCH_UCX_CHECK(
      ucp_config_read("TORCH", nullptr, &config), "failed to read UCP config");

  memset(&params, 0, sizeof(ucp_params_t));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
      UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | UCP_PARAM_FIELD_TAG_SENDER_MASK |
      UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
  params.request_size = sizeof(ucc_coll_req_t);
  params.features = UCP_FEATURE_TAG;
  params.estimated_num_eps = comm_size;
  params.tag_sender_mask = TORCH_UCX_RANK_MASK;
  params.request_init = [](void* request) {
    static_cast<ucc_coll_req_h>(request)->status = UCC_INPROGRESS;
  };
  params.request_cleanup = [](void*) {};
  TORCH_UCX_CHECK(
      ucp_init(&params, config, &context), "failed to init UCP context");
  ucp_config_release(config);

  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  if (st != UCS_OK) {
    logger->logError(
        TORCH_UCC_INIT,
        c10::str("UCX failed to create UCP worker:", ucs_status_string(st)));
    ucp_cleanup(context);
    throw std::runtime_error(ucs_status_string(st));
  }
}

void CommUCX::progress() {
  ucp_worker_progress(worker);
}

void CommUCX::free_request(ucc_coll_req_h request) {
  request->status = UCC_INPROGRESS;
  ucp_request_free(request);
}

CommUCX::~CommUCX() {
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(coll_info);
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(sbuf),
      reinterpret_cast<uint8_t*>(sbuf) + msglen);
  info->store->set(info->getKey("teamr" + std::to_string(info->rank)), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);

  for (int r = 0; r < info->size; r++) {
    if (!info->store->check({info->getKey("teamr" + std::to_string(r))})) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < info->size; r++) {
    std::vector<uint8_t> data =
        info->store->get(info->getKey("teamr" + std::to_string(r)));
    memcpy(
        (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
        data.data(),
        info->msglen);
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  int num_done = info->store->add({info->getKey("ag_done")}, 1);
  if (num_done == info->size) {
    info->store->deleteKey(info->getKey("ag_done"));
    for (int r = 0; r < info->size; r++) {
      info->store->deleteKey(info->getKey("teamr" + std::to_string(r)));
    }
    for (int r = 0; r < info->size; r++) {
      info->store->add({info->getKey("ag_free" + std::to_string(r))}, 1);
    }
  } else {
    info->store->wait({info->getKey("ag_free" + std::to_string(info->rank))});
  }
  info->store->deleteKey(info->getKey("ag_free" + std::to_string(info->rank)));

  return UCC_OK;
}

CommUCC::CommUCC(torch_ucc_oob_coll_info_t* oob_info, const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  TORCH_UCC_CHECK(
      ucc_lib_config_read("TORCH", nullptr, &lib_config),
      "failed to read UCC lib config");
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  TORCH_UCC_CHECK(
      ucc_init(&lib_params, lib_config, &lib), "failed to init UCC lib");
  ucc_lib_config_release(lib_config);
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  TORCH_UCC_CHECK(
      ucc_lib_get_attr(lib, &lib_attr), "failed to query for lib attr");
  TORCH_CHECK(lib_attr.thread_mode == UCC_THREAD_MULTIPLE, "ucc library wasn't initialized with mt support, please check ucc compile options ");
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    // FIXME: would this cause deadlock if only one rank fails?
    TORCH_UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library when failing to read UCC context config");
    logger->logError(TORCH_UCC_INIT, c10::str("failed to read UCC context config: ", ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_context_config_modify(
      context_config,
      NULL,
      "ESTIMATED_NUM_EPS",
      std::to_string(oob_info->size).c_str());
  if (st != UCC_OK) {
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    logger->logError(TORCH_UCC_INIT, c10::str("UCC failed to modify UCC context config: ", ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.n_oob_eps = oob_info->size;
  context_params.oob.oob_ep = oob_info->rank;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test = oob_allgather_test;
  context_params.oob.req_free = oob_allgather_free;
  context_params.oob.coll_info = oob_info;
  st = ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  if (st != UCC_OK) {
    TORCH_UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library when failing to creat UCC context");
    logger->logError(TORCH_UCC_INIT, c10::str("UCC failed to create UCC context: ", ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }
}

void CommUCC::progress() {
  TORCH_UCC_CHECK(ucc_context_progress(context), "failed to progress UCC collective");
}

void CommUCC::free_request(ucc_coll_req_h request) {
  TORCH_UCC_CHECK(ucc_collective_finalize(request), "failed to release UCC request");
}

CommUCC::~CommUCC() {
  TORCH_UCC_CHECK(ucc_context_destroy(context), "failed to destory UCC context");
  TORCH_UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library");
}

std::string ProcessGroupUCCLogger::getLogPrefix() {
  return log_prefix;
}
void ProcessGroupUCCLogger::setLogPrefix(std::string log_prefix_) {
  log_prefix = log_prefix_;
}

ProcessGroupUCCLogger::ProcessGroupUCCLogger() {
  setLogPrefix("[ProcessGroupUCC]");
}
ProcessGroupUCCLogger::ProcessGroupUCCLogger(std::string log_prefix) {
  setLogPrefix(log_prefix);
}



} // namespace c10d
