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

CommUCX::CommUCX(int comm_size) {
  ucp_params_t params;
  ucp_config_t* config;
  ucs_status_t st;
  ucp_worker_params_t worker_params;

  st = ucp_config_read("TORCH", nullptr, &config);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to read UCP config: " << ucs_status_string(st);
    throw std::runtime_error(ucs_status_string(st));
  }
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
  st = ucp_init(&params, config, &context);
  ucp_config_release(config);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to init UCP context: " << ucs_status_string(st);
    throw std::runtime_error(ucs_status_string(st));
  }
  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to create UCP worker: " << ucs_status_string(st);
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

static ucc_status_t oob_ucx_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  torch_ucc_oob_coll_info_ucx_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_ucx_t*>(coll_info);

  memcpy((char*)rbuf + (ptrdiff_t)(info->rank * msglen), sbuf, msglen);
  info->req.msglen = msglen;
  info->req.done = false;
  info->req.sbuf = sbuf;
  info->req.rbuf = rbuf;
  info->req.iter = 0;
  info->req.sendreq = nullptr;
  info->req.recvreq = nullptr;
  *req = coll_info;
  return UCC_OK;
}

#define TORCH_UCX_MAKE_OOB_TAG(_rank, _comm)             \
  ((((uint64_t)(1)) << TORCH_UCX_OOB_BITS_OFFSET) |      \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_comm)) << TORCH_UCX_COMM_BITS_OFFSET))

static ucc_status_t oob_ucx_allgather_test(void* req) {
  torch_ucc_oob_coll_info_ucx_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_ucx_t*>(req);
  size_t msglen = info->req.msglen;
  int rank = info->rank;
  int size = info->size;

  if (info->req.done) {
    return UCC_OK;
  }
  int sendto = (rank + 1) % size;
  int recvfrom = (rank - 1 + size) % size;
  for (; info->req.iter < size - 1; info->req.iter++) {
    if (info->req.iter > 0) {
      if ((info->req.sendreq != nullptr) &&
          (info->req.sendreq->status == UCC_OK)) {
        info->req.sendreq->status = UCC_INPROGRESS;
        ucp_request_free(info->req.sendreq);
        info->req.sendreq = nullptr;
      }
      if ((info->req.recvreq != nullptr) &&
          (info->req.recvreq->status == UCC_OK)) {
        info->req.recvreq->status = UCC_INPROGRESS;
        ucp_request_free(info->req.recvreq);
        info->req.recvreq = nullptr;
      }

      if ((info->req.sendreq != nullptr) || (info->req.recvreq != nullptr)) {
        ucp_worker_progress(info->worker);
        return UCC_INPROGRESS;
      }
    }
    size_t recvdatafrom = (rank - info->req.iter - 1 + size) % size;
    size_t senddatafrom = (rank - info->req.iter + size) % size;
    char* tmprecv = (char*)info->req.rbuf + (ptrdiff_t)(recvdatafrom * msglen);
    char* tmpsend = (char*)info->req.rbuf + (ptrdiff_t)(senddatafrom * msglen);

    ucs_status_ptr_t ucs_st;
    ucp_request_param_t params;
    ucp_tag_t ucp_tag;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
        UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.datatype = ucp_dt_make_contig(msglen);
    params.memory_type = UCS_MEMORY_TYPE_HOST;
    params.cb.send = [](void* request, ucs_status_t status, void* user_data) {
      static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
    };

    ucp_tag = TORCH_UCX_MAKE_OOB_TAG(rank, info->comm_id);
    ucs_st = ucp_tag_send_nbx(info->eps[sendto], tmpsend, 1, ucp_tag, &params);
    if (UCS_PTR_IS_ERR(ucs_st)) {
      LOG(ERROR) << "failed to send OOB message: "
                 << ucs_status_string(UCS_PTR_STATUS(ucs_st));
      return UCC_ERR_NO_MESSAGE;
    }
    info->req.sendreq = reinterpret_cast<ucc_coll_req_h>(ucs_st);

    params.cb.recv = [](void* request,
                        ucs_status_t status,
                        const ucp_tag_recv_info_t* info,
                        void* user_data) {
      static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
    };
    ucp_tag = TORCH_UCX_MAKE_OOB_TAG(recvfrom, info->comm_id);
    ucs_st = ucp_tag_recv_nbx(
        info->worker, tmprecv, 1, ucp_tag, ((uint64_t)-1), &params);
    if (UCS_PTR_IS_ERR(ucs_st)) {
      LOG(ERROR) << "failed to recv OOB message: "
                 << ucs_status_string(UCS_PTR_STATUS(ucs_st));
      return UCC_ERR_NO_MESSAGE;
    }
    info->req.recvreq = reinterpret_cast<ucc_coll_req_h>(ucs_st);
  }

  if ((info->req.sendreq != nullptr) && (info->req.sendreq->status == UCC_OK)) {
    info->req.sendreq->status = UCC_INPROGRESS;
    ucp_request_free(info->req.sendreq);
    info->req.sendreq = nullptr;
  }
  if ((info->req.recvreq != nullptr) && (info->req.recvreq->status == UCC_OK)) {
    info->req.recvreq->status = UCC_INPROGRESS;
    ucp_request_free(info->req.recvreq);
    info->req.recvreq = nullptr;
  }

  if ((info->req.sendreq != nullptr) || (info->req.recvreq != nullptr)) {
    ucp_worker_progress(info->worker);
    return UCC_INPROGRESS;
  }
  info->req.done = true;

  return UCC_OK;
}

static ucc_status_t oob_ucx_allgather_free(void* req) {
  torch_ucc_oob_coll_info_ucx_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_ucx_t*>(req);
  info->req.msglen = 0;
  info->req.done = false;
  info->req.sbuf = nullptr;
  info->req.rbuf = nullptr;
  return UCC_OK;
}

CommUCC::CommUCC() {
  ucc_lib_config_h lib_config;
  ucc_lib_params_t lib_params;
  ucc_status_t st;

  st = ucc_lib_config_read("TORCH", nullptr, &lib_config);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to read UCC lib config: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  st = ucc_init(&lib_params, lib_config, &lib);
  ucc_lib_config_release(lib_config);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init UCC lib: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  st = ucc_lib_get_attr(lib, &lib_attr);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to query for lib attr: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  if (lib_attr.thread_mode != UCC_THREAD_MULTIPLE) {
    LOG(ERROR) << "ucc library wasn't initialized with mt support "
               << "check ucc compile options ";
    throw std::runtime_error("failed to init ucc lib");
  }
}

void CommUCC::create_context(torch_ucc_oob_coll_info_ucx_t* oob_info) {
  ucc_context_config_h context_config;
  ucc_context_params_t context_params;
  ucc_status_t st;

  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    ucc_finalize(lib);
    LOG(ERROR) << "failed to read UCC context config: "
               << ucc_status_string(st);
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
    LOG(ERROR) << "failed to modify UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.participants = oob_info->size;
  context_params.oob.allgather = oob_ucx_allgather;
  context_params.oob.req_test = oob_ucx_allgather_test;
  context_params.oob.req_free = oob_ucx_allgather_free;
  context_params.oob.coll_info = oob_info;
  ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  if (st != UCC_OK) {
    ucc_finalize(lib);
    LOG(ERROR) << "failed to create UCC context: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
}

void CommUCC::destroy_context() {
  ucc_context_destroy(context);
}

void CommUCC::create_team(
    ucc_team_h* team,
    torch_ucc_oob_coll_info_ucx_t* oob_info) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
      UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_ucx_allgather;
  team_params.oob.req_test = oob_ucx_allgather_test;
  team_params.oob.req_free = oob_ucx_allgather_free;
  team_params.oob.coll_info = oob_info;
  team_params.oob.participants = oob_info->size;
  team_params.ep = oob_info->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  st = ucc_team_create_post(&context, 1, &team_params, team);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to post team create: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  do {
    st = ucc_team_create_test(*team);
    ucc_context_progress(context);
  } while (st == UCC_INPROGRESS);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to create UCC team: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
}

void CommUCC::destroy_team(ucc_team_h team) {
  ucc_status_t st;
  do {
    st = ucc_team_destroy(team);
    ucc_context_progress(context);
  } while (st == UCC_INPROGRESS);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to destroy UCC team: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
}

void CommUCC::progress() {
  ucc_context_progress(context);
}

void CommUCC::free_request(ucc_coll_req_h request) {
  ucc_collective_finalize(request);
}

CommUCC::~CommUCC() {
  ucc_finalize(lib);
}

} // namespace c10d
