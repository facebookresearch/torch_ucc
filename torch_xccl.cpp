/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include "torch_xccl.hpp"

namespace c10d {

struct xccl_oob_allgather_req_t {
    xccl_ep_range_t     range;
    void                *sbuf;
    void                *rbuf;
    void                *oob_coll_ctx;
    int                 my_rank;
    size_t              msglen;
    int                 iter;
    torch_ucx_request_t *reqs[2];
};

static xccl_status_t oob_allgather_test(void *req)
{
  xccl_oob_allgather_req_t *oob_req = static_cast<xccl_oob_allgather_req_t*>(req);
  int rank, size, sendto, recvfrom, recvdatafrom, senddatafrom;
  torch_ucx_comm_t *oob_ctx = static_cast<torch_ucx_comm_t*>(oob_req->oob_coll_ctx);
  char *tmpsend = NULL, *tmprecv = NULL;
  size_t msglen = oob_req->msglen;
  torch_ucx_status_t st;

  if (oob_req->range.type == XCCL_EP_RANGE_UNDEFINED) {
    size = oob_ctx->size;
    rank = oob_ctx->rank;
  } else {
    size = oob_req->range.ep_num;
    rank = oob_req->my_rank;
  }

  if (oob_req->iter == 0) {
    tmprecv = (char*) oob_req->rbuf + (ptrdiff_t)(rank * msglen);
    memcpy(tmprecv, oob_req->sbuf, msglen);
  }
  sendto   = (rank + 1) % size;
  recvfrom = (rank - 1 + size) % size;
  if (oob_req->range.type != XCCL_EP_RANGE_UNDEFINED) {
    sendto   = xccl_range_to_rank(oob_req->range, sendto);
    recvfrom = xccl_range_to_rank(oob_req->range, recvfrom);
  }
  for (; oob_req->iter < size - 1; oob_req->iter++) {
    if (oob_req->iter > 0) {
      st = torch_ucx_req_test(oob_ctx, oob_req->reqs, 2, NULL, 1, 2);
      if (st == TORCH_UCX_INPROGRESS) {
        return XCCL_INPROGRESS;
      }
    }
    recvdatafrom = (rank - oob_req->iter - 1 + size) % size;
    senddatafrom = (rank - oob_req->iter + size) % size;
    tmprecv = (char*)oob_req->rbuf + (ptrdiff_t)(recvdatafrom * msglen);
    tmpsend = (char*)oob_req->rbuf + (ptrdiff_t)(senddatafrom * msglen);

    torch_ucx_send_nb(oob_ctx, tmpsend, msglen, sendto, 1,
                      &oob_req->reqs[0], TORCH_UCX_OOB_TAG);

    torch_ucx_recv_nb(oob_ctx, tmprecv, msglen, recvfrom, 1,
                      &oob_req->reqs[1], TORCH_UCX_OOB_TAG);
  }

  st = torch_ucx_req_test(oob_ctx, oob_req->reqs, 2, NULL, 1, 2);
  if (st == TORCH_UCX_INPROGRESS) {
    return XCCL_INPROGRESS;
  }

  return XCCL_OK;
}

static xccl_status_t oob_allgather_free(void *req)
{
  xccl_oob_allgather_req_t *request = static_cast<xccl_oob_allgather_req_t*>(req);
  delete request;

  return XCCL_OK;
}

static int oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                         int my_rank, xccl_ep_range_t range,
                         void *oob_coll_ctx, void **req)
{
  xccl_oob_allgather_req_t *oob_req = new(xccl_oob_allgather_req_t);
  oob_req->sbuf         = sbuf;
  oob_req->rbuf         = rbuf;
  oob_req->msglen       = msglen;
  oob_req->range        = range;
  oob_req->oob_coll_ctx = oob_coll_ctx;
  oob_req->my_rank      = my_rank;
  oob_req->iter         = 0;
  *req = oob_req;

  return oob_allgather_test(oob_req);
}

torch_ucx_status_t torch_xccl_comm_init(torch_ucx_comm_t *p2p_comm,
                                        torch_xccl_comm_t **comm)
{
    torch_xccl_comm_t *xccl_comm;
    xccl_lib_params_t lib_params;
    xccl_lib_config_t *cfg;
    xccl_status_t     st;
  
    xccl_comm = new torch_xccl_comm_t;
    memset(&lib_params, 0, sizeof(lib_params));
    lib_params.field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE |
                            XCCL_LIB_PARAM_FIELD_COLL_TYPES;
  
    lib_params.team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                            XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES;
  
    lib_params.coll_types = XCCL_COLL_CAP_BCAST |
                            XCCL_COLL_CAP_ALLREDUCE |
                            XCCL_COLL_CAP_ALLTOALL |
                            XCCL_COLL_CAP_ALLTOALLV;
  
    cfg = NULL;
    st = xccl_lib_init(&lib_params, cfg, &xccl_comm->xccl_lib);
    if (st != XCCL_OK) {
        fprintf(stderr, "TorchUCC: failed to init XCCL lib\n");
        goto free_comm;
    }
  
    xccl_context_config_t *ctx_config;
    st = xccl_context_config_read(xccl_comm->xccl_lib, "TORCH", NULL, &ctx_config);
    if (st != XCCL_OK) {
        fprintf(stderr, "TorchUCC: failed to read XCCL context config\n");
        goto free_lib;
    }
    
    xccl_context_params_t ctx_params;
  
    ctx_params.field_mask       = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                                  XCCL_CONTEXT_PARAM_FIELD_OOB |
                                  XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE |
                                  XCCL_CONTEXT_PARAM_FIELD_TLS;
  
    ctx_params.thread_mode      = XCCL_THREAD_MODE_MULTIPLE;
  
    ctx_params.completion_type  = XCCL_TEAM_COMPLETION_TYPE_BLOCKING;
  
    ctx_params.tls              = XCCL_TL_UCX;
  
    ctx_params.oob.allgather    = oob_allgather;
    ctx_params.oob.req_test     = oob_allgather_test;
    ctx_params.oob.req_free     = oob_allgather_free;
    ctx_params.oob.coll_context = static_cast<void*>(p2p_comm);
    ctx_params.oob.rank         = p2p_comm->rank;
    ctx_params.oob.size         = p2p_comm->size;
  
    st = xccl_context_create(xccl_comm->xccl_lib, &ctx_params, ctx_config,
                             &xccl_comm->xccl_ctx);
    xccl_context_config_release(ctx_config);
    if (st != XCCL_OK) {
        fprintf(stderr, "TorchUCC: failed to create XCCL context\n");
        goto free_lib;
    }
  
    xccl_team_params_t team_params;
  
    team_params.field_mask           = XCCL_TEAM_PARAM_FIELD_EP_RANGE |
                                       XCCL_TEAM_PARAM_FIELD_OOB;
  
    team_params.range.type           = XCCL_EP_RANGE_STRIDED;
    team_params.range.strided.start  = 0;
    team_params.range.strided.stride = 1;
    team_params.oob.allgather        = oob_allgather;
    team_params.oob.req_test         = oob_allgather_test;
    team_params.oob.req_free         = oob_allgather_free;
    team_params.oob.coll_context     = static_cast<void*>(p2p_comm);
    team_params.oob.rank             = p2p_comm->rank;
    team_params.oob.size             = p2p_comm->size;
  
    st = xccl_team_create_post(xccl_comm->xccl_ctx, &team_params,
                               &xccl_comm->xccl_team);
    if (st != XCCL_OK) {
        fprintf(stderr, "TorchUCC: failed to create XCCL team\n");
        goto free_context;
    }
    while (XCCL_INPROGRESS == xccl_team_create_test(xccl_comm->xccl_team));
    *comm = xccl_comm;

    return TORCH_UCX_OK;
free_context:
    xccl_context_destroy(xccl_comm->xccl_ctx);
free_lib:
    xccl_lib_cleanup(xccl_comm->xccl_lib);
free_comm:
    delete xccl_comm;
    *comm = NULL;
    return TORCH_UCX_ERROR;
}

void torch_xccl_comm_close(torch_xccl_comm_t *comm)
{
    xccl_team_destroy(comm->xccl_team);
    xccl_context_destroy(comm->xccl_ctx);
    xccl_lib_cleanup(comm->xccl_lib);
}

}
