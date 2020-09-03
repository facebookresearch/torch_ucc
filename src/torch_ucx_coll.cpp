/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include <cstdlib>
#include <torch_ucx_coll.hpp>
#include <torch_ucc_ops.hpp>

namespace c10d {

static void torch_ucx_get_coll_config(torch_ucx_coll_config_t *config)
{
    char *env;

    config->chunk     = 1;
    config->reverse   = 0;
    config->max_polls = 10;
 
    env = std::getenv("TORCH_UCC_UCX_CHUNK");
    if (env) {
        config->chunk = std::atoi(env);
    }
    env = std::getenv("TORCH_UCC_UCX_REVERSE");
    if (env) {
        config->reverse = std::atoi(env);
    }
    env = std::getenv("TORCH_UCC_UCX_MAX_POLLS");
    if (env) {
        config->max_polls = std::atoi(env);
    }
}

torch_ucc_status_t torch_ucx_coll_comm_init(torch_ucx_comm_t *p2p_comm,
                                            void **comm)
{
    torch_ucx_coll_comm_t *coll_comm;

    coll_comm = new torch_ucx_coll_comm_t;
    torch_ucx_get_coll_config(&coll_comm->config);
    coll_comm->p2p_comm = p2p_comm;
    coll_comm->last_tag = 0;
#ifdef USE_CUDA
    coll_comm->stream   = 0;
#endif
    *comm = coll_comm;
    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_test(torch_ucc_coll_request_t *request)
{
    torch_ucx_coll_request_t *req = (torch_ucx_coll_request_t*)request;

    return req->status;
}

torch_ucc_status_t torch_ucx_coll_comm_close(void *comm)
{
    torch_ucx_coll_comm_t *coll_comm;

    coll_comm = (torch_ucx_coll_comm_t*)(comm);
#ifdef USE_CUDA
    if (coll_comm->stream != 0) {
        cudaStreamDestroy(coll_comm->stream);
    }
#endif
    delete coll_comm;

    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_progress(torch_ucc_coll_request_t *request)
{
    torch_ucx_coll_request_t *req = (torch_ucx_coll_request_t*)request;

    return req->progress(req);
}

torch_ucc_status_t torch_ucx_coll_free(torch_ucc_coll_request_t *request)
{
    torch_ucx_coll_request_t *req = (torch_ucx_coll_request_t*)request;

    delete req;
    return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_allreduce(void *coll_comm,
                                            void *send_buffer, torch_ucx_memtype_t send_mtype,
                                            void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                            int count, int element_size, at::ScalarType data_type,
                                            ReduceOp op, torch_ucc_coll_request_t **request)
{
    fprintf(stderr, "ProcessGroupUCC: UCX backend doesn't support allreduce\n");
    return TORCH_UCC_ERROR;
}



torch_ucc_coll_ops_t ucx_coll_ops {
    torch_ucx_coll_comm_init,
    torch_ucx_alltoall,
    torch_ucx_coll_allreduce,
    torch_ucx_coll_progress,
    torch_ucx_coll_test,
    torch_ucx_coll_free,
    torch_ucx_coll_comm_close
};

}
