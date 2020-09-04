/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <c10d/Types.hpp>
#include <c10d/ProcessGroup.hpp>
#include <torch_ucc_status.hpp>
#include <torch_ucc_sendrecv.hpp>

namespace c10d {

enum torch_ucx_memtype_t {
    TORCH_UCX_HOST,
    TORCH_UCX_CUDA
};

struct torch_ucc_coll_request_t {
    c10::DeviceIndex dev_index;
    c10::DeviceType  dev_type;
};

struct torch_ucc_coll_ops_t {
    torch_ucc_status_t (*coll_comm_init) (torch_ucx_comm_t *p2p_comm,
                                          void **coll_comm);

    torch_ucc_status_t (*alltoall)(void *coll_comm,
                                   void *send_buffer, torch_ucx_memtype_t send_mtype,
                                   void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                   size_t len, torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*alltoallv)(void *coll_comm,
                                    void *send_buffer, torch_ucx_memtype_t send_mtype,
                                    at::ScalarType send_data_type,
                                    uint32_t *send_lengths, uint32_t *send_offsets,
                                    void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                    at::ScalarType recv_data_type,
                                    uint32_t *recv_lengths, uint32_t *recv_offsets,
                                    torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*allreduce)(void *coll_comm,
                                    void *send_buffer, torch_ucx_memtype_t send_mtype,
                                    void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                    int count, int element_size, at::ScalarType data_type,
                                    ReduceOp op, torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*coll_progress)  (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_test)      (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_finalize)  (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_comm_close)(void *coll_comm);
};

extern torch_ucc_coll_ops_t ucx_coll_ops;
extern torch_ucc_coll_ops_t xccl_coll_ops;

inline torch_ucc_status_t torch_ucc_coll_ops_init(torch_ucc_coll_ops_t *coll_ops)
{
    char *env;

    env = std::getenv("TORCH_UCC_COLL_BACKEND");
    if ((env != NULL) && (!strcasecmp(env, "xccl"))) {
        *coll_ops = xccl_coll_ops;
    } else {
        *coll_ops = ucx_coll_ops;
    }

    return TORCH_UCC_OK;
}

};
