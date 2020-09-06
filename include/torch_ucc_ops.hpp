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
    c10::DeviceIndex         dev_index;
    c10::DeviceType          dev_type;
    std::vector<at::Tensor>  src;
    std::vector<at::Tensor>  dst;
};

struct torch_ucc_coll_ops_t {
    torch_ucc_status_t (*coll_comm_init) (torch_ucx_comm_t *p2p_comm,
                                          void **coll_comm);

    torch_ucc_status_t (*allgather)(void *coll_comm,
                                    at::Tensor &input_tensor,
                                    std::vector<at::Tensor>& output_tensors,
                                    torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*alltoall)(void *coll_comm,
                                   at::Tensor &input_tensor,
                                   at::Tensor &output_tensor,
                                   torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*alltoallv)(void *coll_comm,
                                    at::Tensor &input_tensor,
                                    uint32_t *send_lengths, uint32_t *send_offsets,
                                    at::Tensor &output_tensor,
                                    uint32_t *recv_lengths, uint32_t *recv_offsets,
                                    torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*allreduce)(void *coll_comm, at::Tensor &tensor,
                                    const AllreduceOptions& opts,
                                    torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*barrier)(void *coll_comm, torch_ucc_coll_request_t **request);

    torch_ucc_status_t (*coll_progress)  (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_test)      (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_finalize)  (torch_ucc_coll_request_t *request);

    torch_ucc_status_t (*coll_comm_close)(void *coll_comm);
};

extern torch_ucc_coll_ops_t ucx_coll_ops;

#ifdef WITH_XCCL
extern torch_ucc_coll_ops_t xccl_coll_ops;
#endif

inline void torch_ucc_coll_request_init(torch_ucc_coll_request_t *request,
                                        std::vector<at::Tensor> *srcPtr,
                                        std::vector<at::Tensor> *dstPtr)
{
    if (srcPtr) {
        request->src = *srcPtr;
    }
    if (dstPtr) {
        request->dst = *dstPtr;
    }
}

inline torch_ucc_status_t torch_ucc_coll_ops_init(torch_ucc_coll_ops_t *coll_ops)
{
    char *env;

    env = std::getenv("TORCH_UCC_COLL_BACKEND");
    if ((env != NULL) && (!strcasecmp(env, "xccl"))) {
#ifdef WITH_XCCL
        *coll_ops = xccl_coll_ops;
#else
        fprintf(stderr, "ProcessGroupUCC: plugin wasn't compiled with XCCL support\n");
        return TORCH_UCC_ERROR;
#endif
    } else {
        *coll_ops = ucx_coll_ops;
    }

    return TORCH_UCC_OK;
}

};
