/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <torch_ucc_status.hpp>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <ATen/ATen.h>
#include <torch_ucc_status.hpp>
#include "torch_ucc_sendrecv.hpp"
#include <torch_ucc_ops.hpp>
namespace c10d {

struct torch_ucx_coll_request_t;
typedef torch_ucc_status_t (*torch_ucx_progress_p)(torch_ucx_coll_request_t *request);

struct torch_ucx_coll_config_t {
    int  chunk;
    bool reverse;
    int  max_polls;
};

struct torch_ucx_coll_comm_t {
    torch_ucx_comm_t        *p2p_comm;
    torch_ucx_coll_config_t config;
    uint32_t                last_tag;
#ifdef USE_CUDA
    cudaStream_t            stream;
#endif 
};

struct torch_ucx_coll_request_t {
    torch_ucc_coll_request_t super;
    torch_ucx_coll_comm_t    *comm;
    c10::DeviceIndex         dev_index;
    c10::DeviceType          dev_type;
    uint32_t                 tag;
    torch_ucx_progress_p     progress;
    torch_ucc_status_t       status;
    torch_ucx_memtype_t      src_buf_mtype;
    void                     *src_buffer;
    torch_ucx_memtype_t      dst_buf_mtype;
    void                     *dst_buffer;
    size_t                   len;
    std::vector<int>         send_lengths;
    std::vector<int>         send_offsets;
    std::vector<int>         recv_lengths;
    std::vector<int>         recv_offsets;
    torch_ucx_request_t      **reqs;
    int                      n_sreqs;
    int                      n_rreqs;
};

torch_ucc_status_t torch_ucx_coll_comm_init(torch_ucx_comm_t *p2p_comm,
                                            void **comm);

torch_ucc_status_t torch_ucx_coll_test(torch_ucc_coll_request_t *request);

torch_ucc_status_t torch_ucx_alltoall(void *coll_comm,
                                      void *send_buffer, torch_ucx_memtype_t send_mtype,
                                      void *recv_buffer, torch_ucx_memtype_t recv_mtype,
                                      size_t len, torch_ucc_coll_request_t **request);

torch_ucx_status_t torch_ucx_alltoall_start(torch_ucx_coll_comm_t *comm,
                                            torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoallv_start(torch_ucx_coll_comm_t *comm,
                                             torch_ucx_coll_request_t *request);

torch_ucc_status_t torch_ucx_coll_comm_close(void *comm);

}
