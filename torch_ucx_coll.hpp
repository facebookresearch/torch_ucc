#pragma once

#include "torch_ucc_sendrecv.hpp"

namespace c10d {

struct torch_ucx_coll_request_t;
typedef torch_ucx_status_t (*torch_ucx_progress_p)(torch_ucx_coll_request_t *request);

struct torch_ucx_coll_config_t {
    int  chunk;
    bool reverse;
    int  max_polls;
};

struct torch_ucx_coll_comm_t {
    torch_ucx_comm_t        *p2p_comm;
    torch_ucx_coll_config_t config;
    uint32_t                last_tag;
};

struct torch_ucx_coll_request_t {
    torch_ucx_coll_comm_t   *comm;
    uint32_t                tag;
    torch_ucx_status_t      status;
    torch_ucx_progress_p    progress;
    void                    *src_buffer;
    void                    *dst_buffer;
    size_t                  len;
    torch_ucx_request_t     **reqs;
    int                     n_sreqs;
    int                     n_rreqs;
};

torch_ucx_status_t torch_ucx_coll_comm_init(torch_ucx_comm_t *p2p_comm,
                                            torch_ucx_coll_comm_t **comm);

torch_ucx_status_t torch_ucx_coll_test(torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoall_start(torch_ucx_coll_comm_t *comm,
                                            torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoall_progress(torch_ucx_coll_request_t *request);

void torch_ucx_coll_comm_close(torch_ucx_coll_comm_t *comm);

}
