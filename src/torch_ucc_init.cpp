/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

#include <torch/python.h>
#include <pybind11/chrono.h>

#include "torch_ucc.hpp"

static void __attribute__((constructor)) ProcessGroupUCCConstructor() {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend =
      module.attr("Backend").attr("register_backend");
  register_backend("ucc", py::cpp_function(c10d::ProcessGroupUCC::createProcessGroupUCC));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &c10d::ProcessGroupUCC::createProcessGroupUCC);
}
