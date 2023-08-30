/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

#include <torch/python.h>
#include "torch_ucc.hpp"

using namespace c10d;

extern "C" C10_EXPORT c10::intrusive_ptr<ProcessGroup> createProcessGroupUCC(
  const c10::intrusive_ptr<Store>& store,
  int rank, int size
) {
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
