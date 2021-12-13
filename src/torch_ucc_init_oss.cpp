#include <torch/python.h>
#include "torch_ucc.hpp"

// This function is intentionally designed to take a void * argument
// and return a void * argument. This design is to make the mangled
// symbol simple.
C10_EXPORT void *createProcessGroupUCCForNCCL(void *args) {
  using namespace c10d;
  struct args_t {
    const c10::intrusive_ptr<Store>& store;
    int rank = -1;
    int size = -1;
  };
  args_t *args_ = static_cast<args_t *>(args);
  c10d::ProcessGroupUCC *pg = new ProcessGroupUCC(args_->store, args_->rank, args_->size);
  return static_cast<void *>(pg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
