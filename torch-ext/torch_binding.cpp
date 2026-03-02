#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("rms_norm", torch::kMPS, rms_norm);
#endif

  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("fused_add_rms_norm", torch::kMPS, fused_add_rms_norm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
