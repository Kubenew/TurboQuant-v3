#ifndef TURBOQUANT_CUDA_KERNELS_H
#define TURBOQUANT_CUDA_KERNELS_H

#include <torch/extension.h>

torch::Tensor int4_dequantize_cuda(
    torch::Tensor packed_w,
    torch::Tensor scales,
    torch::Tensor zero_points,
    int64_t group_size,
    int64_t n_rows,
    int64_t n_cols
);

torch::Tensor int4_pack_cuda(
    torch::Tensor w_quant,
    int64_t n_rows,
    int64_t n_cols
);

torch::Tensor int4_gemm_cuda(
    torch::Tensor input,
    torch::Tensor packed_w,
    torch::Tensor scales,
    torch::Tensor zero_points,
    torch::Tensor bias,
    int64_t group_size,
    bool transposed
);

torch::Tensor activation_aware_scale_cuda(
    torch::Tensor weights,
    torch::Tensor activations,
    int64_t group_size
);

#endif // TURBOQUANT_CUDA_KERNELS_H
