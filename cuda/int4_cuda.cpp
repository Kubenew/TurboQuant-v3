#include "cuda_kernels.h"
#include "cuda_kernels.cuh"

torch::Tensor int4_dequantize_cuda(
    torch::Tensor packed_w,
    torch::Tensor scales,
    torch::Tensor zero_points,
    int64_t group_size,
    int64_t n_rows,
    int64_t n_cols
) {
    TORCH_CHECK(packed_w.is_cuda(), "packed_w must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(packed_w.dtype() == torch::kUInt8, "packed_w must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16 || scales.dtype() == torch::kFloat32, 
                "scales must be float16 or float32");
    
    auto output = torch::empty({n_rows, n_cols}, packed_w.options().dtype(torch::kFloat32));
    
    const dim3 blocks((n_cols + 63) / 64, (n_rows + 15) / 16);
    const dim3 threads(64, 16);
    
    dequantize_int4_kernel<<<blocks, threads>>>(
        packed_w.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zero_points.defined() ? zero_points.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        n_rows, n_cols, group_size,
        zero_points.defined()
    );
    
    return output;
}

torch::Tensor int4_pack_cuda(
    torch::Tensor w_quant,
    int64_t n_rows,
    int64_t n_cols
) {
    TORCH_CHECK(w_quant.is_cuda(), "w_quant must be a CUDA tensor");
    TORCH_CHECK(w_quant.dtype() == torch::kInt8 || w_quant.dtype() == torch::kUInt8,
                "w_quant must be int8 or uint8");
    
    int64_t packed_cols = (n_cols + 1) / 2;
    auto packed = torch::empty({n_rows, packed_cols}, w_quant.options());
    
    const dim3 blocks((packed_cols + 63) / 64, (n_rows + 15) / 16);
    const dim3 threads(64, 16);
    
    pack_int4_kernel<<<blocks, threads>>>(
        w_quant.data_ptr<int8_t>(),
        packed.data_ptr<uint8_t>(),
        n_rows, n_cols
    );
    
    return packed;
}

torch::Tensor int4_gemm_cuda(
    torch::Tensor input,
    torch::Tensor packed_w,
    torch::Tensor scales,
    torch::Tensor zero_points,
    torch::Tensor bias,
    int64_t group_size,
    bool transposed
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(packed_w.is_cuda(), "packed_w must be a CUDA tensor");
    
    int64_t n_rows = packed_w.size(0);
    int64_t n_cols = packed_w.size(1) * 2;
    int64_t batch_size = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t k = input.size(2);
    
    auto output = torch::empty({batch_size, seq_len, n_rows}, input.options());
    
    const dim3 blocks((n_rows + 255) / 256, (seq_len + 63) / 64);
    const dim3 threads(256, 1, 1);
    
    if (bias.defined()) {
        int4_gemm_bias_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            packed_w.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            zero_points.defined() ? zero_points.data_ptr<float>() : nullptr,
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, seq_len, n_rows, k, group_size, transposed
        );
    } else {
        int4_gemm_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            packed_w.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            zero_points.defined() ? zero_points.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, seq_len, n_rows, k, group_size, transposed
        );
    }
    
    return output;
}

torch::Tensor activation_aware_scale_cuda(
    torch::Tensor weights,
    torch::Tensor activations,
    int64_t group_size
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
    
    int64_t n_groups = (weights.size(1) + group_size - 1) / group_size;
    auto scales = torch::empty({n_groups}, weights.options().dtype(torch::kFloat32));
    
    const dim3 blocks(n_groups);
    const dim3 threads(256);
    
    awq_scale_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        activations.data_ptr<float>(),
        scales.data_ptr<float>(),
        weights.size(0), weights.size(1),
        activations.size(0), activations.size(1),
        group_size
    );
    
    return scales;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_dequantize", &int4_dequantize_cuda, "INT4 dequantization (CUDA)",
          py::arg("packed_w"),
          py::arg("scales"),
          py::arg("zero_points"),
          py::arg("group_size"),
          py::arg("n_rows"),
          py::arg("n_cols"));
    
    m.def("int4_pack", &int4_pack_cuda, "INT4 packing (CUDA)",
          py::arg("w_quant"),
          py::arg("n_rows"),
          py::arg("n_cols"));
    
    m.def("int4_gemm", &int4_gemm_cuda, "INT4 GEMM (CUDA)",
          py::arg("input"),
          py::arg("packed_w"),
          py::arg("scales"),
          py::arg("zero_points"),
          py::arg("bias"),
          py::arg("group_size"),
          py::arg("transposed"));
    
    m.def("awq_scale", &activation_aware_scale_cuda, "AWQ scale computation (CUDA)",
          py::arg("weights"),
          py::arg("activations"),
          py::arg("group_size"));
}
