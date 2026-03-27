#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 256;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    float* __restrict__ output,
    int64_t n_rows,
    int64_t n_cols,
    int64_t group_size,
    bool has_zero_points
) {
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= n_rows || col >= n_cols) return;
    
    int64_t packed_col = col / 2;
    int64_t elem_idx = col % 2;
    int64_t packed_idx = row * ((n_cols + 1) / 2) + packed_col;
    
    uint8_t packed_val = packed_w[packed_idx];
    int8_t val = elem_idx == 0 ? (int8_t)(packed_val & 0x0F) : (int8_t)((packed_val >> 4) & 0x0F);
    val = (val >= 8) ? (val - 16) : val;
    
    int64_t group_idx = col / group_size;
    float scale = scales[group_idx];
    float zp = has_zero_points ? zero_points[group_idx] : 0.0f;
    
    output[row * n_cols + col] = (float(val) - zp) * scale;
}

__global__ void pack_int4_kernel(
    const int8_t* __restrict__ w_quant,
    uint8_t* __restrict__ packed,
    int64_t n_rows,
    int64_t n_cols
) {
    int64_t packed_col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= n_rows || packed_col >= (n_cols + 1) / 2) return;
    
    int64_t col0 = packed_col * 2;
    int64_t col1 = col0 + 1;
    
    int8_t v0 = col0 < n_cols ? w_quant[row * n_cols + col0] : 0;
    v0 = max(-8, min(7, v0));
    
    uint8_t packed_val = (uint8_t)(v0 & 0x0F);
    
    if (col1 < n_cols) {
        int8_t v1 = w_quant[row * n_cols + col1];
        v1 = max(-8, min(7, v1));
        packed_val |= ((uint8_t)(v1 & 0x0F)) << 4;
    }
    
    packed[row * ((n_cols + 1) / 2) + packed_col] = packed_val;
}

__global__ void int4_gemm_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t n_rows,
    int64_t k,
    int64_t group_size,
    bool transposed
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t seq = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= n_rows || seq >= seq_len) return;
    
    float acc = 0.0f;
    
    for (int64_t g = 0; g < (k + group_size - 1) / group_size; g++) {
        int64_t g_start = g * group_size;
        int64_t g_end = min(g_start + group_size, k);
        int64_t g_size = g_end - g_start;
        
        float scale = scales[g];
        float zp = zero_points != nullptr ? zero_points[g] : 0.0f;
        
        int64_t n_cols_group = transposed ? batch_size : seq_len;
        int64_t n_packed_cols = (g_end + 1) / 2;
        
        for (int64_t col = 0; col < g_size; col++) {
            int64_t actual_col = g_start + col;
            
            int64_t packed_col = col / 2;
            int64_t elem_idx = col % 2;
            int64_t packed_idx = row * ((k + 1) / 2) + g * (group_size / 2) + packed_col;
            
            uint8_t packed_val = packed_w[packed_idx];
            int8_t val = elem_idx == 0 ? (int8_t)(packed_val & 0x0F) : (int8_t)((packed_val >> 4) & 0x0F);
            val = (val >= 8) ? (val - 16) : val;
            
            float w_val = ((float)val - zp) * scale;
            
            if (transposed) {
                acc += input[actual_col * batch_size + seq] * w_val;
            } else {
                acc += input[seq * k + actual_col] * w_val;
            }
        }
    }
    
    output[seq * n_rows + row] = acc;
}

__global__ void int4_gemm_bias_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t n_rows,
    int64_t k,
    int64_t group_size,
    bool transposed
) {
    int4_gemm_kernel<<<gridDim, blockDim>>>(
        input, packed_w, scales, zero_points, output,
        batch_size, seq_len, n_rows, k, group_size, transposed
    );
    
    if (blockIdx.y == 0 && threadIdx.x < n_rows) {
        int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < n_rows && blockIdx.y == 0) {
            for (int64_t s = 0; s < seq_len; s++) {
                output[s * n_rows + row] += bias[row];
            }
        }
    }
}

__global__ void awq_scale_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ activations,
    float* __restrict__ scales,
    int64_t w_rows,
    int64_t w_cols,
    int64_t a_batch,
    int64_t a_cols,
    int64_t group_size
) {
    int64_t group_idx = blockIdx.x;
    int64_t tid = threadIdx.x;
    
    int64_t g_start = group_idx * group_size;
    int64_t g_end = min(g_start + group_size, w_cols);
    
    __shared__ float act_var[MAX_THREADS_PER_BLOCK];
    __shared__ float w_var[MAX_THREADS_PER_BLOCK];
    
    float act_sum = 0.0f;
    float w_sum = 0.0f;
    int64_t n_act = 0;
    int64_t n_w = 0;
    
    for (int64_t i = tid; i < a_cols * a_batch; i += blockDim.x) {
        int64_t col = i % a_cols;
        if (col >= g_start && col < g_end) {
            act_sum += activations[i] * activations[i];
            n_act++;
        }
    }
    act_var[tid] = warp_reduce_sum_f(act_sum);
    
    for (int64_t i = tid; i < w_rows * (g_end - g_start); i += blockDim.x) {
        int64_t row = i / (g_end - g_start);
        int64_t col = i % (g_end - g_start);
        w_sum += weights[row * w_cols + g_start + col] * weights[row * w_cols + g_start + col];
        n_w++;
    }
    w_var[tid] = warp_reduce_sum_f(w_sum);
    
    __syncthreads();
    
    if (tid == 0) {
        float total_act_var = 0.0f;
        float total_w_var = 0.0f;
        
        for (int i = 0; i < blockDim.x; i++) {
            total_act_var += act_var[i];
            total_w_var += w_var[i];
        }
        
        if (n_act > 0 && n_w > 0) {
            float act_std = sqrtf(total_act_var / (n_act * a_batch));
            float w_std = sqrtf(total_w_var / n_w);
            
            scales[group_idx] = w_std / (act_std + 1e-8f);
        } else {
            scales[group_idx] = 1.0f;
        }
    }
}

__global__ void int4_gemm_fast_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t n_rows,
    int64_t k,
    int64_t group_size
) {
    int64_t row = blockIdx.x;
    int64_t seq = blockIdx.y * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    float* s_w = sdata;
    float* s_in = &sdata[group_size];
    
    float acc = 0.0f;
    
    for (int64_t g = 0; g < (k + group_size - 1) / group_size; g++) {
        int64_t g_start = g * group_size;
        int64_t g_end = min(g_start + group_size, k);
        
        float scale = scales[g];
        float zp = zero_points != nullptr ? zero_points[g] : 0.0f;
        
        for (int64_t j = threadIdx.x; j < (g_end - g_start) * n_rows; j += blockDim.x) {
            int64_t local_col = j / n_rows;
            int64_t w_row = j % n_rows;
            
            int64_t actual_col = g_start + local_col;
            int64_t packed_col = local_col / 2;
            int64_t elem_idx = local_col % 2;
            int64_t packed_idx = w_row * ((k + 1) / 2) + g * (group_size / 2) + packed_col;
            
            uint8_t packed_val = packed_w[packed_idx];
            int8_t val = elem_idx == 0 ? (int8_t)(packed_val & 0x0F) : (int8_t)((packed_val >> 4) & 0x0F);
            val = (val >= 8) ? (val - 16) : val;
            
            s_w[j % blockDim.x] = ((float)val - zp) * scale;
        }
        
        for (int64_t j = threadIdx.x; j < (g_end - g_start); j += blockDim.x) {
            s_in[j % blockDim.x] = input[seq * k + g_start + j];
        }
        
        __syncthreads();
        
        for (int64_t j = 0; j < (g_end - g_start); j++) {
            if (threadIdx.x < n_rows) {
                acc += s_in[j] * s_w[j * n_rows + threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    if (threadIdx.x < n_rows && seq < seq_len) {
        output[seq * n_rows + row] = acc;
    }
}
