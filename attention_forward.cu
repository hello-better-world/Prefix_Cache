#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "attention_forward.h"

__global__ void attention_forward_kernel(
    const float* q,       // [num_q × head_dim]
    const float* k,       // [num_kv × head_dim]
    const float* v,       // [num_kv × head_dim]
    float* output,        // [num_q × head_dim]
    int num_q,
    int num_kv,
    int head_dim
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_q) return;

    extern __shared__ float shared_mem[];
    float* logits = shared_mem;             // [num_kv]
    float* softmax_vals = shared_mem + num_kv;

    const float* q_vec = q + qid * head_dim;

    // Compute dot(q, k_j)
    float max_logit = -1e9f;
    for (int j = 0; j < num_kv; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k[j * head_dim + d];
        }
        logits[j] = score;
        if (score > max_logit) max_logit = score;
    }

    // Softmax
    float denom = 0.0f;
    for (int j = 0; j < num_kv; ++j) {
        softmax_vals[j] = expf(logits[j] - max_logit);
        denom += softmax_vals[j];
    }
    for (int j = 0; j < num_kv; ++j) {
        softmax_vals[j] /= denom;
    }

    // Weighted sum: output[qid] = softmax * V
    float* out_vec = output + qid * head_dim;
    for (int d = 0; d < head_dim; ++d) out_vec[d] = 0.0f;
    for (int j = 0; j < num_kv; ++j) {
        float weight = softmax_vals[j];
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] += weight * v[j * head_dim + d];
        }
    }
}

__global__ void attention_blockwise_forward_kernel(
    const float* q,                    // [num_q x head_dim]
    const float** k_blocks,            // 每个 block 是 tokens_per_block x head_dim
    const float** v_blocks,
    int num_blocks,
    int tokens_per_block,
    float* output,                     // [num_q x head_dim]
    int head_dim) {

    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= gridDim.x * blockDim.x) return;

    const float* q_vec = q + qid * head_dim;
    float* out_vec = output + qid * head_dim;

    extern __shared__ float smem[]; // 分配两个部分
    float* logits = smem;           // [num_blocks * tokens_per_block]
    float* scores = smem + num_blocks * tokens_per_block;

    // Step 1: compute q·k for all KV tokens
    int kv_idx = 0;
    for (int b = 0; b < num_blocks; ++b) {
        const float* k_base = k_blocks[b];
        for (int t = 0; t < tokens_per_block; ++t, ++kv_idx) {
            const float* k_vec = k_base + t * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_vec[d] * k_vec[d];
            }
            logits[kv_idx] = score;
        }
    }

    // Step 2: softmax over all logits
    int total_kv = num_blocks * tokens_per_block;
    float max_logit = -1e9f;
    for (int i = 0; i < total_kv; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float denom = 0.0f;
    for (int i = 0; i < total_kv; ++i) {
        scores[i] = expf(logits[i] - max_logit);
        denom += scores[i];
    }
    for (int i = 0; i < total_kv; ++i) {
        scores[i] /= denom;
    }

    // Step 3: weighted sum over V
    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    kv_idx = 0;
    for (int b = 0; b < num_blocks; ++b) {
        const float* v_base = v_blocks[b];
        for (int t = 0; t < tokens_per_block; ++t, ++kv_idx) {
            const float* v_vec = v_base + t * head_dim;
            float weight = scores[kv_idx];
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += weight * v_vec[d];
            }
        }
    }
}

// ================== 包装函数 ==================
void attention_forward(
    const float* q, const float* k, const float* v,
    float* out, int num_q, int num_kv, int head_dim
) {
    dim3 block(256);  // 建议线程数: 256
    dim3 grid((num_q + block.x - 1) / block.x);
    
    // 共享内存大小 = num_kv * sizeof(float)
    size_t shared_bytes = num_kv * sizeof(float);
    
    attention_forward_kernel<<<grid, block, shared_bytes>>>(
        q, k, v, out, num_q, num_kv, head_dim
    );
    cudaDeviceSynchronize();
}

void attention_blockwise_forward(
    const float* q, const float** k_ptrs, const float** v_ptrs,
    int num_blocks, int tokens_per_block,
    float* out, int head_dim
) {
    dim3 block(256);  // 建议线程数: 256
    dim3 grid((num_blocks * tokens_per_block + block.x - 1) / block.x);
    
    // 共享内存大小 = (num_blocks * tokens_per_block) * sizeof(float)
    size_t shared_bytes = 2 * num_blocks * tokens_per_block * sizeof(float);
    
    attention_blockwise_forward_kernel<<<grid, block, shared_bytes>>>(
        q, k_ptrs, v_ptrs, num_blocks, tokens_per_block, out, head_dim
    );
    cudaDeviceSynchronize();
}
