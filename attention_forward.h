// attention_forward.h
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// 声明 attention_forward 内核的包装函数
void attention_forward(
    const float* q, const float* k, const float* v,
    float* out, int num_q, int num_kv, int head_dim
);

// 声明 blockwise 内核的包装函数
void attention_blockwise_forward(
    const float* q, const float** k_ptrs, const float** v_ptrs,
    int num_blocks, int tokens_per_block,
    float* out, int head_dim
);

#ifdef __cplusplus
}
#endif