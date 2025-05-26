#pragma once
#include <vector>
#include "kv_cache_block.hpp"
#include "prefix_cache_manager.hpp"

struct AttentionInput {
    std::vector<std::vector<float>> query;  // Q: num_q × head_dim

    std::vector<KVCacheBlock*> cached_blocks;
    int tokens_per_block;
    int head_dim;

    std::vector<std::vector<float>> k_new;  // 新计算的 K: num_k_new × head_dim
    std::vector<std::vector<float>> v_new;  // 新计算的 V: num_k_new × head_dim
};

class AttentionExecutor {
public:
    AttentionExecutor(int head_dim);
    std::vector<std::vector<float>> run(const AttentionInput& input);  // 输出每个 Q 的 attention 结果
    std::vector<std::vector<float>> run_gpu(const float* d_q, int num_q, const float* d_k, const float* d_v, int num_kv);
    std::vector<std::vector<float>> run_gpu_blockwise(const float* d_q, int num_q, const std::vector<KVCacheBlock*>& un_cached_blocks, 
        const std::vector<std::optional<KVLocation>>& cached_blocks, int tokens_per_block, int cur_device_id);

private:
    int head_dim_;
    float dot(const std::vector<float>& a, const float* b);
    std::vector<float> softmax(const std::vector<float>& logits);
};
