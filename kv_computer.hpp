#pragma once

#include <vector>
#include <random>
#include "kv_cache_block.hpp"

class KVComputer {
public:
    KVComputer(int embed_dim, int head_dim, int tokens_per_block);

    // 构建 KV：embedding → linear → write to block
    void compute_and_fill(KVCacheBlock* block, const std::vector<int>& tokens);

    std::vector<std::vector<float>> q_weight_;
    std::vector<std::vector<float>> k_weight_;
    std::vector<std::vector<float>> v_weight_;

    std::vector<float> embed_token(int token_id);

    std::vector<float> matmul(const std::vector<float>& vec,
                              const std::vector<std::vector<float>>& weight);

private:
    int embed_dim_;
    int head_dim_;
    int tokens_per_block_;
};


