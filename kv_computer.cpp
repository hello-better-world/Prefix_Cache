#include "kv_computer.hpp"
#include <cassert>
#include <cstring>

KVComputer::KVComputer(int embed_dim, int head_dim, int tokens_per_block)
    : embed_dim_(embed_dim), head_dim_(head_dim), tokens_per_block_(tokens_per_block) {

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto init_matrix = [&](std::vector<std::vector<float>>& mat) {
        mat.resize(embed_dim_);
        for (int i = 0; i < embed_dim_; ++i) {
            mat[i].resize(head_dim_);
            for (int j = 0; j < head_dim_; ++j) {
                mat[i][j] = dist(gen);
            }
        }
    };

    init_matrix(q_weight_);
    init_matrix(k_weight_);
    init_matrix(v_weight_);
}

std::vector<float> KVComputer::embed_token(int token_id) {
    std::vector<float> embedding(embed_dim_);
    std::mt19937 gen(token_id);  // 固定种子，确保相同 token 得到一致 embedding
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < embed_dim_; ++i) {
        embedding[i] = dist(gen);
    }
    return embedding;
}

std::vector<float> KVComputer::matmul(const std::vector<float>& vec,
                                      const std::vector<std::vector<float>>& weight) {
    std::vector<float> out(head_dim_, 0.0f);
    for (int j = 0; j < head_dim_; ++j) {
        for (int i = 0; i < embed_dim_; ++i) {
            out[j] += vec[i] * weight[i][j];
        }
    }
    return out;
}

// void KVComputer::compute_and_fill(KVCacheBlock* block, const std::vector<int>& tokens) {
//     assert(tokens.size() <= (size_t)tokens_per_block_);

//     float* k_dst = reinterpret_cast<float*>(block->k_ptr);
//     float* v_dst = reinterpret_cast<float*>(block->v_ptr);
//     std::cout << "aaaaa" << std::endl;
//     for (size_t i = 0; i < tokens.size(); ++i) {
//         std::cout << "aaaaa" << std::endl;
//         auto emb = embed_token(tokens[i]);
//         std::cout << "aaaaa" << std::endl;
//         auto k_vec = matmul(emb, k_weight_);
//         auto v_vec = matmul(emb, v_weight_);
//         std::cout << "bbbbb" << std::endl;

//         // 写入 block 中：按 [token_id][head_dim] 存
//         std::memcpy(k_dst + i * head_dim_, k_vec.data(), sizeof(float) * head_dim_);
//         std::memcpy(v_dst + i * head_dim_, v_vec.data(), sizeof(float) * head_dim_);
//     }
// }

void KVComputer::compute_and_fill(KVCacheBlock* block, const std::vector<int>& tokens) {
    assert(tokens.size() <= (size_t)tokens_per_block_);

    for (size_t i = 0; i < tokens.size(); ++i) {
        auto emb = embed_token(tokens[i]);
        auto k_vec = matmul(emb, k_weight_);
        auto v_vec = matmul(emb, v_weight_);

        // 将 host 上生成的 k 向量拷贝到显存 block 对应位置
        cudaMemcpy(
            static_cast<char*>(block->k_ptr) + i * head_dim_ * sizeof(float),
            k_vec.data(),
            sizeof(float) * head_dim_,
            cudaMemcpyHostToDevice);

        // 同理写入 v 向量
        cudaMemcpy(
            static_cast<char*>(block->v_ptr) + i * head_dim_ * sizeof(float),
            v_vec.data(),
            sizeof(float) * head_dim_,
            cudaMemcpyHostToDevice);
    }
}
