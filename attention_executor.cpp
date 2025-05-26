#include "attention_executor.hpp"
#include "attention_forward.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

AttentionExecutor::AttentionExecutor(int head_dim)
    : head_dim_(head_dim) {}

float AttentionExecutor::dot(const std::vector<float>& a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim_; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<float> AttentionExecutor::softmax(const std::vector<float>& logits) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_vals(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_vals[i] = std::exp(logits[i] - max_logit);
        sum_exp += exp_vals[i];
    }
    for (float& val : exp_vals) {
        val /= sum_exp;
    }
    return exp_vals;
}

std::vector<std::vector<float>> AttentionExecutor::run(const AttentionInput& input) {
    std::vector<std::vector<float>> result;
    for (const auto& q : input.query) {
        std::vector<float> logits;

        std::vector<const float*> all_k_ptrs;
        std::vector<const float*> all_v_ptrs;

        // 加入缓存的 KV
        for (KVCacheBlock* block : input.cached_blocks) {
            const float* k_block = reinterpret_cast<const float*>(block->k_ptr);
            const float* v_block = reinterpret_cast<const float*>(block->v_ptr);
            for (int i = 0; i < input.tokens_per_block; ++i) {
                all_k_ptrs.push_back(k_block + i * input.head_dim);
                all_v_ptrs.push_back(v_block + i * input.head_dim);
                logits.push_back(dot(q, k_block + i * input.head_dim));
            }
        }

        // 加入新算的 KV
        for (size_t i = 0; i < input.k_new.size(); ++i) {
            logits.push_back(dot(q, input.k_new[i].data()));
            all_k_ptrs.push_back(input.k_new[i].data());
            all_v_ptrs.push_back(input.v_new[i].data());
        }

        // softmax → weighted sum
        std::vector<float> attn = softmax(logits);
        std::vector<float> output(head_dim_, 0.0f);
        for (size_t i = 0; i < attn.size(); ++i) {
            for (int j = 0; j < head_dim_; ++j) {
                output[j] += attn[i] * all_v_ptrs[i][j];
            }
        }

        result.push_back(output);
    }

    return result;
}


std::vector<std::vector<float>> AttentionExecutor::run_gpu(
    const float* d_q, int num_q,
    const float* d_k, const float* d_v, int num_kv) {

    float* d_out;
    cudaMalloc(&d_out, sizeof(float) * num_q * head_dim_);

    size_t shared_bytes = sizeof(float) * 2 * num_kv;
    dim3 block(64), grid((num_q + 63) / 64);
    attention_forward(d_q, d_k, d_v, d_out, num_q, num_kv, head_dim_);

    std::vector<std::vector<float>> result(num_q, std::vector<float>(head_dim_));
    std::vector<float> host_out(num_q * head_dim_);
    cudaMemcpy(host_out.data(), d_out, sizeof(float) * num_q * head_dim_, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    for (int i = 0; i < num_q; ++i) {
        for (int j = 0; j < head_dim_; ++j) {
            result[i][j] = host_out[i * head_dim_ + j];
        }
    }

    return result;
}

// std::vector<std::vector<float>> AttentionExecutor::run_gpu_blockwise(
//     const float* d_q, int num_q,
//     const std::vector<KVCacheBlock*>& un_cached_blocks,
//     const std::vector<KVLocation*>& cached_blocks,
//     int tokens_per_block, int cur_device_id) {

//     const int head_dim = head_dim_;
//     const int num_blocks = blocks.size();
//     const int total_kv = num_blocks * tokens_per_block;

//     // 1. 准备 k_ptr/v_ptr 指针数组
//     std::vector<const float*> h_k_ptrs, h_v_ptrs;
//     for (auto* blk : blocks) {
//         if (blk->device_id != cur_device_id && blk->ipc_k.has_value() && blk->ipc_v.has_value()) {
//             // 远程 block → 打开句柄（只打开一次，可缓存优化）
//             void* k_remote = nullptr;
//             void* v_remote = nullptr;
//             cudaIpcOpenMemHandle(&k_remote, blk->ipc_k.value(), cudaIpcMemLazyEnablePeerAccess);
//             cudaIpcOpenMemHandle(&v_remote, blk->ipc_v.value(), cudaIpcMemLazyEnablePeerAccess);
//             h_k_ptrs.push_back(reinterpret_cast<const float*>(k_remote));
//             h_v_ptrs.push_back(reinterpret_cast<const float*>(v_remote));
//         }
//         if (blk->device_id == cur_device_id){
//             h_k_ptrs.push_back(reinterpret_cast<const float*>(blk->k_ptr));
//             h_v_ptrs.push_back(reinterpret_cast<const float*>(blk->v_ptr));
//         }
//     }

//     const float** d_k_ptrs;
//     const float** d_v_ptrs;
//     cudaMalloc(&d_k_ptrs, sizeof(float*) * num_blocks);
//     cudaMalloc(&d_v_ptrs, sizeof(float*) * num_blocks);
//     cudaMemcpy(d_k_ptrs, h_k_ptrs.data(), sizeof(float*) * num_blocks, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_v_ptrs, h_v_ptrs.data(), sizeof(float*) * num_blocks, cudaMemcpyHostToDevice);

//     // 2. 分配输出 buffer
//     float* d_out;
//     cudaMalloc(&d_out, sizeof(float) * num_q * head_dim);

//     // 3. 启动 kernel
//     int threads = 64;
//     int blocks_q = (num_q + threads - 1) / threads;
//     size_t shared_mem = sizeof(float) * total_kv * 2;  // logits + scores

//     attention_blockwise_forward(d_q, d_k_ptrs, d_v_ptrs, num_blocks, tokens_per_block, d_out, head_dim);

//     // 4. 拷回 host
//     std::vector<std::vector<float>> result(num_q, std::vector<float>(head_dim));
//     std::vector<float> tmp(num_q * head_dim);
//     cudaMemcpy(tmp.data(), d_out, sizeof(float) * tmp.size(), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < num_q; ++i)
//         std::copy(tmp.begin() + i * head_dim, tmp.begin() + (i + 1) * head_dim, result[i].begin());

//     // 5. 清理
//     cudaFree(d_k_ptrs);
//     cudaFree(d_v_ptrs);
//     cudaFree(d_out);

//     return result;
// }


std::vector<std::vector<float>> AttentionExecutor::run_gpu_blockwise(
    const float* d_q, int num_q,
    const std::vector<KVCacheBlock*>& un_cached_blocks,
    const std::vector<std::optional<KVLocation>>& cached_blocks,
    int tokens_per_block, int cur_device_id) {

    const int head_dim = head_dim_;
    const int num_blocks = un_cached_blocks.size() + cached_blocks.size();
    const int total_kv = num_blocks * tokens_per_block;

    std::vector<const float*> h_k_ptrs, h_v_ptrs;

    // 1. 加入 cached_blocks
    for (const auto& loc : cached_blocks) {
        if (loc->device_id == cur_device_id) {
            // 本地 block
            h_k_ptrs.push_back(reinterpret_cast<const float*>(loc->local_k_ptr));
            h_v_ptrs.push_back(reinterpret_cast<const float*>(loc->local_v_ptr));
        } else {
            // 远程 block：使用 CUDA IPC 打开
            // void* remote_k_ptr = nullptr;
            // void* remote_v_ptr = nullptr;
            // cudaIpcOpenMemHandle(&remote_k_ptr, loc->ipc_k.value(), cudaIpcMemLazyEnablePeerAccess);
            // cudaIpcOpenMemHandle(&remote_v_ptr, loc->ipc_v.value(), cudaIpcMemLazyEnablePeerAccess);
            // h_k_ptrs.push_back(reinterpret_cast<const float*>(remote_k_ptr));
            // h_v_ptrs.push_back(reinterpret_cast<const float*>(remote_v_ptr));

            // 远程数据：从 host_k/v 拷贝到当前 GPU 临时缓冲区
            void* d_k, *d_v;
            size_t block_bytes = tokens_per_block * head_dim * sizeof(float);
            cudaMalloc(&d_k, block_bytes);
            cudaMalloc(&d_v, block_bytes);
            cudaMemcpy(d_k, loc->remote_host_k.data(), block_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, loc->remote_host_v.data(), block_bytes, cudaMemcpyHostToDevice);
            h_k_ptrs.push_back(reinterpret_cast<float*>(d_k));
            h_v_ptrs.push_back(reinterpret_cast<float*>(d_v));
        }
    }

    // 2. 加入 un_cached_blocks（本设备 freshly computed block）
    for (const auto* blk : un_cached_blocks) {
        h_k_ptrs.push_back(reinterpret_cast<const float*>(blk->k_ptr));
        h_v_ptrs.push_back(reinterpret_cast<const float*>(blk->v_ptr));
    }

    // 3. 上传指针数组
    const float** d_k_ptrs;
    const float** d_v_ptrs;
    cudaMalloc(&d_k_ptrs, sizeof(float*) * num_blocks);
    cudaMalloc(&d_v_ptrs, sizeof(float*) * num_blocks);
    cudaMemcpy(d_k_ptrs, h_k_ptrs.data(), sizeof(float*) * num_blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_ptrs, h_v_ptrs.data(), sizeof(float*) * num_blocks, cudaMemcpyHostToDevice);

    // 4. 分配输出
    float* d_out;
    cudaMalloc(&d_out, sizeof(float) * num_q * head_dim);

    // 5. 启动 kernel
    int threads = 64;
    int blocks_q = (num_q + threads - 1) / threads;
    // size_t shared_mem = sizeof(float) * total_kv * 2;

    attention_blockwise_forward(d_q, d_k_ptrs, d_v_ptrs, num_blocks, tokens_per_block, d_out, head_dim);

    // 6. 下载结果
    std::vector<std::vector<float>> result(num_q, std::vector<float>(head_dim));
    std::vector<float> tmp(num_q * head_dim);
    cudaMemcpy(tmp.data(), d_out, sizeof(float) * tmp.size(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_q; ++i)
        std::copy(tmp.begin() + i * head_dim, tmp.begin() + (i + 1) * head_dim, result[i].begin());

    // 7. 清理
    cudaFree(d_k_ptrs);
    cudaFree(d_v_ptrs);
    cudaFree(d_out);

    return result;
}
