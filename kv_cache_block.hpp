#pragma once
#include <cuda_runtime.h>
#include <optional>
#include <iostream>

// Block 的基本属性
struct KVCacheBlock {
    int block_id;
    int device_id;
    int ref_cnt = 0;

    bool is_remote = false;

    void* k_ptr = nullptr;
    void* v_ptr = nullptr;

    // nvshmem分配空间的初始位置（nvshmem给的案例的指针类型很明确，就是float*）
    float* shm_k_base_ = nullptr;
    float* shm_v_base_ = nullptr;

    std::optional<cudaIpcMemHandle_t> ipc_k;  // 不需要
    std::optional<cudaIpcMemHandle_t> ipc_v;  // 不需要

    // 🔻 用于双向链表：free block 管理
    KVCacheBlock* prev_free_block = nullptr;
    KVCacheBlock* next_free_block = nullptr;

    void incr_ref() { ++ref_cnt; }
    void decr_ref() { --ref_cnt; }

    void reset_hash() {
        // 清理哈希（后续可扩展）
    }

    void print() const {
        std::cout << "[KVCacheBlock] id=" << block_id
                  << " dev=" << device_id
                  << " ref_cnt=" << ref_cnt
                  << " is_remote=" << is_remote
                  << "\n";
    }
};

