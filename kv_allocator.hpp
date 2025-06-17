// #pragma once

// #include <vector>
// #include <unordered_map>
// #include <cuda_runtime.h>
// #include <memory>
// #include "kv_cache_block.hpp"
// #include "free_kv_block_queue.hpp"

// class KVAllocator {
// public:
//     KVAllocator(int device_id, int num_blocks, size_t block_bytes);
//     ~KVAllocator();

//     // 分配一个 block
//     KVCacheBlock* allocate();

//     // 回收一个 block
//     void free(KVCacheBlock* block);

//     // 显存句柄导出（用于 IPC）
//     cudaIpcMemHandle_t export_k_handle(int block_id);
//     cudaIpcMemHandle_t export_v_handle(int block_id);

//     // 获取 block 指针（本地）
//     KVCacheBlock* get_block(int block_id);

// private:
//     int device_id_;
//     size_t block_bytes_;
//     int num_blocks_;

//     std::vector<void*> k_memory_blocks_;
//     std::vector<void*> v_memory_blocks_;
//     std::vector<std::unique_ptr<KVCacheBlock>> blocks_;

//     std::unique_ptr<FreeKVCacheBlockQueue> free_queue_;
// };

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "kv_cache_block.hpp"
#include "free_kv_block_queue.hpp"

class KVAllocator {
public:
    KVAllocator(int device_id, int num_blocks, size_t block_bytes);
    ~KVAllocator();

    KVCacheBlock* allocate();
    void free(KVCacheBlock* block);
    KVCacheBlock* get_block(int block_id);

    void* get_k_base() const { return k_base_; }
    void* get_v_base() const { return v_base_; }

private:
    int device_id_;
    int num_blocks_;
    size_t block_bytes_;

    void* k_base_ = nullptr;
    void* v_base_ = nullptr;
    
    float* shm_k_base_; 
    float* shm_v_base_;  // 放在allocator中还是放在主函数中？

    std::vector<std::unique_ptr<KVCacheBlock>> blocks_;
    std::unique_ptr<FreeKVCacheBlockQueue> free_queue_;
};