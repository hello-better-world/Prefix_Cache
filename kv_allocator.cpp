// #include "kv_allocator.hpp"

// KVAllocator::KVAllocator(int device_id, int num_blocks, size_t block_bytes)
//     : device_id_(device_id), num_blocks_(num_blocks), block_bytes_(block_bytes) {

//     cudaSetDevice(device_id_);

//     k_memory_blocks_.resize(num_blocks_);
//     v_memory_blocks_.resize(num_blocks_);
//     blocks_.resize(num_blocks_);

//     for (int i = 0; i < num_blocks_; ++i) {
//         void* k_ptr;
//         void* v_ptr;
//         // cudaMalloc(&k_ptr, block_bytes_);
//         // cudaMalloc(&v_ptr, block_bytes_);
//         cudaError_t err1 = cudaMalloc(&k_ptr, block_bytes_);
//         if (err1 != cudaSuccess) {
//             std::cerr << "cudaMalloc failed for k_ptr: " << cudaGetErrorString(err1) << "\n";
//             std::exit(EXIT_FAILURE);
//         }

//         cudaError_t err2 = cudaMalloc(&v_ptr, block_bytes_);
//         if (err2 != cudaSuccess) {
//             std::cerr << "cudaMalloc failed for v_ptr: " << cudaGetErrorString(err2) << "\n";
//             std::exit(EXIT_FAILURE);
//         }


//         k_memory_blocks_[i] = k_ptr;
//         v_memory_blocks_[i] = v_ptr;

//         auto block = std::make_unique<KVCacheBlock>();
//         block->block_id = i;
//         block->device_id = device_id_;
//         block->k_ptr = k_ptr;
//         block->v_ptr = v_ptr;
//         blocks_[i] = std::move(block);
//     }

//     free_queue_ = std::make_unique<FreeKVCacheBlockQueue>(blocks_);
// }

// KVAllocator::~KVAllocator() {
//     for (int i = 0; i < num_blocks_; ++i) {
//         cudaFree(k_memory_blocks_[i]);
//         cudaFree(v_memory_blocks_[i]);
//     }
// }

// KVCacheBlock* KVAllocator::allocate() {
//     return free_queue_->popleft();
// }

// void KVAllocator::free(KVCacheBlock* block) {
//     block->reset_hash();
//     block->ref_cnt = 0;
//     free_queue_->append(block);
// }

// cudaIpcMemHandle_t KVAllocator::export_k_handle(int block_id) {
//     cudaIpcMemHandle_t handle;
//     cudaIpcGetMemHandle(&handle, k_memory_blocks_[block_id]);
//     return handle;
// }

// cudaIpcMemHandle_t KVAllocator::export_v_handle(int block_id) {
//     cudaIpcMemHandle_t handle;
//     cudaIpcGetMemHandle(&handle, v_memory_blocks_[block_id]);
//     return handle;
// }

// KVCacheBlock* KVAllocator::get_block(int block_id) {
//     return blocks_[block_id].get();
// }

#include "kv_allocator.hpp"
#include <iostream>

KVAllocator::KVAllocator(int device_id, int num_blocks, size_t block_bytes)
    : device_id_(device_id), num_blocks_(num_blocks), block_bytes_(block_bytes) {
    
    cudaSetDevice(device_id_);

    cudaError_t err1 = cudaMalloc(&k_base_, num_blocks_ * block_bytes_);
    cudaError_t err2 = cudaMalloc(&v_base_, num_blocks_ * block_bytes_);
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "cudaMalloc failed for base pool: "
                  << cudaGetErrorString(err1) << " / "
                  << cudaGetErrorString(err2) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 初始化每个 block，设定其逻辑视图指针
    for (int i = 0; i < num_blocks_; ++i) {
        auto block = std::make_unique<KVCacheBlock>();
        block->block_id = i;
        block->device_id = device_id_;
        block->k_ptr = static_cast<char*>(k_base_) + i * block_bytes_;
        block->v_ptr = static_cast<char*>(v_base_) + i * block_bytes_;
        cudaIpcMemHandle_t k_handle, v_handle;
        cudaIpcGetMemHandle(&k_handle, block->k_ptr);
        cudaIpcGetMemHandle(&v_handle, block->v_ptr);
        block->ipc_k = k_handle;
        block->ipc_v = v_handle;
        blocks_.push_back(std::move(block));
    }

    free_queue_ = std::make_unique<FreeKVCacheBlockQueue>(blocks_);
}

KVAllocator::~KVAllocator() {
    if (k_base_) cudaFree(k_base_);
    if (v_base_) cudaFree(v_base_);
}

KVCacheBlock* KVAllocator::allocate() {
    return free_queue_->popleft();
}

void KVAllocator::free(KVCacheBlock* block) {
    block->reset_hash();  // 不对，放入free_queue_的时候还不应该reset_hash，当这个block被再次使用的时候才会reset_hash
    block->ref_cnt = 0;
    free_queue_->append(block);
}

KVCacheBlock* KVAllocator::get_block(int block_id) {
    return blocks_[block_id].get();
}
