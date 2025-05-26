#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <utility>
#include <mutex>

class RemoteBlockMapper {
public:
    static RemoteBlockMapper& get_instance() {
        static RemoteBlockMapper instance;
        return instance;
    }

    // 获取远程 block 的地址（K or V）
    void* get_or_open_ptr(int remote_dev, int block_id, const cudaIpcMemHandle_t& handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = std::make_pair(remote_dev, block_id);
        if (ptr_cache_.count(key)) {
            return ptr_cache_[key];
        }

        void* ptr = nullptr;
        cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaIpcOpenMemHandle failed");
        }
        ptr_cache_[key] = ptr;
        return ptr;
    }

    // 关闭所有远程句柄（可选）
    void cleanup() {
        for (auto& entry : ptr_cache_) {
            cudaIpcCloseMemHandle(entry.second);
        }
        ptr_cache_.clear();
    }

private:
    RemoteBlockMapper() = default;
    ~RemoteBlockMapper() { cleanup(); }

    std::mutex mutex_;
    std::unordered_map<std::pair<int, int>, void*, pair_hash> ptr_cache_;

    struct pair_hash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ std::hash<int>()(p.second << 16);
        }
    };
};
