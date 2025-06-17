#pragma once
#include <unordered_map>
#include <mutex>
#include <optional>
#include <atomic>
#include <vector>
#include <functional>
#include <memory>
#include "kv_cache_block.hpp"

// struct BlockHashType {
//     std::size_t hash_value;
//     std::vector<int> token_ids;

//     bool operator==(const BlockHashType& other) const {
//         return hash_value == other.hash_value && token_ids == other.token_ids;
//     }
// };
// struct BlockHashType {
//     std::size_t hash_value;
//     std::vector<int> token_ids;

//     bool operator==(const BlockHashType& other) const {
//         return token_ids == other.token_ids;
//     }
// };

// struct BlockHashType {
//     std::vector<int> token_ids;

//     bool operator==(const BlockHashType& other) const {
//         return token_ids == other.token_ids;
//     }
// };

// 自定义 hash
// struct BlockHashTypeHasher {
//     std::size_t operator()(const BlockHashType& key) const {
//         std::size_t h = key.hash_value;
//         for (int t : key.token_ids) {
//             h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
//         }
//         return h;
//     }
// };
// struct BlockHashTypeHasher {
//     std::size_t operator()(const BlockHashType& key) const {
//         // 仅对 token_ids 做哈希，忽略 hash_value 字段
//         std::size_t h = 0;
//         for (int t : key.token_ids) {
//             h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
//         }
//         return h;
//     }
// };
// struct BlockHashTypeHasher {
//     std::size_t operator()(const BlockHashType& key) const {
//         std::size_t h = 0;
//         for (int t : key.token_ids) {
//             h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
//         }
//         return h;
//     }
// };

struct KVLocation {
    int device_id;
    int block_id;
    bool is_remote;  // 不需要

    void* local_k_ptr = nullptr;
    void* local_v_ptr = nullptr;

    std::optional<cudaIpcMemHandle_t> ipc_k;
    std::optional<cudaIpcMemHandle_t> ipc_v;

    std::vector<float> remote_host_k;  // host 上缓存 K 值，以假装这块数据可以被其他GPU上使用
    std::vector<float> remote_host_v;  // host 上缓存 V 值
};

// struct CachedKVEntry {
//     KVLocation loc;
//     std::atomic<int> ref_cnt;
// };

struct CachedKVEntry {
    KVLocation loc;
    std::atomic<int> ref_cnt;

    CachedKVEntry() : ref_cnt(0) {}
    CachedKVEntry(const KVLocation& l, int c) : loc(l), ref_cnt(c) {}
};


class PrefixCacheManager {
public:
    static PrefixCacheManager& get_instance() {
        static PrefixCacheManager instance;
        return instance;
    }

    std::optional<KVLocation> lookup(const std::string& hash);

    // void insert(const BlockHashType& hash, const KVLocation& loc);
    void insert(const std::string& key, const KVLocation& loc);
    // void retain(const BlockHashType& hash);
    void retain(const std::string& hash);
    // void release(const BlockHashType& hash);
    void release(const std::string& hash);

    int get_global_ref_cnt(const std::string& hash);

    bool can_evict(const std::string& hash);

private:
    PrefixCacheManager() = default;

    std::mutex mutex_;
    // std::unordered_map<BlockHashType, CachedKVEntry, BlockHashTypeHasher> table_;
    std::unordered_map<std::string, CachedKVEntry> table_;

};
