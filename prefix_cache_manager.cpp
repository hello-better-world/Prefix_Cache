#include "prefix_cache_manager.hpp"

std::optional<KVLocation> PrefixCacheManager::lookup(const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    // std::cout << "[lookup] key = \"" << hash << "\"\n";
    auto it = table_.find(hash);
    if (it == table_.end()) return std::nullopt;
    return it->second.loc;
}

// void PrefixCacheManager::insert(const BlockHashType& hash, const KVLocation& loc) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     // table_[hash] = CachedKVEntry{loc, 1};  // 初始化ref_cnt = 1
//     table_.emplace(hash, CachedKVEntry{loc, 1});
// }

void PrefixCacheManager::insert(const std::string& hash, const KVLocation& loc) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 先 default 构造 entry
    // std::cout << "[insert] key = \"" << hash << "\"\n";
    auto& entry = table_[hash];
    entry.loc = loc;
    entry.ref_cnt.store(1);
}


void PrefixCacheManager::retain(const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    table_[hash].ref_cnt++;
}

void PrefixCacheManager::release(const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = table_[hash];
    if (--entry.ref_cnt < 0) {
        throw std::runtime_error("release(): ref_cnt < 0");
    }
}

int PrefixCacheManager::get_global_ref_cnt(const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_[hash].ref_cnt.load();
}

bool PrefixCacheManager::can_evict(const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_[hash].ref_cnt == 0;
}
