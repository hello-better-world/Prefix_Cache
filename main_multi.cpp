// #include "kv_allocator.hpp"
// #include "kv_cache_block.hpp"
// #include "prefix_cache_manager.hpp"
// #include "kv_computer.hpp"


// // BlockHashType compute_prefix_hash(const std::vector<int>& token_ids) {
// //     BlockHashType hash;
// //     hash.token_ids = token_ids;

// //     // 简单哈希组合
// //     std::size_t h = 0;
// //     for (int t : token_ids) {
// //         h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
// //     }
// //     hash.hash_value = h;

// //     return hash;
// // }
// // BlockHashType compute_prefix_hash(const std::vector<int>& token_ids) {
// //     BlockHashType hash;
// //     hash.token_ids = token_ids;

// //     std::size_t h = 0;
// //     for (int t : token_ids) {
// //         h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
// //     }

// //     hash.hash_value = h;  // 仅用于 debug，不参与 unordered_map 哈希
// //     return hash;
// // }
// std::string compute_prefix_hash(const std::vector<int>& tokens) {
//     std::string key;
//     key.reserve(tokens.size() * 4);  // 预留空间

//     for (int t : tokens) {
//         key += std::to_string(t);
//         key += ',';  // 用逗号分隔防止拼接歧义（如 "12,3" vs "1,23"）
//     }

//     return key;
// }


// int main() {
//     const int device_id = 0;
//     const int num_blocks = 8;
//     const int tokens_per_block = 2;
//     const int embed_dim = 32;
//     const int head_dim = 16;
//     const size_t block_bytes = tokens_per_block * head_dim * sizeof(float);

//     // 初始化分配器和计算器
//     KVAllocator allocator(device_id, num_blocks, block_bytes);
//     KVComputer kv_computer(embed_dim, head_dim, tokens_per_block);

//     auto& cache = PrefixCacheManager::get_instance();

//     // 模拟 prompt 输入（2 个 prompt）
//     std::vector<std::vector<int>> prompts = {
//         {1, 2, 3, 4},
//         {1, 2, 5, 6}
//     };

//     for (size_t p = 0; p < prompts.size(); ++p) {
//         const auto& prompt = prompts[p];
//         std::cout << "==== Prompt " << p << ": ";
//         for (int t : prompt) std::cout << t << " ";
//         std::cout << "\n";

//         std::vector<KVCacheBlock*> used_blocks;

//         // 模拟以 tokens_per_block 为单位切分
//         for (size_t i = 0; i < prompt.size(); i += tokens_per_block) {
//             size_t end = std::min(i + tokens_per_block, prompt.size());
//             std::vector<int> block_tokens(prompt.begin() + i, prompt.begin() + end);
//             std::string hash = compute_prefix_hash(block_tokens);

//             auto kv_opt = cache.lookup(hash);

//             if (kv_opt.has_value()) {
//                 std::cout << "  Block [" << i << "," << end << "): Hit → GPU"
//                           << kv_opt->device_id << ", block_id=" << kv_opt->block_id << "\n";
//                 cache.retain(hash);
//             } else {
//                 KVCacheBlock* block = allocator.allocate();
//                 if (!block) {
//                     std::cerr << "Error: allocator returned null block.\n";
//                     std::exit(EXIT_FAILURE);
//                 }

//                 kv_computer.compute_and_fill(block, block_tokens);
                
//                 KVLocation loc;
//                 loc.device_id = device_id;
//                 loc.block_id = block->block_id;
//                 loc.is_remote = false;
//                 loc.local_k_ptr = block->k_ptr;
//                 loc.local_v_ptr = block->v_ptr;

//                 cache.insert(hash, loc);

//                 std::cout << "  Block [" << i << "," << end << "): Miss → Allocated block_id="
//                           << block->block_id << "\n";
//             }

//             used_blocks.push_back(allocator.get_block(
//                 cache.lookup(hash).value().block_id));
//         }

//         // 推理结束后：release 所有引用（简化处理）
//         for (size_t i = 0; i < prompt.size(); i += tokens_per_block) {
//             size_t end = std::min(i + tokens_per_block, prompt.size());
//             std::vector<int> block_tokens(prompt.begin() + i, prompt.begin() + end);
//             std::string hash = compute_prefix_hash(block_tokens);
//             auto kv_opt = cache.lookup(hash);
//             if (!kv_opt.has_value()) {
//                 std::cerr << "cache.lookup failed (release): hash not found\n";
//                 continue;
//             }
//             cache.release(hash);

//             KVCacheBlock* block = allocator.get_block(
//                 cache.lookup(hash).value().block_id);

//             // 如果全局 ref_cnt == 0 → 本地 ref_cnt = 0 → 可释放
//             int global_ref = cache.get_global_ref_cnt(hash);
//             int& local_ref = block->ref_cnt;
//             local_ref = global_ref;

//             if (--local_ref == 0 && cache.can_evict(hash)) {
//                 allocator.free(block);
//                 std::cout << "  Freed block_id=" << block->block_id << "\n";
//             }
//         }
//     }

//     return 0;
// }


#include "kv_allocator.hpp"
#include "kv_cache_block.hpp"
#include "prefix_cache_manager.hpp"
#include "kv_computer.hpp"
#include "attention_executor.hpp"

std::string compute_prefix_hash(const std::vector<int>& tokens) {
    std::string key;
    for (int t : tokens) {
        key += std::to_string(t);
        key += ',';
    }
    return key;
}

void verify_local_ipc_equivalence(KVCacheBlock* block, const cudaIpcMemHandle_t& ipc_k, int tokens_per_block, int head_dim) {
    std::cout << "[Local IPC Verify] device_id = " << block->device_id << "\n";
    // cudaSetDevice(block->device_id);

    void* opened_ptr = nullptr;
    cudaIpcOpenMemHandle(&opened_ptr, ipc_k, cudaIpcMemLazyEnablePeerAccess);

    for (int i = 0; i < tokens_per_block; ++i) {
        std::vector<float> original(head_dim);
        std::vector<float> opened(head_dim);

        const void* local_ptr = static_cast<char*>(block->k_ptr) + i * head_dim * sizeof(float);
        const void* remote_ptr = static_cast<char*>(opened_ptr) + i * head_dim * sizeof(float);

        cudaMemcpy(original.data(), local_ptr, sizeof(float) * head_dim, cudaMemcpyDeviceToHost);
        cudaMemcpy(opened.data(),  remote_ptr, sizeof(float) * head_dim, cudaMemcpyDeviceToHost);

        std::cout << "Token " << i << ":\n";
        std::cout << "  Local   : ";
        for (float v : original) std::cout << v << " ";
        std::cout << "\n";

        std::cout << "  IPC Open: ";
        for (float v : opened) std::cout << v << " ";
        std::cout << "\n";

        bool ok = true;
        for (int j = 0; j < head_dim; ++j) {
            if (std::abs(original[j] - opened[j]) > 1e-4) {
                ok = false;
                break;
            }
        }

        if (ok) {
            std::cout << "  ✅ match\n";
        } else {
            std::cout << "  ❌ mismatch!\n";
        }
    }

    cudaIpcCloseMemHandle(opened_ptr);
}


int main() {
    const int num_gpus = 2;
    // const int device_id = 0;
    const int num_blocks = 16;
    const int tokens_per_block = 2;
    const int embed_dim = 32;
    const int head_dim = 16;
    const size_t block_bytes = tokens_per_block * head_dim * sizeof(float);

    std::vector<std::unique_ptr<KVAllocator>> allocators;
    std::vector<std::unique_ptr<KVComputer>> computers;
    std::vector<std::unique_ptr<AttentionExecutor>> executors;

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        allocators.push_back(std::make_unique<KVAllocator>(i, num_blocks, block_bytes));
        computers.push_back(std::make_unique<KVComputer>(embed_dim, head_dim, tokens_per_block));
        executors.push_back(std::make_unique<AttentionExecutor>(head_dim));
    }

    auto& cache = PrefixCacheManager::get_instance();

    // prompts暂时不考虑不能填满block的情况
    std::vector<std::vector<int>> prompts = {
        {1, 2, 3, 4},
        {1, 2, 5, 6},
        {3, 4, 7, 8}
    };

    for (size_t p = 0; p < prompts.size(); ++p) {
        int device_id = p % num_gpus;
        cudaSetDevice(device_id);

        auto& allocator = *allocators[device_id];
        auto& kv_computer = *computers[device_id];
        auto& executor = *executors[device_id];

        const auto& prompt = prompts[p];
        std::cout << "==== Prompt " << p << ": ";
        for (int t : prompt) std::cout << t << " ";
        std::cout << "\n";

        // std::vector<KVCacheBlock*> blocks;
        // cached_blocks在前面，un_cached_blocks在后面，可以分开计算
        std::vector<KVCacheBlock*> un_cached_blocks;
        std::vector<std::optional<KVLocation>> cached_blocks;
        std::vector<std::vector<float>> q_vecs;

        for (size_t i = 0; i < prompt.size(); i += tokens_per_block) {
            size_t end = std::min(i + tokens_per_block, prompt.size());
            std::vector<int> block_tokens(prompt.begin() + i, prompt.begin() + end);
            std::vector<int> prefix_tokens(prompt.begin(), prompt.begin() + end);
            std::string hash = compute_prefix_hash(prefix_tokens);

            auto kv_loc = cache.lookup(hash);
            if (kv_loc.has_value()) {
                // 增加描述，命中的第几个GPU
                std::cout << "  Block [" << i << "," << end << "): Hit → GPU"
                          << kv_loc->device_id << ", block_id=" << kv_loc->block_id << "\n";
                cache.retain(hash);
                
                cached_blocks.push_back(kv_loc);  // 不管是本地还是其他GPU，都存储KVLocation，因为这个类中包含了local和remote
            } else {
                std::cout << "  Block [" << i << "," << end << "): Miss → Allocated\n";
                KVCacheBlock* block = allocator.allocate();
                kv_computer.compute_and_fill(block, block_tokens);

                cudaDeviceSynchronize();

                KVLocation loc;
                loc.device_id = device_id;  // 根据device判断
                loc.block_id = block->block_id;  // 这个也不需要
                loc.local_k_ptr = block->k_ptr;
                loc.local_v_ptr = block->v_ptr;

                // 存储host_k和host_v
                size_t block_bytes = tokens_per_block * head_dim * sizeof(float);
                loc.remote_host_k.resize(tokens_per_block * head_dim);
                loc.remote_host_v.resize(tokens_per_block * head_dim);
                cudaMemcpy(loc.remote_host_k.data(), block->k_ptr, block_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(loc.remote_host_v.data(), block->v_ptr, block_bytes, cudaMemcpyDeviceToHost);

                // cudaIpcMemHandle_t k_handle, v_handle;
                // cudaIpcGetMemHandle(&k_handle, block->k_ptr);
                // cudaIpcGetMemHandle(&v_handle, block->v_ptr);
                // loc.ipc_k = k_handle;
                // loc.ipc_v = v_handle;

                // 立即验证：k_ptr 与自己打开 ipc_k 得到的地址是否一致内容
                // verify_local_ipc_equivalence(block, k_handle, tokens_per_block, head_dim);

                cache.insert(hash, loc);
                cache.retain(hash);
                un_cached_blocks.push_back(block);  // 这里可以存储block，blocks改成un_cached_blocks

                for (int token_id : block_tokens) {
                    auto emb = kv_computer.embed_token(token_id);
                    auto q = kv_computer.matmul(emb, kv_computer.q_weight_);
                    q_vecs.push_back(q);
                }
            }
        }

        // upload Q to GPU
        int num_q = q_vecs.size();
        std::vector<float> host_q(num_q * head_dim);
        for (int i = 0; i < num_q; ++i)
            std::copy(q_vecs[i].begin(), q_vecs[i].end(), host_q.begin() + i * head_dim);

        float* d_q;
        cudaMalloc(&d_q, sizeof(float) * host_q.size());
        cudaMemcpy(d_q, host_q.data(), sizeof(float) * host_q.size(), cudaMemcpyHostToDevice);
        
        std::cout << "[DEBUG] query size = " << q_vecs.size()
                << ", cache KV = " << cached_blocks.size() * tokens_per_block
                << ", new KV = " << un_cached_blocks.size() * tokens_per_block
                << ", total KV = " << (cached_blocks.size() + un_cached_blocks.size()) * tokens_per_block << "\n";

        // 输入 cached_blocks 和 un_cached_blocks
        auto outputs = executor.run_gpu_blockwise(d_q, num_q, un_cached_blocks, cached_blocks, tokens_per_block, device_id);  // 加入device_id的参数，判断读取k_ptr还是ipc_k
        cudaFree(d_q);

        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "Attention output " << i << ": ";
            for (float v : outputs[i]) std::cout << v << " ";
            std::cout << "\n";
        }
    }

    // // 推理结束后：release 所有引用（简化处理）
    // for (size_t i = 0; i < prompt.size(); i += tokens_per_block) {
    //     size_t end = std::min(i + tokens_per_block, prompt.size());
    //     std::vector<int> block_tokens(prompt.begin() + i, prompt.begin() + end);
    //     std::string hash = compute_prefix_hash(block_tokens);
    //     auto kv_opt = cache.lookup(hash);
    //     if (!kv_opt.has_value()) {
    //         std::cerr << "cache.lookup failed (release): hash not found\n";
    //         continue;
    //     }
    //     cache.release(hash);

    //     KVCacheBlock* block = allocator.get_block(
    //         cache.lookup(hash).value().block_id);

    //     // 如果全局 ref_cnt == 0 → 本地 ref_cnt = 0 → 可释放
    //     int global_ref = cache.get_global_ref_cnt(hash);
    //     int& local_ref = block->ref_cnt;
    //     local_ref = global_ref;

    //     if (--local_ref == 0 && cache.can_evict(hash)) {
    //         allocator.free(block);
    //         std::cout << "  Freed block_id=" << block->block_id << "\n";
    //     }
    // }

    return 0;
}
