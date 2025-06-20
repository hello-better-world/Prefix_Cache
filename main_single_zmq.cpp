#include <zmq.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "kv_allocator.hpp"
#include "kv_cache_block.hpp"
#include "prefix_cache_manager.hpp"
#include "kv_computer.hpp"
#include "attention_executor.hpp"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

std::string compute_prefix_hash(const std::vector<int>& tokens) {
    std::string key;
    for (int t : tokens) {
        key += std::to_string(t);
        key += ',';
    }
    return key;
}

int main() {
    const int device_id = 0;
    const int num_blocks = 16;
    const int tokens_per_block = 2;
    const int embed_dim = 32;
    const int head_dim = 16;
    const size_t block_bytes = tokens_per_block * head_dim * sizeof(float);
    float* shm_k_base, *shm_v_base;  // 放在allocator中还是放在主函数中？
    int mype, npes, mype_node;

    KVAllocator allocator(device_id, num_blocks, block_bytes);  // 包含了free_queue_
    KVComputer kv_computer(embed_dim, head_dim, tokens_per_block);
    AttentionExecutor executor(head_dim);
    auto& cache = PrefixCacheManager::get_instance();  // local_cache

    nvshmem_init();  // 或 nvshmemx_init_attr(...) 
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));

    // NVSHMEM 分配共享内存池，可以在cuda kernel中直接使用
    shm_k_base = nvshmem_malloc(num_blocks * block_bytes);
    shm_v_base = nvshmem_malloc(num_blocks * block_bytes);
    if (!shm_k_base || !shm_v_base) {
        std::cerr << "nvshmem_malloc failed for shared pool on device " << device_id_ << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // prompts暂时不考虑不能填满block的情况
    std::vector<std::vector<int>> prompts = {
        {1, 2, 3, 4},
        {1, 2, 5, 6},
        {3, 4, 7, 8}
    };

    zmq::context_t zmq_context(1);
    zmq::socket_t zmq_socket(zmq_context, ZMQ_REQ);
    zmq_socket.connect("tcp://localhost:5555");

    for (size_t p = 0; p < prompts.size(); ++p) {
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
            if (!kv_loc.has_value()) {
                // 本地miss → 发ZeroMQ请求master
                std::string lookup_msg = "lookup:" + std::to_string(hash);
                zmq_socket.send(zmq::buffer(lookup_msg), zmq::send_flags::none);
                zmq::message_t reply;
                zmq_socket.recv(reply, zmq::recv_flags::none);
                std::string resp(static_cast<char*>(reply.data()), reply.size());

                if (resp.starts_with("found:")) {
                    // master命中 → 从远程pe拉取kv
                    size_t sep1 = resp.find(':', 6);
                    size_t sep2 = resp.find(':', sep1 + 1);
                    int remote_pe = std::stoi(resp.substr(6, sep1 - 6));  // 其他PE号，这个很重要
                    uintptr_t addr = std::stoull(resp.substr(sep1 + 1));
                    float* remote_ptr = reinterpret_cast<float*>(addr);  // 地址本身就是float*，这里要不要reinterpret_cast转换

                    // 本地设置空间去接远程pe的数据
                    // put/get的API案例中，dest和source都得是nvshmem分配的内存（称为对称地址）
                    // 本地分配了一大片空间，但是都已经和本地的block绑定了
                    // 所以是不是还得再分配一片对称地址，专门用于存储其他PE的数据，为了简单，我们用完一次就重置，并且相应的数据不加入本地查找表
                    // nvshmemx_float_get_block 函数需要在cuda kernel中调用
                    float* local_k = (float*)malloc(tokens_per_block * head_dim * sizeof(float));
                    float* local_v = (float*)malloc(tokens_per_block * head_dim * sizeof(float));
                    nvshmem_getmem(local_k, remote_ptr, tokens_per_block * head_dim * sizeof(float), remote_pe);
                    nvshmem_getmem(local_v, remote_ptr + tokens_per_block * head_dim, tokens_per_block * head_dim * sizeof(float), remote_pe);

                    // 给block的v_ptr（shm_v_base_）和k_ptr（shm_k_base_）赋值就好了，其他的都不用
                    KVLocation remote_loc;
                    remote_loc.device_id = device_id;
                    remote_loc.block_id = -1;  // remote，无需 block_id
                    remote_loc.local_k_ptr = local_k;
                    remote_loc.local_v_ptr = local_v;
                    remote_loc.remote_pe = remote_pe;

                    cached_blocks.push_back(remote_loc);
                    continue;
                }
            }

            if (kv_loc.has_value()) {
                std::cout << "  Block [" << i << "," << end << "): Hit → GPU"
                        << kv_loc->device_id << ", block_id=" << kv_loc->block_id << "\n";
                cache.retain(hash);  // table_[hash].ref_cnt++;
                cached_blocks.push_back(kv_loc);
                continue;
            }

            // 本地和master都miss，自己生成block
            // ？应该从free_queue_中分配block，free_queue_中包含的block都是带有分配好的空间的
            // 
            std::cout << "  Block [" << i << "," << end << "): Miss → Allocated\n";
            float* k_buf = (float*)nvshmem_malloc(tokens_per_block * head_dim * sizeof(float));
            float* v_buf = (float*)nvshmem_malloc(tokens_per_block * head_dim * sizeof(float));
            kv_computer.compute_and_fill(k_buf, v_buf, block_tokens);  // 假设改过 compute_and_fill 支持 float* 写入

            KVLocation new_loc;  // KVLocation 和 
            new_loc.device_id = device_id;
            new_loc.block_id = -1;
            new_loc.local_k_ptr = k_buf;
            new_loc.local_v_ptr = v_buf;

            cache.insert(hash_str, new_loc);
            cache.retain(hash_str);
            un_cached_blocks.push_back(new_loc);

            // 向 master 注册
            std::string insert_msg = "insert:" + std::to_string(block_hash) + ":" +
                                    std::to_string(device_id) + ":" +
                                    std::to_string(reinterpret_cast<uintptr_t>(k_buf));  // 假设连续K+V
            zmq_socket.send(zmq::buffer(insert_msg), zmq::send_flags::none);
            zmq_socket.recv(reply, zmq::recv_flags::none);
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
