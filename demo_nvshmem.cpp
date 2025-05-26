// Compile: mpicxx -std=c++17 -fopenmp demo_nvshmem.cpp -lzmq -lnvshmem_host -lnvshmem_device -o demo_nvshmem

#include <iostream>
#include <unordered_map>
#include <string>
#include <zmq.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4  // 每个 block 4 个 float，示意用

struct KVLocation {
    float* nvshmem_ptr;
    int pe;
};

std::unordered_map<uint64_t, KVLocation> global_cache_map;

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void master_process(int mype) {
    zmq::context_t context(1);
    zmq::socket_t responder(context, ZMQ_REP);
    responder.bind("tcp://*:5555");

    std::cout << "[Master PE " << mype << "] Ready.\n";

    while (true) {
        zmq::message_t request;
        responder.recv(request, zmq::recv_flags::none);
        std::string req_str(static_cast<char*>(request.data()), request.size());

        if (req_str.starts_with("lookup:")) {
            uint64_t hash = std::stoull(req_str.substr(7));
            auto it = global_cache_map.find(hash);
            if (it != global_cache_map.end()) {
                std::string resp = "found:" + std::to_string(it->second.pe) + ":" +
                    std::to_string(reinterpret_cast<uintptr_t>(it->second.nvshmem_ptr));
                responder.send(zmq::buffer(resp), zmq::send_flags::none);
            } else {
                responder.send(zmq::buffer("not_found"), zmq::send_flags::none);
            }
        } else if (req_str.starts_with("insert:")) {
            size_t sep1 = req_str.find(':', 7);
            size_t sep2 = req_str.find(':', sep1 + 1);
            uint64_t hash = std::stoull(req_str.substr(7, sep1 - 7));
            int pe = std::stoi(req_str.substr(sep1 + 1, sep2 - sep1 - 1));
            uintptr_t addr = std::stoull(req_str.substr(sep2 + 1));
            global_cache_map[hash] = KVLocation{reinterpret_cast<float*>(addr), pe};
            std::cout << "[Master] Inserted hash " << hash << " from PE " << pe << " addr " << addr << "\n";
            responder.send(zmq::buffer("ok"), zmq::send_flags::none);
        }
    }
}

void worker_process(int mype) {
    zmq::context_t context(1);
    zmq::socket_t requester(context, ZMQ_REQ);
    requester.connect("tcp://localhost:5555");

    std::unordered_map<uint64_t, float*> local_cache;
    uint64_t block_hash = 12345;

    // 用 NVSHMEM 分配显存 block
    float* local_kv = (float*)nvshmem_malloc(BLOCK_SIZE * sizeof(float));
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        local_kv[i] = 100.0f * mype + i;  // 写入一些数据用于演示
    }

    // Step 1: 先查本地 cache
    if (local_cache.count(block_hash)) {
        std::cout << "[Worker PE " << mype << "] Local hit.\n";
    } else {
        std::string lookup_msg = "lookup:" + std::to_string(block_hash);
        requester.send(zmq::buffer(lookup_msg), zmq::send_flags::none);
        zmq::message_t reply;
        requester.recv(reply, zmq::recv_flags::none);
        std::string resp(static_cast<char*>(reply.data()), reply.size());

        if (resp.starts_with("found:")) {
            size_t sep1 = resp.find(':', 6);
            size_t sep2 = resp.find(':', sep1 + 1);
            int remote_pe = std::stoi(resp.substr(6, sep1 - 6));
            uintptr_t addr = std::stoull(resp.substr(sep1 + 1));
            float* remote_ptr = reinterpret_cast<float*>(addr);

            float recv_block[BLOCK_SIZE];
            nvshmem_getmem(recv_block, remote_ptr, BLOCK_SIZE * sizeof(float), remote_pe);

            std::cout << "[Worker PE " << mype << "] Read from PE " << remote_pe << ": ";
            for (int i = 0; i < BLOCK_SIZE; ++i)
                std::cout << recv_block[i] << " ";
            std::cout << "\n";
        } else {
            std::cout << "[Worker PE " << mype << "] Cache miss. Insert local block.\n";
            local_cache[block_hash] = local_kv;

            std::string insert_msg = "insert:" + std::to_string(block_hash) + ":" +
                std::to_string(mype) + ":" + std::to_string(reinterpret_cast<uintptr_t>(local_kv));
            requester.send(zmq::buffer(insert_msg), zmq::send_flags::none);
            requester.recv(reply, zmq::recv_flags::none);
        }
    }

    nvshmem_barrier_all();
}

int main(int argc, char** argv) {
    // MPI + NVSHMEM 初始化
    MPI_Init(&argc, &argv);
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mype = nvshmem_my_pe();
    int dev_id = mype;
    checkCuda(cudaSetDevice(dev_id), "Set device failed");

    if (mype == 0)
        master_process(mype);
    else
        worker_process(mype);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
