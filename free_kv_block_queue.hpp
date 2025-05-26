#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include "kv_cache_block.hpp"

// 管理本地 block 的 free queue（双向链表）
class FreeKVCacheBlockQueue {
public:
    FreeKVCacheBlockQueue(std::vector<std::unique_ptr<KVCacheBlock>>& blocks) {
        num_free_blocks_ = blocks.size();
        if (blocks.empty()) return;

        free_list_head_ = blocks[0].get();
        free_list_tail_ = blocks.back().get();

        for (size_t i = 0; i < blocks.size(); ++i) {
            if (i > 0) {
                blocks[i]->prev_free_block = blocks[i - 1].get();
            }
            if (i < blocks.size() - 1) {
                blocks[i]->next_free_block = blocks[i + 1].get();
            }
        }
    }

    KVCacheBlock* popleft() {
        if (!free_list_head_) throw std::runtime_error("No free blocks available");
        KVCacheBlock* block = free_list_head_;
        remove(block);
        return block;
    }

    void append(KVCacheBlock* block) {
        if (!block) return;

        block->prev_free_block = free_list_tail_;
        block->next_free_block = nullptr;

        if (free_list_tail_) {
            free_list_tail_->next_free_block = block;
        } else {
            free_list_head_ = block;
        }

        free_list_tail_ = block;
        ++num_free_blocks_;
    }

    size_t size() const { return num_free_blocks_; }

private:
    void remove(KVCacheBlock* block) {
        if (!block) return;

        if (block->prev_free_block)
            block->prev_free_block->next_free_block = block->next_free_block;

        if (block->next_free_block)
            block->next_free_block->prev_free_block = block->prev_free_block;

        if (block == free_list_head_)
            free_list_head_ = block->next_free_block;

        if (block == free_list_tail_)
            free_list_tail_ = block->prev_free_block;

        block->prev_free_block = nullptr;
        block->next_free_block = nullptr;

        --num_free_blocks_;
    }

    KVCacheBlock* free_list_head_ = nullptr;
    KVCacheBlock* free_list_tail_ = nullptr;
    size_t num_free_blocks_ = 0;
};
