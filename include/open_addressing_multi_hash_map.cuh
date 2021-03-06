#ifndef GANJA_OPEN_ADDRESSING_MULTI_HASH_MAP_CUH
#define GANJA_OPEN_ADDRESSING_MULTI_HASH_MAP_CUH

#include <omp.h>
#include <atomic>
#include <vector>
#include "data_types.cuh"
#include "atomic_helpers.cuh"

template <
    typename index_t,
    index_t bits_key,
    index_t bits_val,
    typename funct_t,
    typename probe_t>
struct OpenAddressingMultiHashMap {

    typedef KeyValuePair_t<index_t,bits_key,bits_val> entry_t;
    typedef std::atomic<entry_t> atomic_t;

    static constexpr index_t bits_for_key = bits_key;
    static constexpr index_t bits_for_val = bits_val;

    atomic_t * data;
    const funct_t hash_func;
    const probe_t prob_func;
    const index_t capacity;
    const index_t probe_length;
    std::atomic<index_t> size = {0};

    OpenAddressingMultiHashMap(
        const index_t capacity_,
        const funct_t hash_func_,
        const probe_t prob_func_) : capacity(capacity_),
                                    probe_length(capacity_),
                                    hash_func(hash_func_),
                                    prob_func(prob_func_) {

        data = new atomic_t[capacity];

        # pragma omp parallel for
        for (index_t index = 0; index < capacity; index++)
            data[index].store(entry_t::get_empty());
    }

    ~OpenAddressingMultiHashMap() {

        delete [] data;
    }

    std::vector<index_t> get(
        const index_t& key) const {

        std::vector<index_t> result;
        index_t index = hash_func(key) % capacity;

        if (key >= entry_t::mask_key)
            return result;

        for (index_t iters = 0; iters < probe_length; ++iters) {

            entry_t probed = data[index].load(std::memory_order_relaxed);

            if (probed == entry_t::get_empty())
                return result;

            if (probed.get_key() == key)
                result.push_back(probed.get_val());

            index = prob_func(index, iters, key) % capacity;
        }

        return result;
    }

    bool add(
        const index_t key,
        const index_t val) {

        if (key >= entry_t::mask_key || val >= entry_t::mask_val)
            return false;

        entry_t nil = entry_t::get_empty(), entry; entry.set_pair(key, val);
        index_t index = hash_func(key) % capacity;

        for (index_t iters = 0; iters < probe_length; ++iters) {

            const entry_t pair = data[index].load(std::memory_order_relaxed);

            if (pair == nil) {
                std::atomic_compare_exchange_strong(&data[index], &nil, entry);

                if (nil == entry_t::get_empty())
                    return ++size;
                else
                    nil = entry_t::get_empty();
            }

            index = prob_func(index, iters, key) % capacity;
        }

        return false;
    }

};

#endif
