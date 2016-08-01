#include <iostream>
#include "include/data_types.cuh"
#include "include/hpc_helpers.cuh"
#include "include/hash_functions.cuh"

template <
    typename entry_t,
    typename index_t>__global__
void init_kernel(
    entry_t * Data,
    index_t capacity) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (index_t index = thid; index < capacity; index += blockDim.x*gridDim.x)
        Data[index] = entry_t::get_empty();
}

template <
    typename entry_t,
    typename index_t,
    typename funct_t,
    typename probe_t>__global__
void insert_kernel(
    entry_t * Data,
    index_t num_elements,
    index_t capacity,
    funct_t hash_func,
    probe_t prob_func) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t bits_key = entry_t::bits_for_key;
    const index_t bits_val = entry_t::bits_for_val;
    const entry_t nil = entry_t::get_empty();


    for (index_t i = thid; i < num_elements; i += blockDim.x*gridDim.x) {

        const index_t key = (i/2) % ((1UL << bits_key)-1);
        const index_t val = i % (1UL << bits_val);

        index_t index = hash_func(key) % capacity;

        for (index_t iters = 0; iters < capacity; ++iters) {

            entry_t entry; entry.set_pair(key, val);
            const entry_t pair = Data[index];

            if (pair == nil) {
                const auto target = (unsigned long long int *) Data + index;
                const auto expect = (unsigned long long int  ) nil.payload;
                const auto value  = (unsigned long long int  ) entry.payload;
                const auto result = atomicCAS(target, expect, value);

                if (result == nil.payload)
                    return;
            }

            index = prob_func(index, key, iters) % capacity;
        }
    }
}

int main () {

    cudaSetDevice(0);

    uint64_t num_elements = 1UL << 20;
    uint64_t capacity = num_elements/0.9;
    typedef KeyValuePair_t<uint64_t, 32, 32> entry_t;

    TIMERSTART(malloc_data)
    entry_t * Data = nullptr;
    cudaMalloc(&Data, sizeof(entry_t)*capacity);                          CUERR
    TIMERSTOP(malloc_data)

    TIMERSTART(init_data)
    init_kernel<<<SDIV(capacity, 1024), 1024>>>(Data, capacity);          CUERR
    TIMERSTOP(init_data)

    const auto hash_func = identity_map_t();
    const auto prob_func = linear_probing_scheme_t();

    TIMERSTART(insert_data)
    insert_kernel<<<SDIV(num_elements, 32), 32>>>(Data,
                                                      num_elements,
                                                      capacity,
                                                      hash_func,
                                                      prob_func);         CUERR
    TIMERSTOP(insert_data)

    std::cout << sizeof(unsigned long long int) << std::endl;

}
