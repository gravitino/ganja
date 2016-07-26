#include <iostream>
#include <random>
#include <algorithm>

#include "include/hpc_helpers.cuh"
#include "include/hash_functions.cuh"
#include "include/robin_hood_multi_hash_map.cuh"
#include "include/open_addressing_multi_hash_map.cuh"

int main (int argc, char * argv[]) {

    // configure experiment
    constexpr uint64_t seed = 42;
    constexpr uint64_t max_elements = 1UL << 28;
    constexpr uint64_t vals_per_key = 1UL << 7;

    // configure hash map
    constexpr double load = 0.80;
    constexpr uint64_t capacity = max_elements/load;
    constexpr uint64_t bits_key = 30, bits_val = 28, bits_cnt = 4;

    // mueller hash + linear probing of keys
    auto hash_func = mueller_hash_uint32_t();
    auto prob_func = linear_probing_scheme_t();

    // classic open addressing and robin hood flavoured open addressing
    typedef OpenAddressingMultiHashMap<uint64_t, bits_key, bits_val,
                                       decltype(hash_func),
                                       decltype(prob_func)>  oa_hash_t;
    typedef RobinHoodMultiHashMap<uint64_t, bits_key, bits_val, bits_cnt,
                                  decltype(hash_func),
                                  decltype(prob_func)> rh_hash_t;

    // init both data structures
    TIMERSTART(init_OA)
    oa_hash_t OAHash(capacity, hash_func, prob_func);
    TIMERSTOP(init_OA)

    TIMERSTART(init_RH)
    rh_hash_t RHHash(capacity, hash_func, prob_func);
    TIMERSTOP(init_RH)

    TIMERSTART(init_keys)
    uint32_t * keys = new uint32_t[max_elements/vals_per_key];

    # pragma omp parallel for
    for (uint64_t i = 0; i < max_elements/vals_per_key; i++)
        keys[i] = i;

    std::mt19937 urng(seed);
    std::shuffle(keys, keys+max_elements/vals_per_key, urng);
    TIMERSTOP(init_keys)

    TIMERSTART(insert_OA)
    size_t insert_error_OA = 0;
    # pragma omp parallel for reduction(+:insert_error_OA)
    for (uint64_t i = 0; i < max_elements/vals_per_key; i++)
        for (uint64_t j = 0; j < keys[i] % vals_per_key; j++)
            insert_error_OA += !OAHash.add(keys[i], j);
    TIMERSTOP(insert_OA)

    TIMERSTART(insert_RH)
    size_t insert_error_RH = 0;
    # pragma omp parallel for reduction(+:insert_error_RH)
    for (uint64_t i = 0; i < max_elements/vals_per_key; i++)
        for (uint64_t j = 0; j < keys[i] % vals_per_key; j++)
            insert_error_RH += !RHHash.add(keys[i], j);
    TIMERSTOP(insert_RH)

    TIMERSTART(query_OA)
    size_t query_error_OA = 0;
    # pragma omp parallel for reduction(+:query_error_OA)
    for (uint64_t i = 0; i < max_elements/vals_per_key; i++) {
        auto result = OAHash.get(keys[i]);
        std::sort(result.begin(), result.end());
        bool local_error = result.size() != (keys[i] % vals_per_key);
        for (uint64_t j = 0; j < result.size(); j++)
            local_error |= result[j] != j;
        query_error_OA += local_error;
    }
    TIMERSTOP(query_OA)

    TIMERSTART(query_RH)
    size_t query_error_RH = 0;
    //# pragma omp parallel for reduction(+:query_error_RH)
    for (uint64_t i = 0; i < max_elements/vals_per_key; i++) {
        auto result = RHHash.get(keys[i]);
        std::sort(result.begin(), result.end());
        bool local_error = result.size() != (keys[i] % vals_per_key);
        for (uint64_t j = 0; j < result.size(); j++)
            local_error |= result[j] != j;
        query_error_RH += local_error;
    }
    TIMERSTOP(query_RH)

    std::cout << "insert errors " << insert_error_OA << ", "
              << insert_error_RH << std::endl;
    std::cout << "query errors " << query_error_OA <<  ", "
              << query_error_RH << std::endl;
    std::cout << "loads " << double(OAHash.size)/OAHash.capacity
              << ", " << double(RHHash.size)/RHHash.capacity << std::endl;

    delete [] keys;
}
