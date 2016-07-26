#include <iostream>

#include "include/hpc_helpers.cuh"
#include "include/hash_functions.cuh"
#include "include/robin_hood_multi_hash_map.cuh"
#include "include/open_addressing_multi_hash_map.cuh"

int main (int argc, char * argv[]) {

    // configure experiment
    constexpr uint64_t seed = 42;
    constexpr uint64_t num_elements = 1UL << 28;

    // configure hash map
    constexpr double load = 0.80;
    constexpr uint64_t capacity = num_elements/load;
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

    // just used as input generator during experiments
    auto insert_hash = mueller_hash_uint32_t();
    auto random_hash = nvidia_hash_uint32_t();

    // init both data structures
    TIMERSTART(init_OA)
    oa_hash_t OAHash(capacity, hash_func, prob_func);
    TIMERSTOP(init_OA)

    TIMERSTART(init_RH)
    rh_hash_t RHHash(capacity, hash_func, prob_func);
    TIMERSTOP(init_RH)

    // fill both data structures
    TIMERSTART(fill_OA)
    size_t insert_error_OA = 0;
    # pragma omp parallel for reduction (+:insert_error_OA) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = insert_hash(seed+insert_hash(i)) % ((1UL << bits_key)-1);
        uint64_t val = i % ((1UL << bits_val)-1);
        insert_error_OA += !OAHash.add(key, val);
    }
    TIMERSTOP(fill_OA)

    TIMERSTART(fill_RH)
    size_t insert_error_RH = 0;
    # pragma omp parallel for reduction (+:insert_error_RH) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = insert_hash(seed+insert_hash(i)) % ((1UL << bits_key)-1);
        uint64_t val = i % ((1UL << bits_val)-1);
        insert_error_RH += !RHHash.add(key, val);
    }
    TIMERSTOP(fill_RH)

    // query both data structures
    TIMERSTART(query_content_OA)
    size_t query_error_OA = 0;
    #pragma omp parallel for reduction (+:query_error_OA) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = insert_hash(seed+insert_hash(i)) % ((1UL << bits_key)-1);
        query_error_OA += OAHash.get(key).size() == 0;
    }
    TIMERSTOP(query_content_OA)

    TIMERSTART(query_content_RH)
    size_t query_error_RH = 0;
    #pragma omp parallel for reduction (+:query_error_RH) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = insert_hash(seed+insert_hash(i)) % ((1UL << bits_key)-1);
        query_error_RH += RHHash.get(key).size() == 0;
    }
    TIMERSTOP(query_content_RH)

    TIMERSTART(query_random_OA)
    size_t hits_OA = 0;
    #pragma omp parallel for reduction (+:hits_OA) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = random_hash(seed+random_hash(i)) % ((1UL << bits_key)-1);
        hits_OA += OAHash.get(key).size() > 0;
    }
    TIMERSTOP(query_random_OA)

    TIMERSTART(query_random_RH)
    size_t hits_RH = 0;
    #pragma omp parallel for reduction (+:hits_RH) schedule(dynamic)
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t key = random_hash(seed+random_hash(i)) % ((1UL << bits_key)-1);
        hits_RH += RHHash.get(key).size() > 0;
    }
    TIMERSTOP(query_random_RH)

    // print stats
    std::cout << "insert errors " << insert_error_OA << ", "
              << insert_error_RH <<std::endl;
    std::cout << "query errors  " <<  query_error_OA << ", "
              << query_error_RH << std::endl;
    std::cout << "hit rates random " << double(hits_OA)/double(num_elements)
              << ", " <<  double(hits_RH)/double(num_elements) << std::endl;
    std::cout << "loads " << double(OAHash.size)/OAHash.capacity
              << ", " << double(RHHash.size)/RHHash.capacity << std::endl;
}
