#include <iostream>
#include "include/data_types.cuh"
#include "include/hpc_helpers.cuh"



int main () {

    cudaSetDevice(0);

    uint64_t capacity = 1UL << 28;
    typedef KeyValuePair_t<uint64_t, 32, 32> entry_t;

    TIMERSTART(malloc_data)
    entry_t * Data = nullptr;
    cudaMalloc(&Data, sizeof(entry_t)*capacity);                          CUERR
    TIMERSTOP(malloc_data)


}
