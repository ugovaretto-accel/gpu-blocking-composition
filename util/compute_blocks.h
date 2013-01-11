#pragma once

#include <cuda_runtime.h>


inline int compute_blocks(int length, int threads_per_block) {
    if(threads_per_block < 1) return 1;
    //integer division:
    //if length is evenly divisable by the number of threads
    //is equivalent to length / threads_per_block, if not
    //it is equivalent to length / threads_per_block  + 1
    return (length + threads_per_block - 1) / threads_per_block;
}

inline dim3 compute_blocks(dim3 grid_size, dim3 threads_per_block) {
    return dim3(compute_blocks(grid_size.x, threads_per_block.x),
                compute_blocks(grid_size.y, threads_per_block.y),
                compute_blocks(grid_size.z, threads_per_block.z));
}
