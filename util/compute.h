#pragma once
#include <algorithm>
#include <cuda_runtime.h>

template < typename KernelT, typename T, typename FunT >
void compute(int nsteps,
             T* d_data_in,
             T* d_data_out,
             dim3 offset,
             dim3 global_grid_size,
             dim3 blocks,
             dim3 threads_per_block 
             FunT operation) {

    for(int step = 0; step != steps; ++step) {  
              KernelT<<<blocks, threads_per_block>>>
                     <T,FunT>(d_data_in,
                              d_data_out,
                              offset,
                              global_grid_size,                                          
                              operation);
        cudaDeviceSynchronize();
        std::swap(d_data_in, d_data_out);
    }

    if(nsteps % 2 == 0 ) std::swap(d_data_in, d_data_out);
}

