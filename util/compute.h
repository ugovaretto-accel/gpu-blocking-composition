#pragma once
#include <algorithm>
#include <cuda_runtime.h>

//T: data type
//FunT: stencil operator applied to every grid element
//KernelT: cuda kernel or cpu 'apply' function - applies
//         the FunT functor to every element in the grid


template < typename T, typename FunT, typename KernelT >
void cuda_compute(int nsteps,
                  T* d_data_in,
                  T* d_data_out,
                  dim3 offset,
                  dim3 global_grid_size,
                  dim3 blocks,
                  dim3 threads_per_block, 
                  FunT operation,
                  KernelT kernel) {

    for(int step = 0; step != nsteps; ++step) {  
              kernel<<<blocks, threads_per_block>>>
                     (d_data_in,
                      d_data_out,
                      offset,
                      global_grid_size,                                          
                      operation);
        std::swap(d_data_in, d_data_out);
    }
    if(nsteps % 2 == 0 ) std::swap(d_data_in, d_data_out);
}

template < typename T, typename FunT, typename KernelT >
void cpu_compute(int nsteps,
                 T* d_data_in,
                 T* d_data_out,
                 dim3 offset,
                 dim3 global_grid_size,
                 FunT operation,
                 KernelT kernel) {

    for(int step = 0; step != nsteps; ++step) {  
              kernel(d_data_in,
                     d_data_out,
                     offset,
                     global_grid_size,                                          
                     operation);
        std::swap(d_data_in, d_data_out);
    }

    if(nsteps % 2 == 0 ) std::swap(d_data_in, d_data_out);
}