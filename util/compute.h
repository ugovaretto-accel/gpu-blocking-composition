#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_error.h"


//T: data type
//FunT: stencil operator applied to every grid element
//KernelT: cuda kernel or cpu 'apply' function - applies
//         the FunT functor to every element in the grid


template < typename T, typename FunT, typename KernelT >
void cuda_compute(int nsteps,
                  T*& d_data_in,
                  T*& d_data_out,
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

#ifdef ENABLE_SURFACE
template < typename FunT, typename KernelT >
void cuda_compute(int nsteps,
                  cudaArray* d_data_in,
                  cudaArray* d_data_out,
                  dim3 offset,
                  dim3 global_grid_size,
                  dim3 blocks,
                  dim3 threads_per_block, 
                  FunT operation,
                  KernelT kernel) {
    
    for(int step = 0; step != nsteps; ++step) {
        if(step % 2 == 0) {
            CHECK_CUDA(cudaBindSurfaceToArray(in_surface, d_data_in));
            CHECK_CUDA(cudaBindSurfaceToArray(out_surface, d_data_out));
        } else {
            CHECK_CUDA(cudaBindSurfaceToArray(out_surface, d_data_in));
            CHECK_CUDA(cudaBindSurfaceToArray(in_surface, d_data_out));
        } 
        kernel<<<blocks, threads_per_block>>>
               (offset,
                global_grid_size,                                          
                operation);
    }
}
template < typename FunT, typename KernelT >
void cuda_compute_tex(int nsteps,
                      cudaArray* d_data_in,
                      cudaArray* d_data_out,
                      dim3 offset,
                      dim3 global_grid_size,
                      dim3 blocks,
                      dim3 threads_per_block, 
                      FunT operation,
                      KernelT kernel) {
    
    for(int step = 0; step != nsteps; ++step) {
        if(step % 2 == 0) {
            CHECK_CUDA(cudaBindTextureToArray(in_texture, d_data_in);
            CHECK_CUDA(cudaBindSurfaceToArray(in_surface, d_data_in));
            CHECK_CUDA(cudaBindSurfaceToArray(out_surface, d_data_out));
        } else {
            CHECK_CUDA(cudaBindTextureToArray(in_texture, d_data_out); 
            CHECK_CUDA(cudaBindSurfaceToArray(out_surface, d_data_in));
            CHECK_CUDA(cudaBindSurfaceToArray(in_surface, d_data_out));
        } 
        kernel<<<blocks, threads_per_block>>>
               (offset,
                global_grid_size,                                          
                operation);
    }
}
#endif

template < typename T, typename FunT, typename KernelT >
T* cpu_compute(int nsteps,
                 T* data_in,
                 T* data_out,
                 dim3 offset,
                 dim3 global_grid_size,
                 FunT operation,
                 KernelT kernel) {

    for(int step = 0; step != nsteps; ++step) {  
              kernel(data_in,
                     data_out,
                     offset,
                     global_grid_size,                                          
                     operation);
        std::swap(data_in, data_out);
    }

    return nsteps % 2 != 0 ? data_in : data_out;
}