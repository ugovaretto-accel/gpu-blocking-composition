#pragma once
#include <cuda_runtime.h>

//launch with size of core space
template < typename T, typename FunT >
__global__ void do_all_2_gpu(const T* in,
	                         T* out,
	                         dim3 offset,
	                         dim3 global_grid_size, //core space + 2 * offset
	                         FunT f ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z + offset.z;
    const int idx = x + global_grid_size.x * (y  + z * global_grid_size.y);
    out[idx] = f(&in[idx], global_grid_size);
}