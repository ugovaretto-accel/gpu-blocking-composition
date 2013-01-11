#pragma once
#include <cuda_runtime.h>

//launch with size of core space
template < typename T, typename FunT >
__global__ void do_all_3d_2_gpu(const T* in,
	                            T* out,
	                            dim3 offset,
	                            dim3 global_grid_size, //core space + 2 * offset
	                            FunT f ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z + offset.z;
    const int idx = x + global_grid_size.x * (y  + z * global_grid_size.y);
    out[idx] = f(in, idx, global_grid_size);
}

//launch with size of 2d core space, iterates on z internally
template < typename T, typename FunT > 
__global__ void do_all_3d_2_z_gpu(const T* in,
	                              T* out,
	                              dim3 offset,
	                              dim3 global_grid_size, //core space + 2 * offset
	                              FunT f) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int slice_stride = global_grid_size.x * global_grid_size.y;
    int idx = x + global_grid_size.x * y; 
    for(int k = 0; k != global_grid_size.z - 2 * offset.z; ++k) {
    	idx +=  k * slice_stride;	
        out[idx] = f(in, idx, global_grid_size);
    }
    
}

//launch with size of 2d core space, iterates on x internally
template < typename T, typename FunT > 
__global__ void do_all_3d_2_x_gpu(const T* in,
	                              T* out,
	                              dim3 offset,
	                              dim3 global_grid_size, //core space + 2 * offset
	                              FunT f) {
    const int y = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int z = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    int idx = global_grid_size.x * ( y + global_grid_size.y * z); 
    for(int i = 0; i != global_grid_size.x - 2 * offset.x; ++i, ++idx) {
         out[idx] = f(in, idx, global_grid_size);
    }
}

//launch with size of 2d core space, iterates on x internally
template < typename T, typename FunT > 
__global__ void do_all_3d_2_y_gpu(const T* in,
	                              T* out,
	                              dim3 offset,
	                              dim3 global_grid_size, //core space + 2 * offset
	                              FunT f) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int z = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    int idx = x + global_grid_size.x * global_grid_size.y * z;
    const int row_stride = global_grid_size.x; 
    for(int i = 0; i != global_grid_size.x - 2 * offset.x; ++i, ++idx) {
        idx += row_stride;
        out[idx] = f(in, idx, global_grid_size);
    }
}
