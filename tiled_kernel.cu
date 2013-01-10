#include <iostream>
#include <vector>
#include "CUDAEventTimer.h"


typedef float REAL_T;

__device__ REAL_T read_global_value(REAL_T* center, int i, int j) {
    if( i < 0 ) i += gridDim.x;
    else( i >= gridDim.x ) i -= gridDim.x;
    if( j < 0 ) j += gridDim.y;
    else( j >= gridDim.y ) j -= gridDim.y;   
    return *(center + i + j * gridDim.x);
}

__device__ REAL_T read_value(REAL_T* center, int width, int i, int j) {
    return *(center + i + j * width);
}


__device__ REAL_T stencil_op(REAL_T* center, int width) {
    return   - 4 * read_value(center, 0, 0) 
             + read_value(center, width, 0, 1)
             - read_value(center, width, 0, -1)
             + read_value(center, width, -1, 0)
             - read_value(center, width, 1, 0));
}

__global__ void apply(REAL_T* in, REAL_T* out, int stencil_offset) {
    //periodic boundary condition
    int in_i = blockIdx.x * blockDim.x + threadIdx.x;
    int in_j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = in_i + in_j * gridDim.x;
    out[idx] = stencil_op(&in[idx], gridDim.x); 
}


const int BLOCK_WIDTH = 16;
const int BLOCK_HEIGHT = 16;
const int CACHE_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT;

__global__ void tile_apply(REAL_T* in, REAL_T* out, int stencil_offset) {

    __shared cache[CACHE_SIZE];

    //block size = size of core space + ghost cells
    //stencil_offset = stencil width / 2
    const int ioff = blockIdx.x * blockDim.x;
    const int joff = blockIdx.y * blockDim.y;
    const dim3 core_grid_dim(gridDim.x - stencil_offset, gridDim.y - stencil_offset);   
    const dim3 core_block_dim(blockDim.x - stencil_offset,
                              blockDim.y - stencil_offset);
    //periodic boundary condition
    int in_i = ioff + threadIdx.x - stencil_offset;
    if(in_i < 0) in_i = core_grid_dim.x + in_i;
    else if( in_i >= core_grid_dim.x ) in_i = in_i - core_grid_dim.x;
    int in_j = joff + trheadIdx.y - stencil_offset;
    if( in_j < 0 ) in_j = core_grid_dim.y + in_j;
    else if( in_j >= core_grid_dim.y ) in_j = in_j - core_grid_dim.y;
    const int global_in_idx = in_i + in_j * gridDim.x * blockDim.x;
    const int cache_idx = threadIdx.x + blockDim.x * threadIdx.y;
    const int global_out_idx = threadIdx.x - stencil_offset + blockIdx.x * core_block_dim.x
                               + (threadIdx.y - stencil_offset + blockIdx.y * core_block_dim.y)
                               * blockDim.x;
                               
    //copy data into shared memory
    cache[cache_idx] = in[global_in_idx];   
    __syncthreads(); 
    //apply stencil operator to core space
    if( threadIdx.x < stencil_offset 
        || threadIdx.x >= (core_block_dim.x + stencil_offset)
        || threadIdx.y < stencil_offset
        || threadIdx.y >= (core_block_dim.y + stencil_offset) ) return;
    out[global_out_idx] = stencil_op(&cache[cache_idx], BLOCK_WIDTH); 
}


int main(int, char**) {

    int width = 1024;
    int height = 1024;
    size_t size = width * height;
    size_t byte_size = size * sizeof(REAL_T);
    std::vector< REAL_T > h_data(size, 0);

    REAL_T* d_data_in = 0;
    REAL_T* d_data_out = 0;
    cudaMalloc(&d_data_in, byte_size);
    cudaMalloc(&d_data_out, byte_size);

    cudaMemcpy(d_data, h_data, cudaMemcpyHostToDevice);
    const int stencil_offset = 1;
    tile_apply<<<blocks, threads_per_block>>>(d_data_in, d_data_out, stencil_offset);

    cudaMemcpy(h_data, d_data_out, byte_size, cudaMemcpyDeviceToHost);

    cuddFree(d_data_out);
    cudaFree(d_data_in);

    cudaDeviceReset();
    return 0;

}



