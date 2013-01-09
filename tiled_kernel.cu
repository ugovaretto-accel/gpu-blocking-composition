
//0 load data into input buffer

//1 copy data into shared memory

//2 perform computation on shared memory data and copy into
//  output buffer

//3 download from output buffer


typedef float REAL_T;

__device__ REAL_T read_value( REAL_T* center, int width, int i, int j ) {
    return *(center + i + j * width);
}


__device__ REAL_T stencil_op(REAL_T* center, int width) {
    return   - 4 * read_value(center, 0, 0) 
             + read_value(center, width, 0, 1)
             - read_value(center, width, 0, -1)
             + read_value(center, width, -1, 0)
             - read_value(center, width, 1, 0));
}

const int BLOCK_WIDTH = 16;
const int BLOCK_HEIGHT = 16;
const int CACHE_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT;

__global__ tile_apply( REAL_T* in, REAL_T* out, int stencil_offset ) {

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
    //apply stencil operator
    out[global_out_idx] = stencil_op(&cache[cache_idx], BLOCK_WIDTH); 
}

