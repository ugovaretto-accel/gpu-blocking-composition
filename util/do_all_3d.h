#pragma once
#include <cuda_runtime.h>


//launch with size of core space
template < typename T, typename FunT >
__global__ void do_all_3d_1_gpu(T* grid,
	                            dim3 offset,
	                            dim3 global_grid_size, //core space + 2 * offset
	                            FunT f ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z + offset.z;
    const int idx = x + global_grid_size.x * (y  + z * global_grid_size.y);
    grid[idx] = f(grid, idx, global_grid_size);
}


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


//launch with size of core space

//Perform operation and computation of output location through
//index mapping.
//
//sample index mapping function to perform scaled copies
//out_coord = out_offset 
//            + in_coord 
//            * (out_global_grid_size - 2*outoffset)
//               / (out_global_grid_size - 2*inoffset)
template < typename T, typename FunT, typename MapIdxFunT >
__global__ void do_all_3d_2_gpu(const T* in,
	                            dim3 in_offset,
/*core space + 2 * offset*/     dim3 in_global_grid_size, 
                                const T* out,
	                            dim3 out_offset,
/*core space + 2 * offset*/     dim3 out_global_grid_size,
	                            FunT f,
                                MapIdxFunT map ) {
    int x = blockDim.x * blockIdx.x + threadIdx.x + in_offset.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y + in_offset.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z + in_offset.z;
    const int in_idx = x + in_global_grid_size.x 
                           * (y  + z * in_global_grid_size.y);
    dim3 out_coords = map(x, y, z, in_offset, in_global_grid_size, 
                          out_offset, out_global_grid_size);
    const int out_idx = out_coords.x
                        + out_global_grid_size.x
                        * (out_coords.y  + out_coords.z 
                                           * out_global_grid_size.y);  
    out[out_idx] = f(in, in_idx, in_global_grid_size);
}


//launch with size of 2d core space, iterates on z internally
template < typename T, typename FunT > 
__global__ void do_all_3d_2_z_gpu(const T* in,
	                              T* out,
	                              dim3 offset,
/*core space + 2 * offset*/       dim3 global_grid_size, 
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
/*core space + 2 * offset*/       dim3 global_grid_size,
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
/*core space + 2 * offset*/       dim3 global_grid_size,
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
