#pragma once
#include <cuda_runtime.h>

template < typename T, typename FunT >
void do_all_3d_1_cpu(T* grid,
                     dim3 offset,
                     dim3 global_grid_size, //core space + 2 * offset
                     FunT f ) {
    const int core_space_width = global_grid_size.x - 2 * offset.x;
    const int core_space_height = global_grid_size.y - 2 * offset.y;
    const int core_space_depth = global_grid_size.z - 2 * offset.z;
    int z = 0;
    int y = 0;
    int idx = 0;
    const int slice_stride = global_grid_size.x * global_grid_size.y;
    const int row_stride = global_grid_size.x;
    for(int k = 0; k != core_space_depth; ++k ) {
        z = slice_stride * (offset.z + k); 
        for(int j = 0; j != core_space_height; ++j) {
            y = row_stride * (offset.y + j);
            for(int i = 0; i != core_space_width; ++i) {
                idx = offset.x + i + y + z;
                grid[idx] = f(grid, 
                              dim3(i + offset.x,
                                   j + offset.y,
                                   k + offset.z),
                              global_grid_size);
            }
        }
    }   
}


template < typename T, typename FunT >
void do_all_3d_2_cpu(const T* in,
                     T* out,
                     dim3 offset,
                     dim3 global_grid_size, //core space + 2 * offset
                     FunT f ) {
    const int core_space_width = global_grid_size.x - 2 * offset.x;
    const int core_space_height = global_grid_size.y - 2 * offset.y;
    const int core_space_depth = global_grid_size.z - 2 * offset.z;
    int z = 0;
    int y = 0;
    int idx = 0;
    const int slice_stride = global_grid_size.x * global_grid_size.y;
    const int row_stride = global_grid_size.x;
    for(int k = 0; k != core_space_depth; ++k ) {
        z = slice_stride * (offset.z + k); 
        for(int j = 0; j != core_space_height; ++j) {
            y = row_stride * (offset.y + j);
            for(int i = 0; i != core_space_width; ++i) {
                idx = offset.x + i + y + z;
                out[idx] = f(in, 
                             dim3(i + offset.x,
                                   j + offset.y,
                                   k + offset.z),
                              global_grid_size);
            }
        }
    }   
}



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
    grid[idx] = f(grid, dim3(x, y, z), global_grid_size);
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
    out[idx] = f(in, dim3(x, y, z), global_grid_size);
}

#ifdef ENABLE_SURFACE
//launch with size of core space
template < typename T, typename FunT >
__global__ void do_all_3d_2_gpu_surf(dim3 offset,
                                     dim3 global_grid_size, //core space + 2 * offset
                                     FunT f ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z + offset.z;
    const T v = f(dim3(x, y, z));
    surf3Dwrite(v, out_surface, x * sizeof(T), y, z);
}
//launch with size of core space
template < typename T, typename FunT >
__global__ void do_all_3d_2_gpu_tex(dim3 offset,
                                    dim3 global_grid_size, //core space + 2 * offset
                                    FunT f ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y + offset.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z + offset.z;
    const T v = f(dim3(x, y, z));
    surf3Dwrite(v, out_surface, x * sizeof(T), y, z);
}
#endif

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
    dim3 out_coords = map(x, y, z, in_offset, in_global_grid_size, 
                          out_offset, out_global_grid_size);
    const int out_idx = out_coords.x
                        + out_global_grid_size.x
                        * (out_coords.y  + out_coords.z 
                                           * out_global_grid_size.y);  
    out[out_idx] = f(in, dim3(x, y, z), in_global_grid_size);
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
    const int xy = x + global_grid_size.x * y;
    const int slice_stride = global_grid_size.x * global_grid_size.y; 
    for(int k = 0; k != global_grid_size.z - 2 * offset.z; ++k) {
    	const int idx =  xy + (k + offset.z) * slice_stride;	
        out[idx] = f(in, dim3(x, y, k + offset.z), global_grid_size);
    }
    
}
//launch with size of 2d core space, iterates on x internally
template < typename T, typename FunT > 
__global__ void do_all_3d_2_x_gpu(const T* in,
                                T* out,
                                dim3 offset,
/*core space + 2 * offset*/     dim3 global_grid_size, 
                                FunT f) {
    const int y = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    const int z = blockDim.y * blockIdx.y + threadIdx.y + offset.z;
    const int yz = global_grid_size.x * y 
                   + global_grid_size.x * global_grid_size.y * z;
    for(int i = 0; i != global_grid_size.x - 2 * offset.x; ++i) {
        const int idx = yz + offset.x + i;      
        out[idx] = f(in, dim3(i + offset.z, y, z), global_grid_size);
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
    const int z = blockDim.y * blockIdx.y + threadIdx.y + offset.z;
    const int xz = x + global_grid_size.x * global_grid_size.y * z; 
    for(int j = 0; j != global_grid_size.y - 2 * offset.y; ++j) {
        const int idx =  xz + (j + offset.y) * global_grid_size.x;    
        out[idx] = f(in, dim3(x, j + offset.y, z), global_grid_size);
    }
    
}
