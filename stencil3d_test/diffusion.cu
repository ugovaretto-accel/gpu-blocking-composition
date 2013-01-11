#include <iostream>
#include <vector>
#include "../util/CUDAEventTimer.h"
#include "../util/compute_blocks.h"
#include "../util/do_all_3d.h"
#include "../util/stencil.h"

typedef double REAL_T;

#ifdef FUNPTR //supported since sm_20
__host__ __device__ 
REAL_T laplacian_3d(const REAL_T* grid, int idx, const dim3 grid_size) {

    return REAL_T(-6) * grid[idx]
           + gv(grid, idx, 1, 0, 0, grid_size) - gv(grid, idx, -1, 0, 0, grid_size)
           + gv(grid, idx, 0, 1, 0, grid_size) - gv(grid, idx, 0, -1, 0, grid_size)
           + gv(grid, idx, 0, 0, 1, grid_size) - gv(grid, idx, 0, 0, -1, grid_size);

}

__host__ __device__
REAL_T diffusion_3d(const REAL_T* grid, int idx, dim3 grid_size) {
    return grid[idx] + REAL_T(0.1) * laplacian_3d(grid, idx, grid_size); 
}

#else
struct laplacian_3d {
    __host__ __device__ 
    REAL_T operator()(const REAL_T* grid, int idx, dim3 grid_size) const {
        return REAL_T(-6) * grid[idx]
             + gv(grid, idx, 1, 0, 0, grid_size) - gv(grid, idx, -1, 0, 0, grid_size)
             + gv(grid, idx, 0, 1, 0, grid_size) - gv(grid, idx, 0, -1, 0, grid_size)
             + gv(grid, idx, 0, 0, 1, grid_size) - gv(grid, idx, 0, 0, -1, grid_size);

    }
};
struct diffusion_3d {
    __host__ __device__ 
    REAL_T operator()(const REAL_T* grid, int idx, dim3 grid_size) const {
        return grid[idx] +  REAL_T(0.1) * l3d(grid, idx, grid_size); 

    }
    laplacian_3d l3d;
};
#endif


int main(int argc, char** argv) {
    cudaDeviceReset();
    if(argc < 7) {
        std::cout << "usage: " << argv[0]
                  << " width height depth <threads per block x y z> [iteration axis]"
                  << std::endl;
        return 1;
    }
    char axis = 0;
    //if axis is set then the launch configuration is done on a 2d slice
    //and the kernel performs the iteration over the specified axis
    if(argc == 8 ) {
        axis = argv[7][0];
        if(axis != 'x' && axis != 'y' && axis != 'z') {
            std::cout << "axis must be one of either 'x', 'y', 'z'"
                      << std::endl;
            return 1;
        }
    } 

    //temporary for ease of testing with COSMO
    const int ioffset = 3;
    const int joffset = 3;
    const int koffset = 1;

    const int width = atoi(argv[1]) + 2 * ioffset;
    const int height = atoi(argv[2]) + 2 * joffset;
    const int depth = atoi(argv[3]) + 2 * koffset;
    const int threads_per_block_x = atoi(argv[4]);
    const int threads_per_block_y = atoi(argv[5]);
    //set threads per block in z direction to zero if axis is set
    const int threads_per_block_z = axis == 0 ? atoi(argv[6]) : 0;
    const size_t size = width * height * depth;
    const size_t byte_size = size * sizeof(REAL_T);
    std::vector< REAL_T > h_data(size, 0);

    REAL_T* d_data_in = 0;
    REAL_T* d_data_out = 0;
    cudaMalloc(&d_data_in, byte_size);
    cudaMalloc(&d_data_out, byte_size);

    cudaMemcpy(d_data_in, &h_data[0], byte_size, cudaMemcpyHostToDevice);
    
    const dim3 threads_per_block = 
        dim3(threads_per_block_x, threads_per_block_y, threads_per_block_z);

    CUDAEventTimer et;

    if(axis == 0 ) {
        //launch on core space only    
        const dim3 blocks = compute_blocks(dim3(width - 2 * ioffset,
                                                height - 2 * joffset,
                                                depth - 2 * koffset),
                                           threads_per_block);
        const dim3 offset(ioffset, joffset, koffset);
        const dim3 global_grid_size(width, height, depth);

       
        et.start();
        for(int step = 0; step != steps; ++step) {  
            do_all_3d_2_gpu<<<blocks, threads_per_block>>>(d_data_in,
                                                           d_data_out,
                                                           offset,
                                                           global_grid_size,
        #ifdef FUNPTR                                                
                                                           laplacian_3d);
        #else           
                                                           laplacian_3d());
        #endif
            cudaDeviceSynchronize();
        }         
        et.stop();
        std::cout << et.elapsed() << std::endl;
    } else if(axis == 'x') {
       //launch on core space only    
        const dim3 blocks = compute_blocks(dim3(height - 2 * joffset,
                                                depth - 2 * koffset),
                                           threads_per_block);
        const dim3 offset(ioffset, joffset, koffset);
        const dim3 global_grid_size(width, height, depth);

       
        et.start();
        do_all_3d_2_x_gpu<<<blocks, threads_per_block>>>(d_data_in,
                                                         d_data_out,
                                                         offset,
                                                         global_grid_size,
    #ifdef FUNPTR                                                
                                                         laplacian_3d);
    #else           
                                                         laplacian_3d());
    #endif    
        et.stop();
        std::cout << et.elapsed() << std::endl;
    } else if(axis == 'y') {
       //launch on core space only    
        const dim3 blocks = compute_blocks(dim3(width - 2 * ioffset,
                                                depth - 2 * koffset),
                                           threads_per_block);
        const dim3 offset(ioffset, joffset, koffset);
        const dim3 global_grid_size(width, height, depth);

       
        et.start();
        do_all_3d_2_y_gpu<<<blocks, threads_per_block>>>(d_data_in,
                                                         d_data_out,
                                                         offset,
                                                         global_grid_size,
    #ifdef FUNPTR                                                
                                                         laplacian_3d);
    #else           
                                                         laplacian_3d());
    #endif    
        et.stop();
        std::cout << et.elapsed() << std::endl;
    } else if(axis == 'z') {
       //launch on core space only    
        const dim3 blocks = compute_blocks(dim3(width - 2 * ioffset,
                                                height - 2 * joffset),
                                           threads_per_block);
        const dim3 offset(ioffset, joffset, koffset);
        const dim3 global_grid_size(width, height, depth);

       
        et.start();
        do_all_3d_2_z_gpu<<<blocks, threads_per_block>>>(d_data_in,
                                                         d_data_out,
                                                         offset,
                                                         global_grid_size,
    #ifdef FUNPTR                                                
                                                         laplacian_3d);
    #else           
                                                         laplacian_3d());
    #endif    
        et.stop();
        std::cout << et.elapsed() << std::endl;
    }
    
    cudaMemcpy(&h_data[0], d_data_out, byte_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_data_out);
    cudaFree(d_data_in);
    return 0;

}



