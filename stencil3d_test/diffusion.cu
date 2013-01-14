#include <iostream>
#include <vector>
#include "../util/CUDAEventTimer.h"
#include "../util/compute_blocks.h"
#include "../util/do_all_3d.h"
#include "../util/stencil.h"
#include "../util/compute.h"

typedef double REAL_T;
#include "../util/stencils.h"   

int main(int argc, char** argv) {
    cudaDeviceReset();
    if(argc < 8) {
        std::cout << "usage: " << argv[0]
                  << " width height depth <threads per block x y z> nsteps "
                     "[iteration axis]" << std::endl;
        return 1;
    }
    char axis = 0;
    //if axis is set then the launch configuration is done on a 2d slice
    //and the kernel performs the iteration over the specified axis
    if(argc == 9 ) {
        axis = argv[8][0];
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
    const int nsteps = atoi(argv[7]);
    const size_t size = width * height * depth;
    const size_t byte_size = size * sizeof(REAL_T);
    std::vector< REAL_T > h_data(size, 1);

    REAL_T* d_data_in = 0;
    REAL_T* d_data_out = 0;
    cudaMalloc(&d_data_in, byte_size);
    cudaMalloc(&d_data_out, byte_size);

    cudaMemcpy(d_data_in, &h_data[0], byte_size, cudaMemcpyHostToDevice);
    
    const dim3 threads_per_block = 
        dim3(threads_per_block_x, threads_per_block_y, threads_per_block_z);

   
    const dim3 offset(ioffset, joffset, koffset);
    const dim3 global_grid_size(width, height, depth);
    
    //compute block size
    dim3 blocks;
    if(axis == 0 ) {
        //launch on core space only    
        blocks = compute_blocks(dim3(width - 2 * ioffset,
                                     height - 2 * joffset,
                                     depth - 2 * koffset),
                                threads_per_block);
        
    } else if(axis == 'x') {
       //launch on core space only    
       blocks = compute_blocks(dim3(height - 2 * joffset,
                                    depth - 2 * koffset),
                               threads_per_block);
    } else if(axis == 'y') {
       //launch on core space only    
       blocks = compute_blocks(dim3(width - 2 * ioffset,
                                    depth - 2 * koffset),
                               threads_per_block);
    } else if(axis == 'z') {
       //launch on core space only    
       dim3 blocks = compute_blocks(dim3(width - 2 * ioffset,
                                         height - 2 * joffset),
                                    threads_per_block);
    }
    do_all_3d_1_gpu<<<blocks, threads_per_block>>>
        (d_data_in, offset, global_grid_size, init(REAL_T(0)));
    CUDAEventTimer timer;
    timer.start();
    //compute
    compute
           (nsteps, d_data_in, d_data_out, offset,
            global_grid_size, blocks, threads_per_block, diffusion_3d(),
            do_all_3d_2_gpu<REAL_T, diffusion_3d> );
    timer.stop();
    std::cout << timer.elapsed() << std::endl;
    //copy data back
    cudaMemcpy(&h_data[0], d_data_out, byte_size, cudaMemcpyDeviceToHost);
    //free resources
    cudaFree(d_data_out);
    cudaFree(d_data_in);
    return 0;
}



