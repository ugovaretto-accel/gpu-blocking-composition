#include <iostream>
#include <vector>
#include "../util/CUDAEventTimer.h"
#include "../util/compute_blocks.h"
#include "../util/do_all_3d.h"
#include "../util/stencil.h"

typedef float REAL_T;

#ifdef FUNPTR //supported since sm_20
__host__ __device__ 
REAL_T laplacian_3d(const REAL_T* center, dim3 grid_size) {

    return REAL_T(-6) * (*center)
           + gv(center, 1, 0, 0, grid_size) - gv(center, -1, 0, 0, grid_size)
           + gv(center, 0, 1, 0, grid_size) - gv(center, 0, -1, 0, grid_size)
           + gv(center, 0, 0, 1, grid_size) - gv(center, 0, 0, -1, grid_size);

}
#else
struct laplacian_3d {
    __host__ __device__ 
    REAL_T operator()(const REAL_T* center, dim3 grid_size) const{
        return REAL_T(-6) * (*center)
             + gv(center, 1, 0, 0, grid_size) - gv(center, -1, 0, 0, grid_size)
             + gv(center, 0, 1, 0, grid_size) - gv(center, 0, -1, 0, grid_size)
             + gv(center, 0, 0, 1, grid_size) - gv(center, 0, 0, -1, grid_size);

    }
};
#endif


int main(int argc, char** argv) {

    if(argc != 7) {
        std::cout << "usage " << argv[0]
                  << "width height depth <threads per block x y z>" << std::endl;
        return 1;
    }

    const int ioffset = 3;
    const int joffset = 3;
    const int koffset = 1;
    const int width = atoi(argv[1]) + ioffset;
    const int height = atoi(argv[2]) + joffset;
    const int depth = atoi(argv[3]) + koffset;
    const int threads_per_block_x = atoi(argv[4]);
    const int threads_per_block_y = atoi(argv[5]);
    const int threads_per_block_z = atoi(argv[6]);
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
    //launch on core space only    
    const dim3 blocks = compute_blocks(dim3(width - ioffset,
                                            height - joffset,
                                            depth - koffset),
                                       threads_per_block);
    const dim3 offset(ioffset, joffset, koffset);
    const dim3 global_grid_size(width, height, depth);

    do_all_2_gpu<<<blocks, threads_per_block>>>(d_data_in,
                                                d_data_out,
                                                offset,
                                                global_grid_size,
#ifdef FUNPTR                                                
                                                laplacian_3d);
#else           
                                                laplacian_3d());
#endif    

    cudaMemcpy(&h_data[0], d_data_out, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data_out);
    cudaFree(d_data_in);

    cudaDeviceReset();
    return 0;

}



