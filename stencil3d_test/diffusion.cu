#include <iostream>
#include <vector>
#include "../util/CUDAEventTimer.h"
#include "../util/Timer.h"
#include "../util/compute_blocks.h"
#include "../util/do_all_3d.h"
#include "../util/compute.h"
#include "../util/stencils.h"
#include "../util/cuda_error.h"   

template <typename T>
struct distance {
    distance(T eps = T()) : epsilon(eps) {}
    bool operator()(const T& v1, const T& v2) const {
        return std::abs(v1 - v2) <= epsilon;
    }
    T epsilon;
};

std::ostream& operator<<(std::ostream& os, const dim3 d) {
    os << d.x << ' ' << d.y << ' ' << d.z;
    return os;
}



typedef double REAL_T;

REAL_T EPS = REAL_T(0.000001);

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
    const int threads_per_block_z = axis == 0 ? atoi(argv[6]) : 1;
    const int nsteps = atoi(argv[7]);
    const size_t size = width * height * depth;
    const size_t byte_size = size * sizeof(REAL_T);
    

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int total_threads = threads_per_block_x 
                              * threads_per_block_y 
                              * threads_per_block_z;                    
    if(prop.maxThreadsPerBlock < total_threads) {
        std::cout << "ERROR: max threads per block count("
                  << prop.maxThreadsPerBlock << ") exceeded("
                  << total_threads << ")" << std::endl;
        return -1;
    }

    std::vector< REAL_T > h_data(size, 1);
    std::vector< REAL_T > h_data_in(size, 1);
    std::vector< REAL_T > h_data_out(size, 1);

    REAL_T* d_data_in = 0;
    REAL_T* d_data_out = 0;
    CHECK_CUDA(cudaMalloc(&d_data_in, byte_size));
    CHECK_CUDA(cudaMalloc(&d_data_out, byte_size));
   
  
    const dim3 threads_per_block = 
        dim3(threads_per_block_x, threads_per_block_y, threads_per_block_z);
    const dim3 offset(ioffset, joffset, koffset);
    const dim3 global_grid_size(width, height, depth);
    
    //compute block size: cover the cases of 2d GPU grid
    //with explicit in-kernel iteration over 3rd dimensions.
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
                                    depth - 2 * koffset,
                                    1 ),
                               threads_per_block);
    } else if(axis == 'y') {
       //launch on core space only    
       blocks = compute_blocks(dim3(width - 2 * ioffset,
                                    depth - 2 * koffset,
                                    1),
                               threads_per_block);
    } else if(axis == 'z') {
       //launch on core space only    
       blocks = compute_blocks(dim3(width - 2 * ioffset,
                                    height - 2 * joffset,
                                    1),
                                    threads_per_block);
    }
    //from here on all computation is done on the core region only
    //i.e. global grid - halo region

    //fill inner(core) region with zeros
    //note that it is required to re-compute the number of
    //threads per block because the actual threads per block
    //used varies if 2d+in-kernel-iteration is selected
    //just copy from host memory for now 
    //do_all_3d_1_gpu<<<blocks, threads_per_block>>>
    //    (d_data_in, offset, global_grid_size, init<REAL_T>(REAL_T(0)));

    do_all_3d_1_cpu(&h_data_in[0], 
                    offset,
                    global_grid_size,
                    init<REAL_T>(REAL_T(0)));
    CHECK_CUDA(cudaMemcpy(d_data_in, 
                          &h_data_in[0],
                          byte_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_data_out,
                          &h_data_in[0],
                          byte_size,
                          cudaMemcpyHostToDevice));

    //GPU                     
    CUDAEventTimer gpu_timer;
    gpu_timer.start();
    //compute
    if(axis == 0)
        cuda_compute
               (nsteps, d_data_in, d_data_out, offset,
                global_grid_size, blocks, threads_per_block, diffusion_3d(),
                do_all_3d_2_gpu<REAL_T, diffusion_3d>);
    else if(axis == 'x')
        cuda_compute
               (nsteps, d_data_in, d_data_out, offset,
                global_grid_size, blocks, threads_per_block, diffusion_3d(),
                do_all_3d_2_x_gpu<REAL_T, diffusion_3d>);
    else if(axis == 'y')
        cuda_compute
               (nsteps, d_data_in, d_data_out, offset,
                global_grid_size, blocks, threads_per_block, diffusion_3d(),
                do_all_3d_2_y_gpu<REAL_T, diffusion_3d>);
    else if(axis == 'z')
        cuda_compute
               (nsteps, d_data_in, d_data_out, offset,
                global_grid_size, blocks, threads_per_block, diffusion_3d(),
                do_all_3d_2_z_gpu<REAL_T, diffusion_3d>);                                                                
    gpu_timer.stop();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Launch config: " 
                  << " blocks: " << blocks << std::endl
                  << " threads per block: " << threads_per_block << std::endl
                  << " grid: " << global_grid_size << std::endl; 
        return -1;
    }       
    std::cout << "GPU: " << gpu_timer.elapsed() << std::endl;
    
    //CPU
    Timer cpu_timer;
    cpu_timer.Start();
    cpu_compute
           (nsteps, &h_data_in[0], &h_data_out[0], offset,
            global_grid_size, diffusion_3d(),
            do_all_3d_2_cpu<REAL_T, diffusion_3d>);
    const double ms = cpu_timer.Stop();
    std::cout << "CPU: " << ms << std::endl;
    
    //copy data back
    cudaMemcpy(&h_data[0], d_data_out, byte_size, cudaMemcpyDeviceToHost);

    //compare results: h_data holds the data transferred from the GPU
    //                 h_data_out holds the data computed on the CPU  
    std::cout << "Valid: " << std::boolalpha
              << std::equal(h_data.begin(), h_data.end(),
                            h_data_out.begin(), distance< REAL_T >(EPS))
              << std::endl;
#if 0
    //print something out
               
    do_all_3d_1_cpu(&h_data[0], 
                    offset,
                    global_grid_size,
                    print<REAL_T>());      
    std::cout << "\n=========================================\n";    
    do_all_3d_1_cpu(&h_data_out[0], 
                    offset,
                    global_grid_size,
                    print<REAL_T>());      
    std::cout << std::endl;
#endif     
    //free resources
    cudaFree(d_data_out);
    cudaFree(d_data_in);
    return 0;
}



