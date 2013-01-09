#include <iostream>
#include <vector>
//#include <cstdio> - uncomment for printf in kernels
//#include <cuda_runtime.h>
//cuda is included automatically when compiling with nvcc

typedef double REAL_T;

//-----------------------------------------------------------------------------
class CUDAEventTimer {
public:
  CUDAEventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CUDAEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t stream = 0) { 
    stream_ = stream;
    cudaEventRecord(start_, stream_);
  } 
  void stop() { 
    cudaEventRecord(stop_, stream_); 
    cudaEventSynchronize(stop_);
  }
  float elapsed() {
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }
private:
  cudaEvent_t start_, stop_;
  cudaStream_t stream_;
};


int compute_blocks(int length, int threads_per_block) {
    //integer division:
    //if length is evenly divisable by the number of threads
    //is equivalent to length / threads_per_block, if not
    //it is equivalent to length / threads_per_block  + 1
    return (length + threads_per_block - 1) / threads_per_block;
}

dim3 compute_blocks(int xsize, int ysize, int zsize,
                    int threads_per_block_x,
                    int threads_per_block_y,
                    int threads_per_block_z) {
    return dim3(compute_blocks(xsize, threads_per_block_x),
                compute_blocks(ysize, threads_per_block_y),
                compute_blocks(zsize, threads_per_block_z));
}


//-----------------------------------------------------------------------------
__device__  REAL_T cell_op( REAL_T v) {
    return cos(v) * exp(v);
}

//-----------------------------------------------------------------------------
__global__ void cuda_kernel(REAL_T* grid,
                            dim3 size, 
                            int x_offset,
                            int y_offset,
                            int z_offset) {
   
    const int i = blockIdx.x * blockDim.x + threadIdx.x;// + x_offset;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;// + y_offset;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;// + z_offset;
    if( i >= size.x || j >= size.y || k >= size.z ) return;
    const int index = i + size.x * (j + size.y * k); 
    grid[ index ] = cell_op( grid[ index ] );

}

typedef long long int CYCLES;
__global__ void cuda_kernel_cycles(CYCLES cycles){
    const CYCLES start = clock64();
    while( (clock64() - start) < cycles );
}


//-----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "usage: " << argv[0] 
                  << " xsize ysize zsize threads_per_block [kernel duration(ms)]\n";
        return 1;
    }
    const int XSIZE = atoi(argv[1]);
    const int YSIZE = atoi(argv[2]);
    const int ZSIZE = atoi(argv[3]);
    const int CUDA_THREADS_PER_BLOCK = atoi(argv[4]);
    const size_t TOTAL_SIZE = XSIZE * YSIZE * ZSIZE;
    const size_t TOTAL_BYTE_SIZE = TOTAL_SIZE * sizeof(REAL_T);

    bool use_cycles = false;
    int time_ms = 0;
    CYCLES cycles = 0;
    if( argc > 5 ) {
        time_ms = atoi(argv[5]);
        use_cycles = true;
    }


    // get clock rate in kHz
    cudaDeviceProp props;
    if( cudaGetDeviceProperties(&props, 0) != cudaSuccess ) return -1;
    const unsigned int CLOCK_RATE_Hz = props.clockRate * 1000;
    std::cout << "Clock rate (GHz): " 
              << CLOCK_RATE_Hz / double(1024 * 1024 * 1024)
              << std::endl;
    cycles = CLOCK_RATE_Hz * (time_ms / 1000.0);

    // 3D grid setup
    std::vector<  REAL_T > h_grid(TOTAL_SIZE, REAL_T(0));
    REAL_T* d_grid = 0;
    if( cudaMalloc(&d_grid, 2*TOTAL_BYTE_SIZE) != cudaSuccess ) return -2;
    if( cudaMemcpy(d_grid, &h_grid[0], TOTAL_BYTE_SIZE, cudaMemcpyHostToDevice)
        != cudaSuccess ) return -3;

    // launch configuration
    const dim3 CUDA_THREADS = dim3(CUDA_THREADS_PER_BLOCK,
                                   CUDA_THREADS_PER_BLOCK,
                                   CUDA_THREADS_PER_BLOCK);
    const dim3 CUDA_BLOCKS = compute_blocks(XSIZE, YSIZE, ZSIZE,
                                            CUDA_THREADS_PER_BLOCK,
                                            CUDA_THREADS_PER_BLOCK,
                                            CUDA_THREADS_PER_BLOCK); 
    int x_offset = 0;
    int y_offset = 0;
    int z_offset = 0;
    const dim3 GRID_SIZE = dim3(XSIZE, YSIZE, ZSIZE);
    cudaDeviceSynchronize();
    // launch one kernel encompassing the entire grid...
    std::cout << "Launching kernel:\n" 
              << "  Grid size: "
              << GRID_SIZE.x << ", " << GRID_SIZE.y << ", " << GRID_SIZE.z << std::endl
              << "  Block size: "
              << CUDA_BLOCKS.x << ", " << CUDA_BLOCKS.y << ", " << CUDA_BLOCKS.z << std::endl;  
    CUDAEventTimer timer;
    timer.start();
    if( use_cycles ) {
        cuda_kernel_cycles<<< CUDA_BLOCKS, CUDA_THREADS >>>(cycles);
    } else {
        cuda_kernel<<< CUDA_BLOCKS, CUDA_THREADS >>>(d_grid,
                                                     GRID_SIZE,  
                                                     x_offset,
                                                     y_offset,
                                                     z_offset);
    }

    timer.stop();
    const float single_elapsed = timer.elapsed();
    std::cout << "Single kernel launch: " << single_elapsed << std::endl;
                                        
    // ...and multiple time the same kernel on the same grid
    std::cout << "Launching kernel:\n" 
              << "  Grid size: "
              << GRID_SIZE.x << ", " << GRID_SIZE.y << ", " << GRID_SIZE.z << std::endl
              << "  Block size: 1, 1, 1" << std::endl;
    cudaDeviceSynchronize();  
    timer.start();
    for( int k = 0; k != CUDA_BLOCKS.z; ++k ) {
        z_offset = k * CUDA_THREADS.z; 
        for( int j = 0; j != CUDA_BLOCKS.y; ++j ) {
            y_offset = j * CUDA_THREADS.y;
            for( int i = 0; i != CUDA_BLOCKS.x; ++i ) {
                x_offset = i * CUDA_THREADS.x;
                if( use_cycles ) {
                    cuda_kernel_cycles<<< 1, CUDA_THREADS >>>(cycles);
                } else {
                    cuda_kernel<<< 1, CUDA_THREADS >>>(d_grid, GRID_SIZE,
                                                       x_offset, y_offset, z_offset);

                }
            }
        }
    }
    timer.stop();
    const float multiple_elapsed = timer.elapsed();
    std::cout << "Multiple kernel launches: " << multiple_elapsed << std::endl;

    std::cout << "Multiple/Single %: " << 100 * multiple_elapsed / single_elapsed << std::endl;

    // cleanup
    cudaFree(d_grid);
    return 0;
}