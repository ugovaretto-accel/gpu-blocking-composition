#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

inline void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}  

inline void CHECK_CUDA() {
	CHECK_CUDA(cudaGetLastError());
}