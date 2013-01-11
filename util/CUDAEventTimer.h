#pragma once
#include <cuda_runtime_api.h>

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