#pragma once

#include <cuda_runtime.h>

struct laplacian_3d {
    template < typename T > 
    __host__ __device__
    T operator()(const T* grid, int idx, dim3 grid_size) const {
        return T(-6) * grid[idx]
             + gv(grid, idx, 1, 0, 0, grid_size)
             - gv(grid, idx, -1, 0, 0, grid_size)
             + gv(grid, idx, 0, 1, 0, grid_size)
             - gv(grid, idx, 0, -1, 0, grid_size)
             + gv(grid, idx, 0, 0, 1, grid_size)
             - gv(grid, idx, 0, 0, -1, grid_size);

    }
};
struct diffusion_3d {
    template < typename T >  
    __host__ __device__
    T operator()(const T* grid, int idx, dim3 grid_size) const {
        return grid[idx] +  T(0.1) * l3d(grid, idx, grid_size); 

    }
    laplacian_3d l3d;
};

template < typename T >
struct init {
    init(T v=T()) : value(v) {}
    __host__ __device__ 
    T operator()(T* grid, int idx, dim3 grid_size) const {
        return value; 
    }
    T value;
};
