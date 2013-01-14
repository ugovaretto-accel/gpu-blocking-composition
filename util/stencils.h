#pragma once

#include <cuda_runtime.h>

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

struct init {
    init(REAL_T v=REAL_T()) : value(v) {}
    __host__ __device__ 
    REAL_T operator()(REAL_T* grid, int idx, dim3 grid_size) const {
        return value; 
    }
    REAL_T value;
};
