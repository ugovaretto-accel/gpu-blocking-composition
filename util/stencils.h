#pragma once

#include <cuda_runtime.h>
#include <cstdio>

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


template < typename T >
struct print;

template <>
struct print<float> {
    __host__ __device__
    float operator()(float* grid, int idx, dim3 grid_size) const {
        printf("%f ", grid[idx]);
        return grid[idx];
    }
};


template <>
struct print<double> {
    __host__ __device__
    double operator()(double* grid, int idx, dim3 grid_size) const {
        printf("%f ", grid[idx]);
        return grid[idx];
    }
};

template <>
struct print<int> {
    __host__ __device__
    int operator()(int* grid, int idx, dim3 grid_size) const {
        printf("%d ", grid[idx]);
        return grid[idx];
    }
};