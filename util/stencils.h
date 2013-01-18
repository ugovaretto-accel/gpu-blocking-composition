#pragma once

#include <cuda_runtime.h>
#include <cstdio> //for printf in CUDA

#include "stencil.h"


struct laplacian_3d {
    template < typename T > 
    __host__ __device__
    T operator()(const T* grid, 
                 const dim3& center,
                 const dim3& grid_size) const {
        
        return T(-6) * gv(grid, center, 0, 0, 0, grid_size)
             + gv(grid, center, 1, 0, 0, grid_size)
             + gv(grid, center, -1, 0, 0, grid_size)
             + gv(grid, center, 0, 1, 0, grid_size)
             + gv(grid, center, 0, -1, 0, grid_size)
             + gv(grid, center, 0, 0, 1, grid_size)
             + gv(grid, center, 0, 0, -1, grid_size);

    }
};
struct diffusion_3d {
    template < typename T >  
    __host__ __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size) const {
        return gv(grid, center, 0, 0, 0, grid_size)
               +  T(0.1) * l3d(grid, center, grid_size); 
    }
    laplacian_3d l3d;
};

template < typename T >
struct init {
    init(T v=T()) : value(v) {}
    __host__ __device__ 
    T operator()(T* grid, 
                const dim3&/*center*/,
                const dim3&/*grid_size*/) const {
        return value; 
    }
    T value;
};


template < typename T >
struct print;

template <>
struct print<float> {
    __host__ __device__
    float operator()(float* grid,
                     const dim3& center,
                     const dim3& grid_size) const {
        printf("%f ", gv(grid, center, 0, 0, 0, grid_size));
        return gv(grid, center, 0, 0, 0, grid_size);
    }
};


template <>
struct print<double> {
    __host__ __device__
    double operator()(double* grid,
                      const dim3& center,
                      const dim3& grid_size) const {
        printf("%f ", gv(grid, center, 0, 0, 0, grid_size));
        return gv(grid, center, 0, 0, 0, grid_size);
    }
};

template <>
struct print<int> {
    __host__ __device__
    int operator()(int* grid,
                   const dim3& center,
                   const dim3& grid_size) const {
        printf("%d ", gv(grid, center, 0, 0, 0, grid_size));
        return gv(grid, center, 0, 0, 0, grid_size);
    }
};