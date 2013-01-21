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
struct laplacian_3d_surface {
    template < typename T > 
    __host__ __device__
    T operator()(const surface<void, 3> grid, 
                 const dim3& center,
                 const dim3& grid_size) const {
        T v;
        surf3DRead(&v, grid,
                   center.x * sizeof(T),
                   center.y, center.z);
        T ret += T(-6) * v;
        surf3DRead(&v, grid,
                   (center.x + 1) * sizeof(T),
                   center.y, center.z);
        ret += v;
        surf3DRead(&v, grid,
                   (center.x - 1) * sizeof(T),
                   center.y, center.z);
        ret += v;
        surf3DRead(&v, grid,
                   center.x * sizeof(T),
                   center.y + 1, center.z);
        ret += v;
        surf3DRead(&v, grid,
                   center.x * sizeof(T),
                   center.y - 1, center.z);
        ret += v;
        surf3DRead(&v, grid,
                   center.x * sizeof(T),
                   center.y, center.z + 1);
        ret += v;
        surf3DRead(&v, grid,
                   center.x * sizeof(T),
                   center.y, center.z - 1);
        ret += v;
        return ret;
    }
};
struct diffusion_3d_surface {
    template < typename T >  
    __host__ __device__
    T operator()(surface<void, 3> grid,
                 const dim3& center,
                 const dim3& grid_size) const {
        T v;
        surf3DRead(&v, grid, center.x * sizeof(T), center.y, center.z );
        return v + T(0.1) * l3d(surf, center, grid_size); 
    }
    laplacian_3d_surface l3d;
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