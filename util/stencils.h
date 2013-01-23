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

#ifdef ENABLE_SURFACE
template < typename T > 
struct laplacian_3d_surface {
    __device__
    T operator()(const dim3& center) const {
        T v;
        surf3Dread(&v, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z);
        T ret = T(-6) * v;
        surf3Dread(&v, in_surface,
                   (center.x + 1) * sizeof(T),
                   center.y, center.z);
        ret += v;
        surf3Dread(&v, in_surface,
                   (center.x - 1) * sizeof(T),
                   center.y, center.z);
        ret += v;
        surf3Dread(&v, in_surface,
                   center.x * sizeof(T),
                   center.y + 1, center.z);
        ret += v;
        surf3Dread(&v, in_surface,
                   center.x * sizeof(T),
                   center.y - 1, center.z);
        ret += v;
        surf3Dread(&v, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z + 1);
        ret += v;
        surf3Dread(&v, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z - 1);
        ret += v;
        return ret;
    }
};

template <typename T>
struct diffusion_3d_surface {
    __device__
    T operator()(const dim3& center) const {
        T v;
        surf3Dread(&v, in_surface, center.x * sizeof(T), center.y, center.z );
        //printf("%f\n", v);
        return v + T(0.1) * l3d(center); 
    }
    laplacian_3d_surface< T > l3d;
};

#ifdef ENABLE_TEXTURE
template < typename T > 
struct laplacian_3d_texture {
    __device__
    T operator()(const dim3& center) const {     
        T ret = T(-6) * tex3D(in_texture, center.x, center.y, center.x);
        ret += tex3D(in_texture, center.x - 1,
                      center.y, center.z);
        ret += tex3D(in_texture, center.x + 1,
                     center.y, center.z);
        ret += tex3D(in_texture, center.x,
                     center.y + 1, center.z);
        ret += tex3D(in_texture, center.x,
                     center.y - 1, center.z);
        ret += tex3D(in_texture, center.x,
                     center.y, center.z + 1);
        ret += tex3D(in_texture, center.x,
                     center.y, center.z - 1);
        return ret;
    }
};


template <typename T>
struct diffusion_3d_texture {
    __device__
    T operator()(const dim3& center) const {
        const T v = tex3D(in_texture, center.x, center.y, center.z );
        //printf("%f\n", v);
        return v + T(0.1) * l3d(center); 
    }
    laplacian_3d_texture< T > l3d;
};
#endif
#endif

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