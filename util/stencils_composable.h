#pragma once

#include <cuda_runtime.h>
#include <cstdio> //for printf in CUDA

#include "stencil.h"

struct accessor {
    template <typename T>
    __host__ __device__
    T operator()(const T* grid,
                 int x, int, y, int z,
                 const dim3& grid_size) const {
        return grid_3d_read(grid, x, y, z, grid_size);
    }
    template <typename T>
    __host__ __device__
    void operator()(T* grid,
                    const T& v,
                    int x, int y, int z,
                    const dim3& grid_size) const {
        grid_3d_write(grid, v, x, y, z, grid_size);
    }
};
#ifdef ENABLE_SURFACE
template < typename T >
struct surface_accessor {
    __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size) const {
        surf3Dread(&v_, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z);
        return v_;
    }
    void operator()(const T* grid,
                 const T& v,
                 const dim3& center,
                 const dim3& grid_size) const {
        surf3Dwrite(v_, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z);
    }
    mutable T v_;
};
#endif
#ifdef ENABLE_TEXTURE
template < typename T >
struct texture_accessor {
    __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size) const {
        return tex3D(in_texture,
                     center.x,
                     center.y,
                     center.z);
    }
};
template <>
struct texture_accessor<double> {
    __device__
    double operator()(const T* grid,
                      const dim3& center,
                      const dim3& grid_size) const {
        v_ = tex3D(in_texture,
                   center.x,
                   center.y,
                   center.z);
        return __hiloint2double(v.y, v.x);
    }
    mutable int2 v_;
};
#endif

template <typename S1, typename S2>
struct composite_stencil {
    __host__ __device__
    composite_stencil(const S1& s1 = S1(),
                      const S2& s2 = S2()) : s1_(s1), s2_(s2) {}
    template < typename T > 
    __host__ __device__
    T operator()(const T* grid,
                 const AccessorT& gv, 
                 const dim3& center,
                 const dim3& grid_size) const {
        return s1_(grid, s2_, center, grid_size);
    S1 s1_;
    S2 s2_;     
};

struct laplacian_3d {
    template < typename T, typename AccessorT > 
    __host__ __device__
    T operator()(const T* grid,
                 const AccessorT& gv, //get value through accessor 
                 const dim3& center,
                 const dim3& grid_size) const {
        
        return T(-6) * gv(grid, center.x, center.y, center.z, grid_size)
             + gv(grid, center.x + 1, center.y, center.z, grid_size)
             + gv(grid, center.x - 1, center.y, center.z, grid_size)
             + gv(grid, center.x, center.y + 1, center.z, grid_size)
             + gv(grid, center.x, center.y - 1, center.z, grid_size)
             + gv(grid, center.x, center.y, center.z + 1, grid_size)
             + gv(grid, center.x, center.y, center.z + 1, grid_size);

    }
};
struct diffusion_3d {
    template < typename T, typename AccessorT >  
    __host__ __device__
    T operator()(const T* grid,
                 const AccessorT& gv, //get value through accessor 
                 const dim3& center,
                 const dim3& grid_size) const {
        const T v = gv(grid, center.x, center.y, center.z, grid_size);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);
        return v + T(0.1) * l3d(grid, center, grid_size); 
    }
    laplacian_3d l3d;
};


template < typename T >
struct init {
    init(T v=T()) : value(v) {}
    template < typename AccessorT >
    __host__ __device__ 
    T operator()(T* grid,
                 const AccessorT& /*accessor*/    
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