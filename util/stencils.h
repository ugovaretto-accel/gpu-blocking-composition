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
    template < typename T > 
    __device__
    T operator()(const T* p, 
                 const ptrdiff_t& row_stride,
                 const ptrdiff_t& slice_stride) const {  
#if __CUDA_ARCH__ >= 350 && LDG
        // accessing directly *seems* to give a ~0.8 % performance
        // increase, but hard to notice given the variability of
        // the timing         
        // const T v4 = __ldg(p + row_stride);
        // const T v5 = __ldg(p - row_stride);
        // const T v6 = __ldg(p + slice_stride);
        // const T v7 = __ldg(p - slice_stride);
        // const T v1 = __ldg(p);
        // const T v2 = __ldg(p - 1);
        // const T v3 = __ldg(p + 1);
        const T v1 = grid_read(p);
        const T v2 = grid_read(p,  1);
        const T v3 = grid_read(p, -1);
        const T v4 = grid_read(p,  0,  1, row_stride);
        const T v5 = grid_read(p,  0, -1, row_stride);
        const T v6 = grid_read(p,  0,  0,  1, row_stride, slice_stride);
        const T v7 = grid_read(p,  0,  0, -1, row_stride, slice_stride);
        return T(-6)*v1 + v2 + v3 + v4 + v5 + v6 + v7;
#else        
        return T(-6) * p[0]
             + p[1]
             + p[-1]
             + p[slice_stride]
             + p[-slice_stride]
             + p[row_stride]
             + p[-slice_stride];

#endif             
    }
};
struct diffusion_3d {
    template < typename T >  
    __host__ __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size) const {
        const T v = gv(grid, center, 0, 0, 0, grid_size);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);
        return v
               +  T(0.1) * l3d(grid, center, grid_size); 
    }
    template < typename T >  
    __device__
    T operator()(const T* grid,
                 int row_stride,
                 int slice_stride) const {
#if __CUDA_ARCH__ >= 350 && LDG
        const T v = __ldg(grid);
#else       
        const T v = grid[0]; 
#endif        //printf("%d: %f\n", center, v);
        return v
               +  T(0.1) * l3d(grid, row_stride, slice_stride); 
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
        const T v = tex3D(in_texture, center.x, center.y, center.z);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);  
        T ret = T(-6) * v;
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

template <> 
struct laplacian_3d_texture<double> {
    __device__
    double operator()(const dim3& center) const {
        int2 v = tex3D(in_texture, center.x, center.y, center.z);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);  
        double ret = -6 * __hiloint2double(v.y, v.x);
        v = tex3D(in_texture, center.x - 1,
                  center.y, center.z);
        ret += __hiloint2double(v.y, v.x);
        v = tex3D(in_texture, center.x + 1,
                  center.y, center.z);
        ret += __hiloint2double(v.y, v.x);
        v = tex3D(in_texture, center.x,
                  center.y + 1, center.z);
        ret += __hiloint2double(v.y, v.x);
        v = tex3D(in_texture, center.x,
                  center.y - 1, center.z);
        ret += __hiloint2double(v.y, v.x); 
        v = tex3D(in_texture, center.x,
                  center.y, center.z + 1);
        ret += __hiloint2double(v.y, v.x);
        v = tex3D(in_texture, center.x,
                  center.y, center.z - 1);
        ret += __hiloint2double(v.y, v.x);
        return ret;
    }
};

template <typename T>
struct diffusion_3d_texture {
    __device__
    double operator()(const dim3& center) const {
        const T v = tex3D(in_texture, center.x, center.y, center.z);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);   
        return v + T(0.1) * l3d(center); 
    }
    laplacian_3d_texture< T > l3d;
};

template <>
struct diffusion_3d_texture<double> {
    __device__
    double operator()(const dim3& center) const {
        const int2 v = tex3D(in_texture, center.x, center.y, center.z);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);   
        return  __hiloint2double(v.y, v.x)+ 0.1 * l3d(center); 
    }
    laplacian_3d_texture< double > l3d;
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
