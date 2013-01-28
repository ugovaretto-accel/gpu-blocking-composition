#pragma once

#include <cuda_runtime.h>
#include <cstdio> //for printf in CUDA

#include "stencil.h"



template < typename RetT,
           int X_MIN_OFF = 0, int Y_MIN_OFF = 0, int Z_MIN_OFF = 0,
           int X_MAX_OFF = 0, int Y_MAX_OFF = 0, int Z_MAX_OFF = 0 >
struct stencil {
    typedef return_type = RetT;
    enum { x_min_off = X_MIN_OFF... };
    enum { x_extent = X_MAX_OFF - X_MIN_OFF ...};
};
//allows to treat leaf accessors(the ones that return data)
//the same as stencils and by doing so it permits to compose
//stencils at will
struct dummy_accessor_type {
};

template < typename T >
struct accessor : stencil< T >{
    typedef stencil< T >::return_type return_type;
    template < typename U,
               typename AccessorT = dummy_accessor_type >
    __host__ __device__
    return_type operator()(const U* grid,
                 const dim3& xyz,
                 const dim3& grid_size,
                 const AccessorT& acc = AccessorT()) const {
        return return_type(grid_3d_read(grid, xyz.x, xyz.y, xyz.z, grid_size));
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
    template < typename AccessorT = dummy_accessor_type>
    __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size,
                 const AccessorT& accessor = AccessorT()) const {
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
                 const dim3& center,
                 const dim3& grid_size,
                 const AccessorT& accessor ) const {
        return s1_(grid, center, grid_size, s2_);
    S1 s1_;
    S2 s2_;     
};

struct laplacian_3d {
    template < typename T, typename AccessorT > 
    __host__ __device__
    T operator()(const T* grid,
                 const dim3& center,
                 const dim3& grid_size,
                 const AccessorT& gv //get value through accessor ) 
                ) const {
        
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