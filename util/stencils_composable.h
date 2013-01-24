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
                 const dim3& grid_size) {
        surf3Dwrite(v_, in_surface,
                   center.x * sizeof(T),
                   center.y, center.z);
    }
    T v_;
};

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
    int2 v_;
};
#endif

template <typename S1, typename S2>
struct composite_stencil {
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
                 const AccessorT& gv, 
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
                 const AccessorT& gv,
                 const dim3& center,
                 const dim3& grid_size) const {
        const T v = gv(grid, center.x, center.y, center.z, grid_size);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);
        return v + T(0.1) * l3d(grid, center, grid_size); 
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