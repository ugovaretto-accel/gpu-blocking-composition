#pragma once

#include <cuda_runtime.h>
#include <cstdio> //for printf in CUDA

#include "stencil.h"

template < typename RetT,
           int X_MIN_OFF = 0, int Y_MIN_OFF = 0, int Z_MIN_OFF = 0,
           int X_MAX_OFF = 0, int Y_MAX_OFF = 0, int Z_MAX_OFF = 0 >
struct stencil {
    typedef RetT return_type;
    enum { x_min_off = X_MIN_OFF,
           y_min_off = Y_MIN_OFF,
           z_min_off = Z_MIN_OFF,
           x_max_off = X_MAX_OFF,
           y_max_off = Y_MAX_OFF,
           z_max_off = Z_MAX_OFF};
    enum { x_extent = X_MAX_OFF - X_MIN_OFF,
           y_extent = Y_MAX_OFF - Y_MIN_OFF,
           z_extent = Z_MAX_OFF - Z_MIN_OFF}
};

struct accessor {
    template < typename T >
    __host__ __device__
    T operator()(const T* grid,
                 int x, int y, int z,
                 ptrdiff_t row_offset,
                 ptrdiff_t slice_offset) const {
        return grid_read(grid, x, y, z, row_offset, slice_offset);
    }
    template < typename T >
    __host__ __device__
    T operator()(const T* grid,
                 int x, int y,
                 ptrdiff_t row_offset) const {
        return grid_read(grid, x, y, row_offset);
    }
    template < typename T >
    __host__ __device__
    T operator()(const T* grid,
                 int x) const {
        return grid_read(grid, x);
    }
    template < typename T >
    __host__ __device__
    T operator()(const T* grid) const {
        return grid_read(grid);
    }
    template <typename T>
    __host__ __device__
    void operator()(T* grid,
                    const T& v,
                    int x, int y, int z,
                    ptrdiff_t row_offset,
                    ptrdiff_t slice_offset) const {
        grid_write(grid, v, x, y, z, row_offset, slice_offset);
    }
    template <typename T>
    __host__ __device__
    void operator()(T* grid,
                    const T& v,
                    int x, int y,
                    ptrdiff_t row_offset) const {
        grid_write(grid, v, x, y, row_offset);
    }
    template <typename T>
    __host__ __device__
    void operator()(T* grid,
                    const T& v,
                    int x) const {
        grid_write(grid, v, x);
    }
    template <typename T>
    __host__ __device__
    void operator()(T* grid,
                    const T& v) const {
        grid_write(grid, v);
    }
};
#ifdef ENABLE_SURFACE
template < typename T >
struct surface_accessor {
    __device__
    T operator()(const T* grid,
                 int x, int y, int z) const {
        surf3Dread(&v_, in_surface, x * sizeof(T), y, z);
        return v_;
    }
    __device_
    void operator()(T* grid,
                    const T& v,
                    int x, int y, int z) const {
        surf3Dwrite(v_, in_surface, x * sizeof(T), y, z);
    }
    mutable T v_;
};
#endif
#ifdef ENABLE_TEXTURE
template < typename T >
struct texture_accessor {
    __device__
    T operator()(const T* grid,int x, int y, int z) const {
        return tex3D(in_texture, x, y, z);
    }
};
template <>
struct texture_accessor<double> {
    __device__
    double operator()(const T* grid, int x, int y, int z) const {
        v_ = tex3D(in_texture, x, y, z);
        return __hiloint2double(v.y, v.x);
    }
    mutable int2 v_;
};
#endif

template <typename S1, typename S2>
struct composite_stencil : 
    stencil< typename S1::return_type,
             S1::x_min_off - S2::x_min_off,
             S1::y_min_off - S2::y_min_off,
             S2::z_min_off - S2::z_min_off,
             S1::x_max_off + S2::x_max_off,
             S1::y_max_off + S2::y_max_off,
             S2::z_max_off + S2::z_max_off >  {
    typedef typename S1::return_type return_type;                
#ifndef __CUDA_ARCH__    
    __host__
#endif
    __device__
    composite_stencil(const S1& s1 = S1(),
                      const S2& s2 = S2()) : s1_(s1), s2_(s2) {}
    template < typename T, typename AccessorT > 
#ifndef __CUDA_ARCH__    
    __host__
#endif 
    __device__
    T operator()(const T* grid,
                 int x, int y, int z,
                 ptrdiff_t row_offset,
                 ptrdiff_t slice_offset,
                 const AccessorT& accessor ) const {
        return s1_(grid, x, y, z, row_offset, slice_offset,
                   s2_(grid, x, y, z, row_offset, slice_offset, accessor));
    }
    template < typename T, typename AccessorT > 
#ifndef __CUDA_ARCH__    
    __host__
#endif 
    __device__
    T operator()(const T* grid,
                 int x, int y,
                 ptrdiff_t row_offset,
                 const AccessorT& accessor ) const {
        return s1_(grid, x, y, row_offset,
                   s2_(grid, x, y, row_offset, accessor));
    }
    template < typename T, typename AccessorT > 
#ifndef __CUDA_ARCH__    
    __host__
#endif 
    __device__
    T operator()(const T* grid,
                 int x, 
                 const AccessorT& accessor ) const {
        return s1_(grid, x,
                   s2_(grid, x, accessor));
    }
    template < typename T, typename AccessorT > 
#ifndef __CUDA_ARCH__    
    __host__
#endif 
    __device__
    T operator()(const T* grid, 
                 const AccessorT& accessor ) const {
        return s1_(grid,
                   s2_(grid, accessor));
    }
        template < typename T, typename AccessorT > 
#ifndef __CUDA_ARCH__    
    __host__
#endif 
    __device__
    T operator()(const T* grid,
                 ptrdiff_t row_offset,
                 ptrdiff_t slice_offset, 
                 const AccessorT& accessor ) const {
        return s1_(grid, row_offset, slice_offset,
                   s2_(grid, row_offset, slice_offset, accessor));
    }                               
    mutable S1 s1_;
    mutable S2 s2_;     
};

//struct index3d {
//    int x, y, z;
//};

// template < typename RetT, typename CombinerT >
// struct fan_in_stencil {

//     RetT operator()(const T1* g1, const T2* g2... ) {
//         return combine(g1, x, y, z, row_offset1, slice_offset1,
//                        g2...., );
//     }  
// };

// template < typename CombinerT >
// struct fan_out_stencil {

//     RetT operator()(const T1* gin, T1* g1, T2* g2... ) {
//         return combine(g1, row_offset1, slice_offset1,
//                        g2...., );
//     }  
// };


//==============================================================================
template <typename T>
struct laplacian_3d : stencil<T, -1, -1, -1, 1, 1, 1> {
    template < typename U, typename AccessorT > 
    __host__ __device__
    return_type operator()(const U* grid,
                           int x, int y, int z,
                           ptrdiff_t row_offset,
                           ptrdiff_t slice_offset,
                           const AccessorT& gv //get value through accessor ) 
                          ) const {
        
        return T(-6) * gv(grid)
             + gv(grid, x + 1)
             + gv(grid, x - 1)
             + gv(grid, x, y + 1, row_offset)
             + gv(grid, x, y - 1, row_offset)
             + gv(grid, x, y, z + 1, row_offset, slice_offset)
             + gv(grid, x, y, z + 1, row_offset, slice_offset);

    }
    template < typename U, typename AccessorT > 
    __host__ __device__
    return_type operator()(const U* grid,
                           ptrdiff_t row_offset,
                           ptrdiff_t slice_offset,
                           const AccessorT& gv //get value through accessor ) 
                          ) const {
        
        return T(-6) * gv(grid)
             + gv(grid, +1)
             + gv(grid, -1)
             + gv(grid, 0, +1, row_offset)
             + gv(grid, 0, -1, row_offset)
             + gv(grid, 0, 0, +1, row_offset, slice_offset)
             + gv(grid, 0, 0, -1, row_offset, slice_offset);

    }
};
template <typename T>
struct madd_3d : stencil<T> {
    template < typename T, typename AccessorT >  
    __host__ __device__
    T operator()(const T* grid,
                 int x, int y, int z,
                 ptrdiff_t row_offset,
                 ptrdiff_t slice_offset,
                 const AccessorT& gv ) const {
        const T v = gv(grid);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);
        return v + T(0.1) * gv(grid, x, y, z, row_offset, slice_offset); 
    }
    template < typename T, typename AccessorT >  
    __host__ __device__
    T operator()(const T* grid,
                 ptrdiff_t row_offset,
                 ptrdiff_t slice_offset,
                 const AccessorT& gv ) const {
        const T v = gv(grid);
        //printf("%d %d %d: %f\n", center.x, center.y, center.z, v);
        return v + T(0.1) * gv(grid, row_offset, slice_offset); 
    }
};

template < typename T >
struct diffusion_3d : composite_stencil< madd_3d< T >, laplacian_3d< T > > {};

template <typename StencilT1, typename StencilT2 >
composite_stencil<StencilT1, StencilT2> 
operator >>=(const StencilT1& s1, const StencilT2&s2 s ) {
    return composite_stencil< S1, S2 >(s1, s2);
}

// <typename S1, typename S2, typename S3 >
// struct composite {
//     typedef composite_stencil<S1, composite_stencil< S2, S3 > > type;
// };




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