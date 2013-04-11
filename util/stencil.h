#pragma once

//halo_size = total halo width(left + right)
//all dimensions in bytes
inline size_t padded_halo_size(size_t halo_size,
	                           const size_t buffer_size,
                               const size_t alignment_size,
                               const size_t element_size) {
    while((buffer_size + halo_size) % alignment_size) {
    	halo_size += element_size;
    }
    return halo_size / element_size;
}

template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* data,
	                   const dim3& center,
#ifdef PTRDIFF_T                       
                       ptrdiff_t xoff, ptrdiff_t yoff, ptrdiff_t zoff,
#else                       
	                   int xoff, int yoff, int zoff,
#endif                       
                       const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG
	const ptrdiff_t idx = center.x + xoff
	                      + grid_size.x * (yoff + center.y
	                      + grid_size.y * (zoff + center.z));            
	return __ldg(data + idx);
#else
	return data[center.x + xoff
	            + grid_size.x * (yoff + center.y
	                             + grid_size.y * (zoff + center.z))];
#endif	            
}
template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* data,
	                   ptrdiff_t center,
	                   ptrdiff_t xoff, ptrdiff_t yoff, ptrdiff_t zoff,
                       const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG
	const ptrdiff_t idx = center + xoff
	                   + grid_size.x * (yoff + grid_size.y * zoff);                  
	return __ldg(data + idx);
#else
	return data[center + xoff
	                   + grid_size.x * (yoff + grid_size.y * zoff)];
#endif	            
}
#define gv grid_3d_value_offset

template< typename T >
__host__ __device__
T grid_3d_read(const T* data, ptrdiff_t x, ptrdiff_t y, ptrdiff_t z,
               const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG	
	const ptrdiff_t idx = x + grid_size.x * (y + grid_size.y * z);            
	return __ldg(data + idx);
#else
	return data[x + grid_size.x * (y + grid_size.y * z)];
#endif	     	
}

template< typename T >
__host__ __device__
void grid_3d_write(T* data, const T& v, ptrdiff_t x, ptrdiff_t y, ptrdiff_t z,
	               const dim3& grid_size) {
	data[x + grid_size.x * (y + grid_size.y * z)] = v;
}

template< typename T >
__host__ __device__
T grid_read(const T* p) {
#if __CUDA_ARCH__ >= 350 && LDG
    return __ldg(p);    	
#else
    return *p; 
#endif
}

template< typename T >
__host__ __device__
T grid_read(const T* p, int xoff) {
#if __CUDA_ARCH__ >= 350 && LDG
    return __ldg(p + xoff);        
#else
    return p[xoff]; 
#endif
}

template< typename T >
__host__ __device__
T grid_read(const T* p, int xoff, int yoff, ptrdiff_t row_offset) {
#if __CUDA_ARCH__ >= 350 && LDG
    return __ldg(p + xoff + yoff * row_offset);        
#else
    return p[xoff + yoff * row_offset]; 
#endif
}

template< typename T >
__host__ __device__
T grid_read(const T* p, int xoff, int yoff, int zoff,
            ptrdiff_t row_offset, ptrdiff_t slice_stride) {
#if __CUDA_ARCH__ >= 350 && LDG
    return __ldg(p + xoff + yoff * row_offset + zoff * slice_stride);        
#else
    return p[xoff + yoff * row_offset + zoff * slice_stride]; 
#endif
}
