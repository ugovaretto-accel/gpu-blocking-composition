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
	                   int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
#if __CUDA_ARCH__ >= 35 && LDG
	const int idx = center.x + xoff
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
	                   int center,
	                   int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
#if __CUDA_ARCH__ >= 35 && LDG
	const int idx = center + xoff
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
T grid_3d_read(const T* data, int x, int y, int z, const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG	
	const int idx = x + grid_size.x * (y + grid_size.y * z);            
	return __ldg(data + idx);
#else
	return data[x + grid_size.x * (y + grid_size.y * z)];
#endif	     	
}

template< typename T >
__host__ __device__
void grid_3d_write(T* data, const T& v, int x, int y, int z,
	               const dim3& grid_size) {
	data[x + grid_size.x * (y + grid_size.y * z)] = v;
}

