#pragma once

template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* data,
	                   const dim3& center,
	                   int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG	
	const size_t idx = center.x + xoff
	            + grid_size.x * (yoff + center.y
	                             + grid_size.y * (zoff + center.z));            
	return __ldg(data + idx);
#else
	return data[center.x + xoff
	            + grid_size.x * (yoff + center.y
	                             + grid_size.y * (zoff + center.z))];
#endif	            
}
#define gv grid_3d_value_offset

template< typename T >
__host__ __device__
T grid_3d_read(const T* data, int x, int y, int z, const dim3& grid_size) {
#if __CUDA_ARCH__ >= 350 && LDG	
	const size_t idx = center.x + grid_size.x * (center.y
	                            + grid_size.y * center.z));            
	return __ldg(data + idx);
#else
	return data[center.x + grid_size.x * (center.y
	                     + grid_size.y * center.z)];
#endif	     	
}

template< typename T >
__host__ __device__
void grid_3d_write(T* data, const T& v, int x, int y, int z,
	               const dim3& grid_size) {
	data[center.x + grid_size.x * (center.y
	              + grid_size.y * center.z)] = v;
}

