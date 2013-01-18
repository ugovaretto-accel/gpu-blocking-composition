#pragma once

template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* data,
	                   const dim3& center,
	                   int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
	return data[center.x + xoff
	            + grid_size.x * (yoff + center.y
	                             + grid_size.y * (zoff + center.z))];
}
#define gv grid_3d_value_offset
