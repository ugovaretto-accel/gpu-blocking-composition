#pragma once

template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* center, int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
	return *(center + xoff + grid_size.x * (yoff + grid_size.y * zoff));
}
#define gv grid_3d_value_offset