#pragma once

template < typename T >
__host__ __device__
T grid_3d_value_offset(const T* data, int idx, int xoff, int yoff, int zoff,
                       const dim3& grid_size) {
	return *(data + (idx + (xoff + grid_size.x * (yoff + grid_size.y * zoff))));
}
#define gv grid_3d_value_offset
