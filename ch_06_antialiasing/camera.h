#ifndef CAMERAH
#define CAMRRAH

#include "../ch_03_ray/ray.h"

template <typename T>
class camera {
   public:
    __device__ camera() {
        lower_left_corner = vec3<T>(-2.0, -1.0, -1.0);
        horizontal = vec3<T>(4.0, 0.0, 0.0);
        vertical = vec3<T>(0.0, 2.0, 0.0);
        origin = vec3<T>(0.0, 0.0, 0.0);
    }

    __device__ ray<T> get_ray(T u, T v) {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

    vec3<T> lower_left_corner;
    vec3<T> horizontal;
    vec3<T> vertical;
    vec3<T> origin;
};
#endif