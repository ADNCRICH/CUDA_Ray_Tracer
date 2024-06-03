#ifndef RAYH
#define RAYH
#include <../ch_02_vector/vec3.h>

template <typename T>
class ray {
   public:
    __device__ ray() {}
    __device__ ray(const vec3<T>& a, const vec3<T>& b) : Ori(a), Dir(b) {}
    __device__ vec3<T> origin() const { return Ori; }
    __device__ vec3<T> direction() const { return Dir; }
    __device__ vec3<T> point_at_parameter(T t) const { return Ori + t * Dir; }

    vec3<T> Ori;
    vec3<T> Dir;
};

#endif