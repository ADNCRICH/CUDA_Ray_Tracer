#ifndef RAYH
#define RAYH
#include <../ch_02_vector/vec3.h>

template <typename T = float>
class ray {
   public:
    __device__ ray() {}
    __device__ ray(const vec3<T>& a, const vec3<T>& b) : A(a), B(b) {}
    __device__ vec3<T> origin() const { return A; }
    __device__ vec3<T> direction() const { return B; }
    __device__ vec3<T> point_at_parameter(T t) const { return A + t * B; }

    vec3<T> A;
    vec3<T> B;
};

#endif