#ifndef HITABLEH
#define HITABLEH
#include "../ch_03_ray/ray.h"

template <typename T>
class material;

template <typename T>
struct hit_record {
    T t;
    vec3<T> p;
    vec3<T> normal;
    material<T>* material_ptr;
};

template <typename T>
class hitable {
   public:
    __device__ virtual bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const = 0;
};

#endif