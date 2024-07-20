#ifndef SPHEREH
#define SPHEREH
#include "hitable.h"

template <typename T>
class sphere : public hitable<T> {
   public:
    __device__ sphere() {}
    __device__ sphere(vec3<T> cen, T r) : center(cen), radius(r) {};
    __device__ virtual bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const;
    vec3<T> center;
    T radius;
};

template <typename T>
__device__ bool sphere<T>::hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const {
    vec3<T> oc = r.origin() - center;
    T a = dot(r.direction(), r.direction());
    T b = dot(oc, r.direction());  // remove factor 2
    T c = dot(oc, oc) - radius * radius;
    T tt = b * b - a * c;  // 1/4 * b*b - a*c
    if (tt > 0) {
        T temp = (-b - sqrt(tt)) / a;        // don't forget that a and c have to divided by 2
        if (t_min < temp && temp < t_max) {  // hit entry point (front side)
            rec.t = temp;
            rec.p = r.point_at_parameter(temp);
            rec.normal = (rec.p - center) / radius; // normalize
            return true;
        }
        temp = (-b + sqrt(tt)) / a;
        if (t_min < temp && temp < t_max) {  // hit exit point (back side)
            rec.t = temp;
            rec.p = r.point_at_parameter(temp);
            rec.normal = (rec.p - center) / radius; // normalize
            return true;
        }
    }
    return false;
}

#endif