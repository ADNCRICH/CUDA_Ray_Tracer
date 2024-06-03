#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

template <typename T>
class hitable_list : public hitable<T> {
   public:
    __device__ hitable_list() {}
    __device__ hitable_list(hitable<T> **l, int n) : list(l), list_size(n) {}
    __device__ virtual bool hit(const ray<T> &r, T t_min, T t_max, hit_record<T> &rec) const;
    hitable<T> **list;
    int list_size;
};

template <typename T>
__device__ bool hitable_list<T>::hit(const ray<T> &r, T t_min, T t_max, hit_record<T> &rec) const {  // try to find nearest hit and store in reec
    hit_record<T> temp_rec;
    bool ch = false;        // is hit something?
    float closest = t_max;  // to find nearest hit
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest, temp_rec)) {
            ch = true;
            closest = temp_rec.t;
            rec = temp_rec;
        }
    }
    return ch;
}

#endif
