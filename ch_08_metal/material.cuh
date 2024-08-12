#ifndef MATERIALH
#define MATERIALH
#include "hitable.cuh"

template <typename T>
__device__ vec3<T> random_in_unit_sphere(curandState* rand_state) {
    vec3<T> p;
    do {
        p = 2.0 * vec3<T>(curand_uniform_double(rand_state), curand_uniform_double(rand_state), curand_uniform_double(rand_state)) - vec3<T>(1, 1, 1);
    } while (p.squared_length() >= 1.0);  // reject untill got vector in sphere
    return p;
}

// reflect v on surface with normal n ; reflected = v + 2n * (-v)•n
template <typename T>
__device__ vec3<T> reflect(const vec3<T>& v, const vec3<T>& n) {
    return v - 2 * dot(n, v) * n;
}

template <typename T>
class material {
   public:
    // scattering is how ray reflected on lambertian material (ex. BRDF)
    __device__ virtual bool scatter(const ray<T>& r_in, const hit_record<T>& rec, vec3<T>& attenuation, ray<T>& scattered, curandState* rand_state) const = 0;
};

template <typename T>
class lambertian : public material<T> {
   public:
    __device__ lambertian(const vec3<T>& a) : albedo(a) {}  // albedo reflection factor of surface in [0, 1], vec3 for 3 channels of color
    __device__ virtual bool scatter(const ray<T>& r_in, const hit_record<T>& rec, vec3<T>& attenuation, ray<T>& scattered, curandState* rand_state) const {
        vec3<T> target = rec.normal + random_in_unit_sphere<T>(rand_state);
        scattered = ray<T>(rec.p, target);
        attenuation = albedo;
        return true;
    }

    vec3<T> albedo;
};

template <typename T>
class metal : public material<T> {
   public:
    __device__ metal(const vec3<T>& a, T f) : albedo(a) { fuzz = min(max(f, 0.0), 1.0); }
    __device__ virtual bool scatter(const ray<T>& r_in, const hit_record<T>& rec, vec3<T>& attenuation, ray<T>& scattered, curandState* rand_state) const {
        vec3<T> reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray<T>(rec.p, reflected + fuzz * random_in_unit_sphere<T>(rand_state));
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }
    vec3<T> albedo;
    T fuzz;
};

#endif