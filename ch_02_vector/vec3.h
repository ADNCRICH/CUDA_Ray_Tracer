#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>

#include <iostream>

template <typename T>
class vec3 {
   public:
    __host__ __device__ vec3() {
        e[0] = T(0);
        e[1] = T(0);
        e[2] = T(0);
    }
    __host__ __device__ vec3(T e0, T e1, T e2) {
        e[0] = T(e0);
        e[1] = T(e1);
        e[2] = T(e2);
    }
    __host__ __device__ inline T x() { return e[0]; }
    __host__ __device__ inline T y() { return e[1]; }
    __host__ __device__ inline T z() { return e[2]; }
    __host__ __device__ inline T r() { return e[0]; }
    __host__ __device__ inline T g() { return e[1]; }
    __host__ __device__ inline T b() { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }  // sign +v
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline T operator[](int i) const { return e[i]; }
    __host__ __device__ inline T& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const T t);
    __host__ __device__ inline vec3& operator/=(const T t);

    __host__ __device__ inline T length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline T squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void make_unit_vector();

    template <typename U>
    __host__ __device__ inline vec3<T>& operator=(const vec3<U>& other) {
        static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, "T and U must be arithmetic types");

        e[0] = const_cast<vec3<U>&>(other).e[0];
        e[1] = const_cast<vec3<U>&>(other).e[1];
        e[2] = const_cast<vec3<U>&>(other).e[2];
        return *this;
    }

    T e[3];
};

template <typename T>
inline std::istream& operator>>(std::istream& is, vec3<T>& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const vec3<T>& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

template <typename T>
__host__ __device__ inline void vec3<T>::make_unit_vector() {
    T k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

template <typename T>
__host__ __device__ inline vec3<T> operator+(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator-(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator*(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator/(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator+(T t, const vec3<T>& v) {
    return vec3(t + v.e[0], t + v.e[1], t + v.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator+(const vec3<T>& v, T t) {
    return vec3(t + v.e[0], t + v.e[1], t + v.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator-(T t, const vec3<T>& v) {
    return vec3(t - v.e[0], t - v.e[1], t - v.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator-(const vec3<T>& v, T t) {
    return vec3(t - v.e[0], t - v.e[1], t - v.e[2]);
}

template <typename T, typename U>
__host__ __device__ inline vec3<T> operator*(U t, const vec3<T>& v) {
    return vec3<T>(t * v.e[0], t * v.e[1], t * v.e[2]);
}

template <typename T, typename U>
__host__ __device__ inline vec3<T> operator*(const vec3<T>& v, U t) {
    return vec3<T>(t * v.e[0], t * v.e[1], t * v.e[2]);
}

template <typename T>
__host__ __device__ inline vec3<T> operator/(vec3<T> v, T t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

template <typename T>
__host__ __device__ inline T dot(const vec3<T>& v1, const vec3<T>& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

template <typename T>
__host__ __device__ inline vec3<T> cross(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator*=(const T t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T>& vec3<T>::operator/=(const T t) {
    T k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

template <typename T>
__host__ __device__ inline vec3<T> unit_vector(vec3<T> v) {
    return v / v.length();
}

#endif
