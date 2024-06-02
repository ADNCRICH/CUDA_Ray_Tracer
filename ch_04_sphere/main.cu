#include <bits/stdc++.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ch_02_vector/vec3.h"
#include "../ch_03_ray/ray.h"
#include "stb_image_write.h"
using namespace std;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// sorry for my sucky english, but I'll try my best to explain this

// sphere radius R, center at C
// (x-xc) * (x-xc) + (y-yc) * (y-yc) + (z-zc) * (z-zc) = R*R, P = (x, y, z), C = (xc, yc, zc)
// (P-C)•(P-C) = R*R
// P = p(t); (p(t)-C)•(p(t)-C) = R*R
// p(t) = A + tB; (A+tB-C)•(A+tB-C) = R*R
// (t*t*B•B) + (2*t*(A-C)•B) + ((A-C)•(A-C) - R*R) = 0

// ray hit sphere if solved quadratic for t and got 2 real answer
// b*b - 4*a*c > 0
template <typename com_t>
__device__ bool hit_sphere(const vec3<com_t> &center, com_t radius, const ray<com_t> &r) {
    vec3<com_t> AC = r.origin() - center;
    com_t a = dot(r.direction(), r.direction());
    com_t b = 2 * dot(AC, r.direction());
    com_t c = dot(AC, AC) - radius * radius;
    return b * b - 4 * a * c > 0;
}

template <typename com_t>
__device__ vec3<com_t> color(ray<com_t> r) {
    if (hit_sphere(vec3<com_t>(0, 0, -1), 0.5f, r))
        return vec3<com_t>(1, 0, 0) * 255.99f;
    com_t t = (unit_vector(r.direction()).y() + 1) * 0.5;                              // (-inf, inf) -> (0, 1)
    return ((1 - t) * vec3<com_t>(1, 1, 1) + t * vec3<com_t>(0.5, 0.7, 1)) * 255.99f;  // blend color white and blue
}

template <typename out_t, typename com_t>
__global__ void render(vec3<out_t> *output, int X, int Y, vec3<com_t> lower_left_corner, vec3<com_t> horizontal, vec3<com_t> vertical, vec3<com_t> origin) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= X || y >= Y) return;
    com_t u = com_t(x) / com_t(X), v = com_t(Y - y - 1) / com_t(Y);
    ray<com_t> r(origin, lower_left_corner + u * horizontal + v * vertical);
    output[X * y + x] = color(r);
}

int main() {
    int nx = 2000, ny = 1000;
    int thread_size = 16;
    vec3<uint8_t> *output;
    cudaMallocManaged(&output, nx * ny * sizeof(vec3<float>));

    dim3 threads(thread_size, thread_size);
    dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size);

    vec3<float> lower_left_corner(-2, -1, -1);
    vec3<float> horizontal(4, 0, 0);
    vec3<float> vertical(0, 2, 0);
    vec3<float> origin(0, 0, 0);

    render<<<blocks, threads>>>(output, nx, ny, lower_left_corner, horizontal, vertical, origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stbi_write_jpg("./sphere.jpg", nx, ny, 3, output, 100);

    checkCudaErrors(cudaFree(output));
}