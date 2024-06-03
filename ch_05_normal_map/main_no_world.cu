#include <bits/stdc++.h>
#include <time.h>
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

// sphere in front of image plane; (A-C)•B < 0 ; -b > 0
// a > 0 anytime, b < 0 since shpere is convex, c > 0 when camera outside sphere
// imply -b > sqrt(b*b - 4*a*c) promising that seeing sphere in front of camera (t > 0) since a > 0
// (-b - sqrt(b*b - 4*a*c))/(2*a) > 0 (use negative term to get first hit position)
template <typename com_t>
__device__ com_t hit_sphere(const vec3<com_t> &center, com_t radius, const ray<com_t> &r) {
    vec3<com_t> AC = r.origin() - center;
    com_t a = dot(r.direction(), r.direction());
    com_t b = 2 * dot(AC, r.direction());
    com_t c = dot(AC, AC) - radius * radius;
    com_t tt = b * b - 4 * a * c;
    if (tt < 0)
        return -1;
    else
        return (-b - sqrt(tt)) / (2.0f * a);
}

template <typename com_t>
__device__ vec3<com_t> color(ray<com_t> r) {
    com_t t = hit_sphere(vec3<com_t>(0, 0, -1), 0.5, r);
    if (t > 0) {
        vec3<com_t> N = unit_vector(r.point_at_parameter(t) - vec3<com_t>(0, 0, -1));
        return (vec3<com_t>(N.x(), N.y(), N.z()) * 0.5 + 0.5) * 255.99;
    }
    t = (unit_vector(r.direction()).y() + 1) * 0.5;
    return ((1 - t) * vec3<com_t>(1, 1, 1) + t * vec3<com_t>(0.5, 0.7, 1)) * 255.99;
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
    clock_t st, ed;
    vec3<uint8_t> *output;
    cudaMallocManaged(&output, nx * ny * sizeof(vec3<double>));

    dim3 threads(thread_size, thread_size);
    dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size);

    vec3<double> lower_left_corner(-2, -1, -1);
    vec3<double> horizontal(4, 0, 0);
    vec3<double> vertical(0, 2, 0);
    vec3<double> origin(0, 0, 0);

    st = clock();
    render<<<blocks, threads>>>(output, nx, ny, lower_left_corner, horizontal, vertical, origin);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    ed = clock();

    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";

    stbi_write_jpg("./sphere_no_world.jpg", nx, ny, 3, output, 100);

    checkCudaErrors(cudaFree(output));
}