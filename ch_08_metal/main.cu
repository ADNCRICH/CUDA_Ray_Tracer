#include <bits/stdc++.h>
#include <curand_kernel.h>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ch_06_antialiasing/camera.h"
#include "hitable_list.cuh"
#include "material.cuh"
#include "sphere.cuh"
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

template <typename T>
__global__ void init_world(hitable<T> **list, hitable<T> **world, camera<T> **cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // init only once!
        list[0] = new sphere<T>(vec3<T>(0, 0, -1), 0.5, new lambertian<T>(vec3<T>(0.8, 0.3, 0.3)));
        list[1] = new sphere<T>(vec3<T>(0, -100.5, -1), 100, new lambertian<T>(vec3<T>(0.8, 0.8, 0.0)));
        list[2] = new sphere<T>(vec3<T>(1, 0, -1), 0.5, new metal<T>(vec3<T>(0.8, 0.6, 0.2), 1.0));
        list[3] = new sphere<T>(vec3<T>(-1, 0, -1), 0.5, new metal<T>(vec3<T>(0.8, 0.8, 0.8), 0.3));
        *world = new hitable_list<T>(list, 4);
        *cam = new camera<T>();
    }
}

__global__ void init_random(int X, int Y, curandState *rand_state) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= X || y >= Y) return;
    int idx = X * y + x + X * Y * blockIdx.z;
    // int idx = X * y + x;
    curand_init(1230123 + idx, 0, 0, &rand_state[idx]);  // each pixel use same seed but different sequence
    // curand_init(1984, idx, 0, &rand_state[idx]);
}

template <typename O, typename T>
__global__ void render(vec3<O> *output, int X, int Y, int sample, camera<T> **cam, hitable<T> **world, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= X || y >= Y) return;
    int idx = X * y + x + X * Y * blockIdx.z;
    // int idx = X * y + x;
    vec3<T> out_color;
    curandState current_rand_state = rand_state[idx];
    for (int i = 0; i < sample; i++) {
        T u = T(x + curand_uniform_double(&current_rand_state)) / T(X);
        T v = T(Y - y - 1 + curand_uniform_double(&current_rand_state)) / T(Y);
        ray<T> r = (*cam)->get_ray(u, v);
        out_color += color(r, world, &current_rand_state);
    }
    out_color /= (T)sample;

    // gamma correction
    out_color[0] = sqrt(out_color[0]);
    out_color[1] = sqrt(out_color[1]);
    out_color[2] = sqrt(out_color[2]);

    output[X * y + x] += out_color * 255.99;  // Remove Denominator(sample) and magic would happen
}

template <typename T>
__device__ vec3<T> color(ray<T> &r, hitable<T> **world, curandState *rand_state) {
    // if (state > 3) return vec3<T>(0, 0, 0); // exceed depth limit
    hit_record<T> rec;
    ray<T> curr_ray = r;
    vec3<T> curr_attenuation = vec3<T>(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        if ((*world)->hit(curr_ray, 0.001, FLT_MAX, rec)) {
            ray<T> scattered;
            vec3<T> attenuation;
            if (rec.material_ptr->scatter(curr_ray, rec, attenuation, scattered, rand_state)) {
                curr_attenuation *= attenuation;
                curr_ray = scattered;
            } else
                return vec3<T>(0, 0, 0);
            // return 0.5 * color(reflect_ray, world, rand_state); // recursive cause stack to overflow
        } else {
            T t = unit_vector(r.direction()).y() * 0.5 + 0.5;
            return ((1 - t) * vec3<T>(1, 1, 1) + t * vec3<T>(0.5, 0.7, 1)) * curr_attenuation;
        }
    }
    return vec3<T>(0, 0, 0);  // exceed loop limit
}

template <typename T>
__global__ void free_mem(hitable<T> **world, camera<T> **cam) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < (**((hitable_list<T> **)world)).list_size; i++)
            delete (**(hitable_list<T> **)world).list[i];
        delete *world;
        delete *cam;
    }
}

int main() {
    // cout << cudaDeviceSetLimit(cudaLimitStackSize, 1024*64) << "\n";
    int nx = 2000, ny = 1000;
    int thread_size = 8;
    int ns = 1600;  // number of sampling for anti-aliasing
    clock_t st, ed;

    using comp_t = float;

    vec3<comp_t> *output_t;
    vec3<uint8_t> *output;
    hitable<comp_t> **list, **world;  // hitable is base class for sphere and hitable_list
    camera<comp_t> **cam;
    curandState *rand_state;

    // good practice to use generic pointer (void type)
    // cudaMallocManaged is used to allocate memory that is accessible from all CPUs and GPUs
    checkCudaErrors(cudaMallocManaged((void **)&output, nx * ny * sizeof(vec3<uint8_t>)));
    checkCudaErrors(cudaMallocManaged((void **)&output_t, nx * ny * sizeof(vec3<comp_t>)));
    // cudaMalloc is used to allocate memory that is accessible only from GPUs
    checkCudaErrors(cudaMalloc((void **)&list, 4 * sizeof(hitable<comp_t> *)));
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hitable<comp_t> *)));
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera<comp_t> *)));
    checkCudaErrors(cudaMalloc((void **)&rand_state, nx * ny * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&rand_state, nx * ny * sizeof(curandState)));

    init_world<<<1, 1>>>(list, world, cam);  // since constructor can call only by GPU
    checkCudaErrors(cudaGetLastError());

    dim3 threads(thread_size, thread_size);
    dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size);
    // dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size,ns); // too much curand_init leads to slow down

    init_random<<<blocks, threads>>>(nx, ny, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    st = clock();
    render<<<blocks, threads>>>(output_t, nx, ny, ns, cam, world, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    ed = clock();

    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";

    for (int i = 0; i < nx * ny; i++)
        output[i] = output_t[i];

    stbi_write_jpg("Metal_Material.jpg", nx, ny, 3, output, 100);

    free_mem<<<1, 1>>>(world, cam);  // Free memory on CPU

    checkCudaErrors(cudaFree(output));  // Free memory on GPU
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(rand_state));
}