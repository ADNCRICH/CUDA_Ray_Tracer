#include <bits/stdc++.h>
#include <curand_kernel.h>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ch_05_normal_map/hitable_list.h"
#include "../ch_05_normal_map/sphere.h"
#include "camera.h"
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
        list[0] = new sphere<T>(vec3<T>(0, 0, -1), 0.5);
        list[1] = new sphere<T>(vec3<T>(0, -100.5, -1), 100);
        *world = new hitable_list<T>(list, 2);
        *cam = new camera<T>();
    }
}

__global__ void init_random(int X, int Y, curandState *rand_state) {
    int x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= X || y >= Y) return;
    int idx = X * y + x + X * Y * blockIdx.z;
    // int idx = X * y + x;
    curand_init(1230123 + idx, 0, 0, &rand_state[idx]);  // each pixel use same seed but different sequence
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
        out_color += color(r, world);
    }

    output[X * y + x] += out_color / (T)sample * 255.99;  // Remove Denominator(sample) and magic would happen
}

template <typename T>
__device__ vec3<T> color(ray<T> &r, hitable<T> **world) {
    hit_record<T> temp;
    if ((*world)->hit(r, 0, DBL_MAX, temp))
        return temp.normal * 0.5 + 0.5;
    else {
        T t = unit_vector(r.direction()).y() * 0.5 + 0.5;
        return (1 - t) * vec3<T>(1, 1, 1) + t * vec3<T>(0.5, 0.7, 1);
    }
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

template <typename T1, typename T2>
__global__ void crop_image(int x, int y, int w, int h, int X, int Y, vec3<T1> *data, vec3<T2> *des){
    int i = blockDim.x * blockIdx.x + threadIdx.x, j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < x || i >= x + w || j < y || j >= y + h || x >= X || y >= Y) return;
    des[(j-y) * w + (i-x)] = data[j * X + i];
}

int main() {
    int nx = 2000, ny = 1000;
    int nx_c = 350, ny_c = 215, cx = 900, cy = 600; // crop image
    int thread_size = 8;
    int ns = 100;  // number of sampling for anti-aliasing
    clock_t st, ed;

    vec3<double> *output_t;
    vec3<uint8_t> *output, *output_c;
    hitable<double> **list, **world;  // hitable is base class for sphere and hitable_list
    camera<double> **cam;
    curandState *rand_state;

    // good practice to use generic pointer (void type)
    // cudaMallocManaged is used to allocate memory that is accessible from all CPUs and GPUs
    checkCudaErrors(cudaMallocManaged((void **)&output, nx * ny * sizeof(vec3<uint8_t>)));
    checkCudaErrors(cudaMallocManaged((void **)&output_t, nx * ny * sizeof(vec3<double>)));
    // cudaMalloc is used to allocate memory that is accessible only from GPUs
    checkCudaErrors(cudaMalloc((void **)&list, 2 * sizeof(hitable<double> *)));
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hitable<double> *)));
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera<double> *)));
    checkCudaErrors(cudaMalloc((void **)&rand_state, nx * ny * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&rand_state, nx * ny * sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged((void **)&output_c, nx_c * ny_c * sizeof(vec3<uint8_t>)));

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

    crop_image<<<blocks, threads>>>(cx, cy, nx_c, ny_c, nx, ny, output_t, output_c);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    ed = clock();

    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";

    for (int i = 0; i < nx * ny; i++)
        output[i] = output_t[i];

    stbi_write_jpg("sphere_world.jpg", nx, ny, 3, output, 100);
    stbi_write_jpg("sphere_world_cropped.jpg", nx_c, ny_c, 3, output_c, 100);

    free_mem<<<1, 1>>>(world, cam);  // Free memory on CPU

    checkCudaErrors(cudaFree(output));  // Free memory on GPU
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(rand_state));
}