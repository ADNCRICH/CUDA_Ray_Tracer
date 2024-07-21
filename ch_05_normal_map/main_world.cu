#include <bits/stdc++.h>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "hitable_list.h"
#include "sphere.h"
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
__global__ void init_world(hitable<T> **list, hitable<T> **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // init only once!
        list[0] = new sphere<T>(vec3<T>(0, 0, -1), 0.5);
        list[1] = new sphere<T>(vec3<T>(0, -100.5, -1), 100);
        *world = new hitable_list<T>(list, 2);
    }
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

template <typename O, typename T>
__global__ void render(vec3<O> *output, int X, int Y, vec3<T> lower_left, vec3<T> horizontal, vec3<T> vertical, vec3<T> origin, hitable<T> **world) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= X || y >= Y) return;
    T u = (T)x / (T)X, v = (T)(Y - y - 1) / (T)Y;
    ray<T> r(origin, lower_left + u * horizontal + v * vertical);
    output[y * X + x] = color(r, world) * 255.99;
}

template <typename T>
__global__ void free_mem(hitable<T> **world) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < (**((hitable_list<T> **)world)).list_size; i++)
            delete (**(hitable_list<T> **)world).list[i];
        delete *world;
    }
}

template <typename T>
__global__ void crop_image(int x, int y, int w, int h, int X, int Y, vec3<T> *data, vec3<T> *des){
    int i = blockDim.x * blockIdx.x + threadIdx.x, j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < x || i >= x + w || j < y || j >= y + h || x >= X || y >= Y) return;
    des[(j-y) * w + (i-x)] = data[j * X + i];
}

int main() {
    int nx = 2000, ny = 1000;
    int nx_c = 350, ny_c = 215, cx = 900, cy = 600; // crop image
    int thread_size = 16;
    clock_t st, ed;

    vec3<uint8_t> *output, *output_c;
    hitable<double> **list, **world;  // hitable is base class for sphere and hitable_list

    // good practice to use generic pointer (void type)
    // cudaMallocManaged is used to allocate memory that is accessible from all CPUs and GPUs
    checkCudaErrors(cudaMallocManaged((void **)&output, nx * ny * sizeof(vec3<uint8_t>)));
    // cudaMalloc is used to allocate memory that is accessible only from GPUs
    checkCudaErrors(cudaMalloc((void **)&list, 2 * sizeof(hitable<double> *)));
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hitable<double> *)));

    checkCudaErrors(cudaMallocManaged((void **)&output_c, nx_c * ny_c * sizeof(vec3<uint8_t>)));

    init_world<<<1, 1>>>(list, world);  // since constructor can call only by GPU
    checkCudaErrors(cudaGetLastError());

    dim3 threads(thread_size, thread_size);
    dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size);

    st = clock();
    render<<<blocks, threads>>>(output, nx, ny, vec3<double>(-2, -1, -1), vec3<double>(4, 0, 0),
                                vec3<double>(0, 2, 0), vec3<double>(0, 0, 0), world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    crop_image<<<blocks, threads>>>(cx, cy, nx_c, ny_c, nx, ny, output, output_c);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    ed = clock();

    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";

    stbi_write_jpg("sphere_world.jpg", nx, ny, 3, output, 100);
    stbi_write_jpg("sphere_world_cropped.jpg", nx_c, ny_c, 3, output_c, 100);

    free_mem<<<1, 1>>>(world);  // Free memory on CPU

    checkCudaErrors(cudaFree(output));  // Free memory on GPU
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(world));
}