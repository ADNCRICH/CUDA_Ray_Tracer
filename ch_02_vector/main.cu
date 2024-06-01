#include <bits/stdc++.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <time.h>

#include "vec3.h"
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

__global__ void image(vec3<uint8_t> *output, int X, int Y) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= X | y >= Y) return;
    output[y * X + x] = vec3<uint8_t>(255.99 * float(x) / float(X), 255.99 * float(Y - y - 1) / float(Y), 255.99 * 0.2);
}

int main() {
    int nx = 2000, ny = 1000;
    int thread_size = 16;
    vec3<uint8_t> *output;
    clock_t st, ed;

    cudaMallocManaged(&output, sizeof(vec3<uint8_t>) * nx * ny);
    dim3 threads(thread_size, thread_size);
    dim3 blocks((nx + thread_size - 1) / thread_size, (ny + thread_size - 1) / thread_size);

    st = clock();
    image<<<blocks, threads>>>(output, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    ed = clock();

    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";
    stbi_write_jpg("./image_cuda.jpg", nx, ny, 3, output, 100);

    checkCudaErrors(cudaFree(output));
}