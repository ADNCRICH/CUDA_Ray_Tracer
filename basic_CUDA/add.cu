#include <bits/stdc++.h>
using namespace std;

__global__ void add_one_thread(int n, float *x, float *y) {
    int this_thread = threadIdx.x;  // current thread index = [0, second number]
    int blockSize = blockDim.x;     // number of all thread in this block = second number
    for (int i = this_thread; i < n; i += blockSize)
        y[i] = x[i] + y[i];
}

__global__ void add_block_thread(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // gridDim.x is number of block defined by first number
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}
// linux
// nvcc add.cu -o ../bin/add_cu
// nsys nvprof ../bin/add_cu && rm report*.nsys-rep && rm report*.sqlite

// windows
// nvcc add.cu -o ../bin/add_cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\" -I "C:/Program Files/CodeBlocks/MinGW/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/x86_64-w64-mingw32" && del ..\bin\*.exp && del ..\bin\*.lib
// nsys profile --stats=true ..\bin\add_cu.exe && del report*.nsys-rep && del report*.sqlite
int main() {
    int N = 1 << 24;

    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++)
        x[i] = 1.0f,
        y[i] = 2.0f;

    add_one_thread<<<1, 256>>>(N, x, y);

    int block_size = 256;                               // number of thread in each block
    int num_block = (N + block_size - 1) / block_size;  // round up number of block to support all workload

    add_block_thread<<<num_block, block_size>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 6.0f));
    cout << "Max error: " << maxError << "\n";

    cudaFree(x);
    cudaFree(y);
}