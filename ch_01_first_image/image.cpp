#include <bits/stdc++.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <time.h>
using namespace std;

int main() {
    int nx = 2000, ny = 1000, channels = 3;
    uint8_t* output;
    output = (uint8_t*)malloc(nx * ny * channels * sizeof(uint8_t));
    clock_t st, ed;
    st = clock();
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            // float r = float(i) / float(nx);
            // float g = float(j) / float(ny);
            // float b = 0.2;

            // uint8_t ir = int(255.99 * r);
            // uint8_t ig = int(255.99 * g);
            // uint8_t ib = int(255.99 * b);

            output[(nx * (ny - j - 1) + i) * 3] = int(255.99 * float(i) / float(nx));
            output[(nx * (ny - j - 1) + i) * 3 + 1] = int(255.99 * float(j) / float(ny));
            output[(nx * (ny - j - 1) + i) * 3 + 2] = int(255.99 * 0.2);
        }
    }
    ed = clock();
    double timer_seconds = ((double)(ed - st)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";
    // cout << CLOCKS_PER_SEC;
    stbi_write_jpg("./first_image_cpp.jpg", nx, ny, 3, output, 100);
}