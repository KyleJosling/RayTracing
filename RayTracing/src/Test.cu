#include "Test.cuh"
#include <iostream>
#include <math.h>
#include <iostream>

// Kernel function to add the elements of two arrays
__global__
void add(uint32_t * image_data, uint32_t width, uint32_t height)
{

  // thread index contains the index of the current thread within its block
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  image_data[x + y * width] = 0xFF0000FF;

}

void add_wrapper(uint32_t * image_data, const uint32_t &width, const uint32_t &height)
{
    std::cout << width << std::endl;
    std::cout << height << std::endl;

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            image_data[i * width + j] = 0xFF00FFFF;
        }
    }

    uint32_t tx = 8;
    uint32_t ty = 8;
    dim3 numBlocks((width + tx - 1)/tx, (height + ty -1) / ty); 
    dim3 blockSize(tx, ty);
    add<<<numBlocks, blockSize>>>(image_data, width, height);
    cudaDeviceSynchronize();
}