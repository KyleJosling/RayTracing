#include "Test.cuh"
#include <iostream>
#include <math.h>
#include <iostream>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
// #include <cublas_v2.h> TODO? 


__device__
uint32_t another_method(float u, float v){

  uint8_t r = (uint8_t)(u * 255.0f);
  uint8_t g = (uint8_t)(v * 255.0f);
  uint8_t b = (uint8_t)(0 * 255.0f);
  uint8_t a = (uint8_t)(1 * 255.0f);
  uint32_t result = (a << 24 | b << 16 | g << 8 | r);

  return result;
}

// Kernel function to add the elements of two arrays
__global__
void add(uint32_t * image_data, uint32_t width, uint32_t height)
{

  // thread index contains the index of the current thread within its block
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  image_data[x + y * width] = another_method((float)x/(float)width, (float)y/(float)height);

}

void add_wrapper(uint32_t * image_data, const uint32_t &width, const uint32_t &height)
{
    uint32_t tx = 8;
    uint32_t ty = 8;
    dim3 numBlocks((width + tx - 1)/tx, (height + ty -1) / ty); 
    dim3 blockSize(tx, ty);
    add<<<numBlocks, blockSize>>>(image_data, width, height);
    cudaDeviceSynchronize();
}