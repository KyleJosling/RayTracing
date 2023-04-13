#include "GpuRenderer.cuh"

#include <iostream>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>

#include "Scene.h"
#include "Camera.h"
#include "Ray.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
// #include <cublas_v2.h> TODO? 

namespace Utils{
  __device__
  static uint32_t ConvertToRGBA(const glm::vec4& color){
    uint8_t r = (uint8_t)(color.r * 255.0f);
    uint8_t g = (uint8_t)(color.g * 255.0f);
    uint8_t b = (uint8_t)(color.b * 255.0f);
    uint8_t a = (uint8_t)(color.a * 255.0f);
    uint32_t result = (a << 24 | b << 16 | g << 8 | r);
    return result;
  }

}
struct HitPayload {

  float HitDistance = -1.0f;
  glm::vec3 WorldPosition;
  glm::vec3 WorldNormal;

  int ObjectIndex;

};

struct GpuPayload{

  uint32_t width;
  uint32_t height;
  uint32_t * image_data;
  glm::vec3 * ray_directions;


  const Sphere * spheres;
  uint32_t num_spheres;

  const Material * materials;
  uint32_t num_materials;

  glm::vec3 ray_position;

};

__device__
uint32_t another_method(float u, float v){

  uint8_t r = (uint8_t)(u * 255.0f);
  uint8_t g = (uint8_t)(v * 255.0f);
  uint8_t b = (uint8_t)(0 * 255.0f);
  uint8_t a = (uint8_t)(1 * 255.0f);
  uint32_t result = (a << 24 | b << 16 | g << 8 | r);

  return result;
}

__device__ 
HitPayload TraceRay(Ray ray){

  // TODO everything is a miss rn
  HitPayload payload;
  payload.HitDistance = -1.0f;
  return payload;
}

__device__ 
uint32_t PerPixel(uint32_t x, uint32_t y, GpuPayload *gpu_payload){

  Ray ray;
  // ray.Origin = m_ActiveCamera->GetPosition();
  // ray.Direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalImage->GetWidth()];
  ray.Origin = gpu_payload->ray_position;
  ray.Direction = gpu_payload->ray_directions[x + y * gpu_payload->width];

  if (x == 0 && y == 0){
    printf("Ray Origin at X: %d, Y: %d is :%f \n", x, y, ray.Origin.z);
    printf("Ray Direction at X: %d, Y: %d is :%f \n", x, y, ray.Direction.z);
  }

  glm::vec3 color(0.0f);

  float multiplier = 1.0f;

  int bounces = 5;
  for (int i = 0; i < bounces; i++){

      HitPayload payload = TraceRay(ray);
      
      if (payload.HitDistance < 0.0f){
          glm::vec3 skyColor = glm::vec3(0.6f, 0.7f, 0.9f);
          color+= skyColor * multiplier;
          break;
      }

      glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
      float lightIntensity = glm::max(glm::dot(payload.WorldNormal, -lightDir), 0.0f); // == cos(angle)

      const Sphere& sphere = gpu_payload->spheres[payload.ObjectIndex];
      // const Material &material = active_scene->DeviceMaterials[sphere.MaterialIndex];

      glm::vec3 sphereColor = glm::vec3(0, 1, 0);
      sphereColor *= lightIntensity;
      color += sphereColor * multiplier;

      multiplier *= 0.5f;

      ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;

      // glm::vec3 rando = glm::vec3(v1, v2, v3);
      ray.Direction = glm::reflect(ray.Direction,
          payload.WorldNormal + (0.5f)); // TODO add randomness + material roughness
  }

  return Utils::ConvertToRGBA(glm::vec4(color, 1.0f));
}

// Kernel function to add the elements of two arrays
__global__
void add(GpuPayload *gpu_payload)
{

  // thread index contains the index of the current thread within its block
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gpu_payload->width || y >= gpu_payload->height) return;

  gpu_payload->image_data[x + y * gpu_payload->width] = PerPixel(x, y, gpu_payload);
  // gpu_payload->image_data[x + y * gpu_payload->width] =  another_method((float)x/(float)gpu_payload->width, (float)y/(float)gpu_payload->height);

  // accumulation_data[x + y * width] = PerPixel(x, y); TODO

}

void gpu_render(const Scene &scene, const Camera &camera, uint32_t * image_data, const uint32_t &width, const uint32_t &height)
{

    thrust::device_vector<glm::vec3> ray_directions = camera.GetRayDirections();
    thrust::device_vector<Sphere> spheres = scene.DeviceSpheres;

    GpuPayload *payload_host = new GpuPayload();

    payload_host->width = width;
    payload_host->height = height;
    payload_host->image_data = image_data;
    payload_host->ray_directions = thrust::raw_pointer_cast(ray_directions.data());
    payload_host->spheres = thrust::raw_pointer_cast(scene.DeviceSpheres.data());
    payload_host->num_spheres = scene.DeviceSpheres.size();
    payload_host->materials = thrust::raw_pointer_cast(scene.DeviceMaterials.data());
    payload_host->num_materials = scene.DeviceMaterials.size();
    payload_host->ray_position = glm::vec3(0.0f, 0.0f, 6.0f);

    GpuPayload * payload_device;
    cudaMalloc(&payload_device, sizeof(GpuPayload));
    cudaMemcpy(payload_device, payload_host, sizeof(GpuPayload), cudaMemcpyHostToDevice);

    uint32_t tx = 8;
    uint32_t ty = 8;
    dim3 numBlocks((width + tx - 1)/tx, (height + ty -1) / ty); 
    dim3 blockSize(tx, ty);
    add<<<numBlocks, blockSize>>>(payload_device);
    cudaDeviceSynchronize();
    cudaMemcpy(payload_host, payload_device, sizeof(GpuPayload), cudaMemcpyDeviceToHost);
    // cudaFree(payload);
}