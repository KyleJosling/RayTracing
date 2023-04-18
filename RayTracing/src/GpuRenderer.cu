#include "GpuRenderer.cuh"

#include <iostream>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "Scene.h"
#include "Camera.h"
#include "Ray.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
// #include <cublas_v2.h> TODO? 


__device__ curandState_t rand_state;
__global__ void setupCurand(unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &rand_state);
}

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

  __device__
  glm::vec3 Random(){

    float rand1 = curand_uniform(&rand_state) - 0.5f;
    float rand2 = curand_uniform(&rand_state) - 0.5f;
    float rand3 = curand_uniform(&rand_state) - 0.5f;
    return glm::vec3(rand1, rand2, rand3);
  }

}
struct HitPayload {

  float HitDistance = -1.0f;
  glm::vec3 WorldPosition;
  glm::vec3 WorldNormal;

  int ObjectIndex;

};

struct GpuPayload{

  uint32_t frame_index;
  uint32_t width;
  uint32_t height;
  uint32_t * image_data;
  glm::vec4 * accumulation_data;
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
HitPayload Miss(Ray &ray){
  HitPayload payload;
  payload.HitDistance = -1.0f;
  return payload;
}

__device__
HitPayload ClosestHit(Ray &ray, float hitDistance, int objectIndex, const Sphere *spheres){

  HitPayload payload;
  payload.HitDistance = hitDistance;
  payload.ObjectIndex = objectIndex;

  const Sphere &closestSphere = spheres[objectIndex];

  glm::vec3 origin = ray.Origin - closestSphere.Position;
  payload.WorldPosition = origin + ray.Direction * hitDistance;
  payload.WorldNormal = glm::normalize(payload.WorldPosition);

  
  payload.WorldPosition+= closestSphere.Position;


  return payload;
}

__device__ 
HitPayload TraceRay(Ray ray, const Sphere * spheres, int num_spheres){

    // (bx^2 + by^2 + bz^2)t^2 + (2(axbx + ayby + azbz))t + (ax^2 + ay^2 + az^2- r^2) = 0
    // Solving for t, the distance along the ray where it intersects w the sphere
    // a -> origin of ray, b -> direction of ray
    // r is the radius of the sphere

    int closestSphere = -1;
    float hitDistance = FLT_MAX;

    for (size_t i = 0; i < num_spheres; i++){

        const Sphere &sphere = spheres[i];

        glm::vec3 origin = ray.Origin - sphere.Position;

        // a, b, c how they appear in the quadratic formula
        // float a = coord.x * coord.x + coord.y * coord.y + coord.z * coord.z; // This is just the dot product
        float a = glm::dot(ray.Direction, ray.Direction);
        // The second term is also the dot product between origin (a) and direction (b)
        float b = 2.0f * glm::dot(origin, ray.Direction);
        // Again, same w this one
        float c = glm::dot(origin, origin) - sphere.Radius * sphere.Radius;

        // Quadratic formula
        float discriminant = b * b - 4.0f * a * c;

        // If there was a discriminant, we hit the sphere with this ray (that corresponds to this pixel)
        if (discriminant < 0.0f)
            continue;

        // Get the solutions - we only calculate the cloests
        float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);

        if (closestT > 0.0f && closestT < hitDistance){
            hitDistance = closestT;
            closestSphere = (int)i;
        }

    }

    if (closestSphere < 0)
        return Miss(ray); // Send to our Miss shader


    return ClosestHit(ray, hitDistance, closestSphere, spheres);


}

__device__ 
glm::vec4 PerPixel(uint32_t x, uint32_t y, GpuPayload *gpu_payload){

  Ray ray;
  // ray.Origin = m_ActiveCamera->GetPosition();
  // ray.Direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalImage->GetWidth()];
  ray.Origin = gpu_payload->ray_position;
  ray.Direction = gpu_payload->ray_directions[x + y * gpu_payload->width];

  glm::vec3 color(0.0f);

  float multiplier = 1.0f;

  int bounces = 5;
  for (int i = 0; i < bounces; i++){

      HitPayload payload = TraceRay(ray, gpu_payload->spheres, gpu_payload->num_spheres);
      
      if (payload.HitDistance < 0.0f){
          glm::vec3 skyColor = glm::vec3(0.6f, 0.7f, 0.9f);
          color+= skyColor * multiplier;
          break;
      }

      glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
      float lightIntensity = glm::max(glm::dot(payload.WorldNormal, -lightDir), 0.0f); // == cos(angle)

      const Sphere& sphere = gpu_payload->spheres[payload.ObjectIndex];
      const Material &material = gpu_payload->materials[sphere.MaterialIndex];

      glm::vec3 sphereColor = material.Albedo; 
      sphereColor *= lightIntensity;
      color += sphereColor * multiplier;

      multiplier *= 0.5f;

      ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;

      ray.Direction = glm::reflect(ray.Direction,
          payload.WorldNormal + (material.Roughness * Utils::Random())); // TODO add randomness + material roughness
          //payload.WorldNormal + (0.5f)); // TODO add randomness + material roughness
  }

  return glm::vec4(color, 1.0f);
}

// Kernel function to add the elements of two arrays
__global__
void add(GpuPayload *gpu_payload)
{

  // Initialize curand

  // thread index contains the index of the current thread within its block
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gpu_payload->width || y >= gpu_payload->height) return;

  gpu_payload->accumulation_data[x + y * gpu_payload->width] += PerPixel(x, y, gpu_payload);

// Normalize the color + clamp it
  glm::vec4 accumulatedColor = gpu_payload->accumulation_data[x + y * gpu_payload->width];
  accumulatedColor /= (float)gpu_payload->frame_index;
  accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

  gpu_payload->image_data[x + y * gpu_payload->width] = Utils::ConvertToRGBA(accumulatedColor); 
}

void gpu_render(const Scene &scene, const Camera &camera, uint32_t * image_data, glm::vec4 * accumulation_data, const uint32_t &width, const uint32_t &height, uint32_t frame_index)
{


    thrust::device_vector<glm::vec3> ray_directions = camera.GetRayDirections();
    thrust::device_vector<Sphere> spheres = scene.DeviceSpheres;


    GpuPayload *payload_host = new GpuPayload();

    // TODO this probably only needs to be assigned once
    payload_host->frame_index = frame_index;
    payload_host->width = width;
    payload_host->height = height;
    payload_host->image_data = image_data;
    payload_host->accumulation_data =  accumulation_data;
    payload_host->ray_directions = thrust::raw_pointer_cast(ray_directions.data());
    payload_host->spheres = thrust::raw_pointer_cast(scene.DeviceSpheres.data());
    payload_host->num_spheres = scene.DeviceSpheres.size();
    payload_host->materials = thrust::raw_pointer_cast(scene.DeviceMaterials.data());
    payload_host->num_materials = scene.DeviceMaterials.size();
    payload_host->ray_position = camera.GetPosition();

    GpuPayload * payload_device;
    cudaMalloc(&payload_device, sizeof(GpuPayload));
    cudaMemcpy(payload_device, payload_host, sizeof(GpuPayload), cudaMemcpyHostToDevice);

    uint32_t tx = 8;
    uint32_t ty = 8;
    dim3 numBlocks((width + tx - 1)/tx, (height + ty -1) / ty); 
    dim3 blockSize(tx, ty);

    if (frame_index == 1){
      setupCurand<<<numBlocks, blockSize>>>(0);
    }

    add<<<numBlocks, blockSize>>>(payload_device);
    cudaDeviceSynchronize();
    cudaMemcpy(payload_host, payload_device, sizeof(GpuPayload), cudaMemcpyDeviceToHost);
    // cudaFree(payload);
}