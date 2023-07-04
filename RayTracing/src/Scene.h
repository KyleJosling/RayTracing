#pragma once

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>

struct Material {

    glm::vec3 Albedo {1.0f};
    float Roughness = 0.5f;
    float Metallic = 0.0f;
    glm::vec3 EmissionColor{ 0.0f };
    float EmissionPower = 0.0f;
};

struct Sphere{

    glm::vec3 Position{0.0f};
    float Radius = 0.5f;
    int MaterialIndex = 0;
};

struct Scene {

    void OnUpdate();

    // std::vector<Sphere> Spheres;
    // std::vector<Material> Materials;
    thrust::host_vector<Sphere> Spheres;
    thrust::host_vector<Material> Materials;
    thrust::device_vector<Sphere> DeviceSpheres;
    thrust::device_vector<Material> DeviceMaterials;

};