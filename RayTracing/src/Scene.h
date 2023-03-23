#pragma once


#include <vector>
#include <glm/glm.hpp>

struct Material {

    glm::vec3 Albedo;
    float Roughness = 0.5f;
    float Metallic = 0.0f;
};

struct Sphere{

    glm::vec3 Position{0.0f};
    float Radius = 0.5f;
    int MaterialIndex = 0;
};

struct Scene {

    std::vector<Sphere> Spheres;
    std::vector<Material> Materials;

};
