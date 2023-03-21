#pragma once

#include <memory>
#include <glm/glm.hpp>

#include <Walnut/Image.h>

#include "Camera.h"
#include "Ray.h"
#include "Scene.h"

class Renderer {

public:
    Renderer() = default;

    void Render(const Scene &scene, const Camera &camera);

    void OnResize(uint32_t width, uint32_t height);

    std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; };

private:

    struct HitPayload {

        float HitDistance = -1.0f;
        glm::vec3 WorldPosition;
        glm::vec3 WorldNormal;

        int ObjectIndex;

    };

    glm::vec4 PerPixel(uint32_t x, uint32_t y); // RayGen shader
    HitPayload TraceRay(const Ray &ray);
    HitPayload ClosestHit(const Ray &ray, float hitDistance, int objectIndex);
    HitPayload Miss(const Ray &ray);

    const Scene *m_ActiveScene = nullptr;
    const Camera *m_ActiveCamera = nullptr;

	std::shared_ptr<Walnut::Image> m_FinalImage;
	uint32_t* m_ImageData = nullptr;

};
