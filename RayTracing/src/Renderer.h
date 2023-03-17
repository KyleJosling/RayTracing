#pragma once

#include <memory>
#include <glm/glm.hpp>

#include <Walnut/Image.h>

#include "Camera.h"
#include "Ray.h"

class Renderer {

public:
    Renderer() = default;

    void Render(const Camera &camera);

    void OnResize(uint32_t width, uint32_t height);

    std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; };

private:

    glm::vec4 TraceRay(const Ray &ray);

	std::shared_ptr<Walnut::Image> m_FinalImage;
	uint32_t* m_ImageData = nullptr;

};
