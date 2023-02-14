#pragma once

#include <memory>

#include <Walnut/Image.h>

class Renderer {

public:
    Renderer() = default;

    void Render();

    void OnResize(uint32_t width, uint32_t height);

    std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; };

private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	uint32_t* m_ImageData = nullptr;

};
