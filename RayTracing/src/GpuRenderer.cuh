#pragma once
#include <cstdint>
#include "Scene.h"
#include "Camera.h"

void gpu_render(const Scene &scene, const Camera &camera, uint32_t * image_data, const uint32_t &width, const uint32_t &height);