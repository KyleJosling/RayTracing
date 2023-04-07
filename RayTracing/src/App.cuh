#pragma once

#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

class App {
public:

    App();

__host__ void Render();

private:

    Renderer m_Renderer;
    Camera m_Camera;
    Scene m_Scene;
};
