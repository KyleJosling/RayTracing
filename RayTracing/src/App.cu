#include "App.cuh"

App::App():
m_Camera(45.0f, -0.1f, 100.0f){

    Material pinkSphere; 
    pinkSphere.Albedo = { 1.0f, 0.0f, 1.0f };
    pinkSphere.Roughness = 0.0f;
    m_Scene.Materials.push_back(pinkSphere);

    Material blueSphere;
    blueSphere.Albedo = { 0.2f, 0.3f, 1.0f };
    blueSphere.Roughness = 0.1f;
    m_Scene.Materials.push_back(blueSphere);

    {
        Sphere sphere;
        sphere.Position = { 0.0f, 0.0f, 0.0f };
        sphere.Radius = 1.0f;
        sphere.MaterialIndex = 0;
        m_Scene.Spheres.push_back(sphere);
    }

    {
        Sphere sphere;
        sphere.Position = { 0.0f, -101.0f, 0.0f };
        sphere.Radius = 100.0f;
        sphere.MaterialIndex = 1;
        m_Scene.Spheres.push_back(sphere);
    }
}

void App::Render(){

    // m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
    // m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);
    // m_Renderer.Render(m_Scene, m_Camera);
}