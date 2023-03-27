#include "Renderer.h"
#include "Walnut/Random.h"
#include <cstring>
#include <algorithm>
#include <thread>
#include <execution>

namespace Utils{
    static uint32_t ConvertToRGBA(const glm::vec4& color){
        uint8_t r = (uint8_t)(color.r * 255.0f);
        uint8_t g = (uint8_t)(color.g * 255.0f);
        uint8_t b = (uint8_t)(color.b * 255.0f);
        uint8_t a = (uint8_t)(color.a * 255.0f);
        uint32_t result = (a << 24 | b << 16 | g << 8 | r);
        return result;
    }

}

void Renderer::OnResize(uint32_t width, uint32_t height){


    if (m_FinalImage){

        // No resize necessary
        if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height) return;

        // TODO fork the repo and add this function
        m_FinalImage->Resize(width, height);


    } else {
        m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
    }

    delete[] m_ImageData;
    m_ImageData = new uint32_t[width * height];

    delete[] m_AccumulationData;
    m_AccumulationData = new glm::vec4[width * height];

    m_ImageHorizontalIterator.resize(width);
    for (uint32_t i = 0; i < width; i++){
        m_ImageHorizontalIterator[i] = i;
    }

    m_ImageVerticalIterator.resize(height);
    for (uint32_t i = 0; i < height; i++){
        m_ImageVerticalIterator[i] = i;
    }

}

void Renderer::Render(const Scene &scene, const Camera &camera){

    m_ActiveScene = &scene;
    m_ActiveCamera = &camera;


    if (m_FrameIndex == 1) {
        memset(m_AccumulationData,
            0, 
            m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(glm::vec4));
    }

#define MT 1
#if MT
    // 1920 x 1080 ~2m
    std::for_each(std::execution::par, m_ImageVerticalIterator.begin(), m_ImageVerticalIterator.end(),
        [this](uint32_t y){
            std::for_each(std::execution::par, m_ImageHorizontalIterator.begin(), m_ImageHorizontalIterator.end(),
                [this, y](uint32_t x){

                    glm::vec4 color = PerPixel(x, y);
                    // Accumulate the colors to form a smoother image
                    m_AccumulationData[x + y * m_FinalImage->GetWidth()] += color;

                    // Normalize the color + clamp it
                    glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalImage->GetWidth()];
                    accumulatedColor /= (float)m_FrameIndex;
                    color = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

                    m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(color);

                });
    });
#else
    // Render pixels
    for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++){

        for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++) {

            glm::vec4 color = PerPixel(x, y);
            // Accumulate the colors to form a smoother image
            m_AccumulationData[x + y * m_FinalImage->GetWidth()] += color;

            // Normalize the color + clamp it
            glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalImage->GetWidth()];
            accumulatedColor /= (float)m_FrameIndex;
            color = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

            m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(color);
        }
    }
#endif

    m_FinalImage->SetData(m_ImageData);

    if (m_Settings.Accumulate)
        m_FrameIndex++;
    else
        m_FrameIndex = 1;

}

Renderer::HitPayload Renderer::TraceRay(const Ray &ray){


    // (bx^2 + by^2 + bz^2)t^2 + (2(axbx + ayby + azbz))t + (ax^2 + ay^2 + az^2- r^2) = 0
    // Solving for t, the distance along the ray where it intersects w the sphere
    // a -> origin of ray, b -> direction of ray
    // r is the radius of the sphere

    int closestSphere = -1;
    float hitDistance = FLT_MAX;

    for (size_t i = 0; i < m_ActiveScene->Spheres.size(); i++){

        const Sphere &sphere = m_ActiveScene->Spheres[i];

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


    return ClosestHit(ray, hitDistance, closestSphere);


}

glm::vec4 Renderer::PerPixel(uint32_t x, uint32_t y){

    Ray ray;
    ray.Origin = m_ActiveCamera->GetPosition();
    ray.Direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalImage->GetWidth()];

    glm::vec3 color(0.0f);

    float multiplier = 1.0f;

    int bounces = 5;
    for (int i = 0; i < bounces; i++){

        Renderer::HitPayload payload = TraceRay(ray);
        
        if (payload.HitDistance < 0.0f){
            glm::vec3 skyColor = glm::vec3(0.6f, 0.7f, 0.9f);
            color+= skyColor * multiplier;
            break;
        }

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        float lightIntensity = glm::max(glm::dot(payload.WorldNormal, -lightDir), 0.0f); // == cos(angle)

        const Sphere& sphere = m_ActiveScene->Spheres[payload.ObjectIndex];
        const Material &material = m_ActiveScene->Materials[sphere.MaterialIndex];

        glm::vec3 sphereColor = material.Albedo;
        sphereColor *= lightIntensity;
        color += sphereColor * multiplier;

        multiplier *= 0.5f;

        ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;
        // if ( x == 100 && y == 100) std::cout << Walnut::Random::Vec3(-0.05f, 0.05f)[0] << std::endl;
        // float v1 = -0.5f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f+0.5f)));
        // float v2 = -0.5f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f+0.5f)));
        // float v3 = -0.5f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5f+0.5f)));

        // glm::vec3 rando = glm::vec3(v1, v2, v3);
        ray.Direction = glm::reflect(ray.Direction,
        payload.WorldNormal + (material.Roughness * Walnut::Random::Vec3(-0.5f, 0.5f)));
        // payload.WorldNormal + (material.Roughness * rando));


    }

    return glm::vec4(color, 1.0f);
}

Renderer::HitPayload Renderer::ClosestHit(const Ray &ray, float hitDistance, int objectIndex){

    Renderer::HitPayload payload;
    payload.HitDistance = hitDistance;
    payload.ObjectIndex = objectIndex;

    const Sphere &closestSphere = m_ActiveScene->Spheres[objectIndex];

    glm::vec3 origin = ray.Origin - closestSphere.Position;
    payload.WorldPosition = origin + ray.Direction * hitDistance;
    payload.WorldNormal = glm::normalize(payload.WorldPosition);

    
    payload.WorldPosition+= closestSphere.Position;


    return payload;
}

Renderer::HitPayload Renderer::Miss(const Ray &ray){
    Renderer::HitPayload payload;
    payload.HitDistance = -1.0f;
    return payload;
}
