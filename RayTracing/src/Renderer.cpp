#include "Renderer.h"
#include "Walnut/Random.h"

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

}

void Renderer::Render(const Camera &camera){

    Ray ray;
    ray.Origin = camera.GetPosition();

    // Render pixels
    for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++){

        for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++) {

            ray.Direction = camera.GetRayDirections()[x + y * m_FinalImage->GetWidth()];

            glm::vec4 color = TraceRay(ray);
            color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));

            m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(color);
        }
    }

    m_FinalImage->SetData(m_ImageData);

}

glm::vec4 Renderer::TraceRay(const Ray &ray){

    float radius = 0.5f;
    glm::vec3 sphereOrigin(0.0f, 0.0f, 0.0f);
    glm::vec3 lightDirection(-1.0f, -1.0f, -1.0f);
    lightDirection = glm::normalize(lightDirection);
    // rayDirection = glm::normalize(rayDirection);

    // (bx^2 + by^2 + bz^2)t^2 + (2(axbx + ayby + azbz))t + (ax^2 + ay^2 + az^2- r^2) = 0
    // Solving for t, the distance along the ray where it intersects w the sphere
    // a -> origin of ray, b -> direction of ray
    // r is the radius of the sphere
    
    // a, b, c how they appear in the quadratic formula
    // float a = coord.x * coord.x + coord.y * coord.y + coord.z * coord.z; // This is just the dot product
    float a = glm::dot(ray.Direction, ray.Direction);
    // The second term is also the dot product between origin (a) and direction (b)
    float b = 2.0f * glm::dot(ray.Origin, ray.Direction);
    // Again, same w this one
    float c = glm::dot(ray.Origin, ray.Origin) - radius * radius;

    // Quadratic formula
    float discriminant = b * b - 4.0f * a * c;

    // If there was a discriminant, we hit the sphere with this ray (that corresponds to this pixel)
    if (discriminant < 0.0f)
        return glm::vec4( 0, 0, 0, 1);

    // Get the solutions - we only calculate the cloests
    float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);

    glm::vec3 hitPosition = ray.Origin + ray.Direction * closestT;

    // Get the normal vector to the sphere at the hit position.
    // The vector points from the origin to the hit position, since its just a sphere.
    glm::vec3 normalToSphere = glm::normalize(hitPosition - sphereOrigin);

    // Get the light intensity, based on how much the ray bounces back torwards the camera
    // We use the negative of the light direction so the dot product gives us the correct
    // result (the scalar magnitude of the vector going TOWARDS the light source, which is
    // much more useful to us.)
    // the dot product gives us the cos(angle between vectors)
    float light = std::max(glm::dot(normalToSphere, -lightDirection), 0.0f);

    glm::vec3 sphereColor = (normalToSphere * 0.5f + 0.5f) * light;
    return glm::vec4(sphereColor, 1.0f);
    // return glm::vec4(1.0f, 0.0f, 0.0f, light);

}
