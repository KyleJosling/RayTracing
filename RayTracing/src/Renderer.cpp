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

void Renderer::Render(){
    // Render pixels
    for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++){

        for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++) {

            glm::vec2 coord = { 
                (float)x / (float)m_FinalImage->GetWidth(),
                (float)y / (float)m_FinalImage->GetHeight()
            };
            coord = coord * 2.0f - 1.0f; // Map the coordinate from 0->1 to -1->1

            glm::vec4 color = PerPixel(coord);
            color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));

            m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(color);
        }
    }

    m_FinalImage->SetData(m_ImageData);

}

glm::vec4 Renderer::PerPixel(glm::vec2 coord){

    uint8_t r = coord.x * 255.0f;
    uint8_t g = coord.y * 255.0f;

    glm::vec3 rayOrigin(0.0f, 0.0f, 1.0f);
    // Currently, we define our direction's z coordinate as -1
    glm::vec3 rayDirection(coord.x, coord.y, -1.0f);

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
    float a = glm::dot(rayDirection, rayDirection);
    // The second term is also the dot product between origin (a) and direction (b)
    float b = 2.0f * glm::dot(rayOrigin, rayDirection);
    // Again, same w this one
    float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;

    // Quadratic formula
    float discriminant = b * b - 4.0f * a * c;

    // If there was a discriminant, we hit the sphere with this ray (that corresponds to this pixel)
    if (discriminant < 0.0f)
        return glm::vec4( 0, 0, 0, 1);

    // Get the solutions - we only calculate the cloests
    float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);

    glm::vec3 hitPosition = rayOrigin + rayDirection * closestT;

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
    // return glm::vec4(sphereColor, 1.0f);
    return glm::vec4(1.0f, 0.0f, 0.0f, light);

}
