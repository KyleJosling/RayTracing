#include "Scene.h"

void Scene::OnUpdate(){
    DeviceMaterials = Materials;
    DeviceSpheres = Spheres;
    // thrust::device_vector<Sphere> DeviceSpheres;
    // thrust::device_vector<Material> DeviceMaterials;
};