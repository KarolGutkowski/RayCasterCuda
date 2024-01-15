#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "cuda_runtime.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "bvh_structures/BVHNode.h"
#include "ray.h"
#include "lights_count.h"
#include "vec3.h"

void render(uchar3* grid, int width, int height, sphere* spheres, hitable_list_cpu* world, camera* camera, BVHNode*  bvh, uint32_t*  indices, uint32_t nodes_used, float3* light_postions, float3* light_colors);

#endif // !CPU_RENDERER_H
