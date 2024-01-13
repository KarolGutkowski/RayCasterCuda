#ifndef KERNEL_H
#define KERNEL_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include "vec3.h"
#include "ray.h"
#include "cuda_helpers/cuda_helper.h"
#include "cpu_camera.h"
#include "camera.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
#include "bvh_structures/BVHNode.h"

void launchKernel(uchar3* grid, 
	const int width, 
	const int height, 
	sphere** d_list, 
	hitable_list** d_world,
	camera** d_camera,
	BVHNode* bvh_d,
	uint32_t nodes_used);
void create_world_on_gpu(sphere** d_list, hitable_list** d_world, camera** d_camera, Camera cpu_camera);
void destroy_world_resources_on_gpu(sphere** d_list, hitable_list** d_world, camera** d_camera);
void update_camera_launcher(camera** d_camera, Camera cpu_camera);
#endif // ! KERNEL_H

