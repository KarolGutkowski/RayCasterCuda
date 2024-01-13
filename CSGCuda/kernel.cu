#include "kernel.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include <curand.h>
#include <curand_kernel.h>
#include "spheres_count.h"

#define THREADS_X 32
#define THREADS_Y 32
#define BVH_STACK_SIZE 16

__device__ vec3 calculateColor(hit_record rec, ray r)
{
	vec3 ambient = rec.color * 0.2f;

	vec3 total;
	for (int i = 0; i < 10; i++) //get lights values
	{
		vec3 lightDir = unit_vector(vec3(i / 2 - 5, i / 2, 0.0f) - rec.p);

		float diff = fmaxf(dot(rec.normal, lightDir), 0.0f);
		vec3 diffuse = diff * rec.color * 0.1f;

		vec3 viewDir = unit_vector(r.origin() - rec.p);
		/*if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			printf("ray origin = (%f,%f,%f)\n", r.origin().x(), r.origin().y(), r.origin().z());
		}*/
		vec3 reflectDir = reflect(-lightDir, unit_vector(rec.normal));

		float spec = pow(fmaxf(dot(viewDir, reflectDir), 0.0), 40.0f);
		vec3 specular = spec * rec.color * 0.1f;
		total += diffuse + specular;
	}
	return clamp_color(total + ambient);
}


__device__ vec3 get_color(const ray& r, const hitable_list** world, BVHNode* nodes, const uint32_t nodes_used) {
	ray cur_ray = r;
	hit_record rec;

	BVHNode stack[BVH_STACK_SIZE];
	int stackIdx = 0;
	stack[stackIdx++] = nodes[0];
	while (stackIdx)
	{
		BVHNode* current = &stack[stackIdx-1];
		stackIdx--;

		if (current->spheresCount == 0) // internal node, no primitives directly in it
		{
			if (current->intersectAABB(cur_ray))
			{
				if(current->leftFirst < nodes_used && current->leftFirst + 1 < nodes_used)
					stack[stackIdx++] = nodes[current->leftFirst];
					stack[stackIdx++] = nodes[current->leftFirst+1];
			}
		}
		else
		{
			uint32_t first = current->leftFirst;
			uint32_t last = current->leftFirst + current->spheresCount -1;
			if ((*world)->hit_range(cur_ray, 0.001f, FLT_MAX, rec, first, last)) {
				return calculateColor(rec, cur_ray);
			}
		}

	}

	vec3 unit_direction = unit_vector(cur_ray.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	vec3 c = (1.0f - t) * vec3(0.1, 0.1, 0.1) + t * vec3(0.2, 0.3, 0.5);
	return c;
	return vec3(0.0, 0.0, 0.0);
}

__global__ void color_grid_kernel(
	uchar3* grid, 
	const int width, 
	const int height, 
	hitable_list** world, 
	camera** camera, 
	BVHNode* bvh,
	uint32_t nodes_used)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || column >= width)
		return;

	int idx = row * width + column;
	
	float u = column / (float)width;
	float v = row / (float)height;
	ray r = (*camera)->get_ray(u, v);
	vec3 col = get_color(r, world, bvh, nodes_used);

	grid[idx].x = col.r() * 255;
	grid[idx].y = col.g() * 255;
	grid[idx].z = col.b() * 255;
}

__global__ void create_world(sphere** d_list, hitable_list** d_world, camera** d_camera,
	float originX, float originY, float originZ,
	float lookAtX, float lookAtY, float lookAtZ,
	float upX, float upY, float upZ);
__global__ void free_world(sphere** d_list, hitable_list** d_world, camera** d_camera);
__host__ void create_world_launcher(sphere** d_list, hitable_list** d_world, camera** d_camera, Camera cpu_camera);
__global__ void update_camera(camera** d_camera,
	float originX, float originY, float originZ,
	float lookAtX, float lookAtY, float lookAtZ,
	float upX, float upY, float upZ);

__global__ void destroy_camera(camera** d_camera)
{
	delete* d_camera;
}

void launchKernel(uchar3* grid,
	const int width,
	const int height,
	sphere** d_list,
	hitable_list** d_world,
	camera** d_camera,
	BVHNode* bvh_d,
	uint32_t nodes_used)
{
	dim3 blockDim = dim3(THREADS_X, THREADS_Y);
	dim3 gridDim = dim3((width + THREADS_X - 1) / THREADS_X, (height + THREADS_Y - 1) / THREADS_Y);

	color_grid_kernel<<<gridDim, blockDim>>>(grid, width, height, d_world, d_camera, bvh_d, nodes_used);

	//destroy_camera << <1, 1 >> > (d_camera);

	checkCudaErrors(cudaDeviceSynchronize());
}

void create_world_on_gpu(sphere** d_list, hitable_list** d_world, camera** d_camera, Camera cpu_camera)
{
	create_world_launcher(d_list, d_world, d_camera, cpu_camera);
}

void destroy_world_resources_on_gpu(sphere** d_list, hitable_list** d_world, camera** d_camera)
{
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
}


__host__ void create_world_launcher(sphere** d_list, hitable_list** d_world, camera** d_camera, Camera cpu_camera)
{
	vec3 up = cpu_camera.getUpVector();
	vec3 origin = cpu_camera.getOrigin();
	vec3 lookat = cpu_camera.getLookAt();

	create_world << <1, 1 >> > (d_list, d_world, d_camera, 
		origin.x(), origin.y(), origin.z(),
		lookat.x(), lookat.y(), lookat.z(),
		up.x(), up.y(), up.z());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void update_camera_launcher(camera** d_camera, Camera cpu_camera)
{
	vec3 up = cpu_camera.getUpVector();
	vec3 origin = cpu_camera.getOrigin();
	vec3 lookat = cpu_camera.getLookAt();
	update_camera << <1, 1 >> > (d_camera,
		origin.x(), origin.y(), origin.z(),
		lookat.x(), lookat.y(), lookat.z(),
		up.x(), up.y(), up.z());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__device__ vec3 randomOnGPU(curandState* localState)
{
	return vec3(curand_uniform(localState), curand_uniform(localState), curand_uniform(localState));
}

__global__ void create_world(sphere** d_list, hitable_list** d_world, camera** d_camera,
				float originX, float originY, float originZ,
				float lookAtX, float lookAtY, float lookAtZ,
				float upX, float upY, float upZ) {
	if (threadIdx.x == 0 && blockIdx.x == 0) 
	{
		*d_world = new hitable_list(d_list, SPHERES_COUNT);
		*d_camera = new camera(vec3(originX, originY, originZ), vec3(lookAtX, lookAtY, lookAtZ), vec3(upX, upY, upZ), 90.0f, 16.0f/9.0f, 1.0f);
	}
}

__global__ void update_camera(camera** d_camera,
	float originX, float originY, float originZ,
	float lookAtX, float lookAtY, float lookAtZ,
	float upX, float upY, float upZ) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_camera = new camera(vec3(originX, originY, originZ), vec3(lookAtX, lookAtY, lookAtZ), vec3(upX, upY, upZ), 90.0f, 16.0f / 9.0f, 1.0f);
	}
}


__global__ void free_world(sphere** d_list, hitable_list** d_world, camera** d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < SPHERES_COUNT; i++)
		{
			delete* (d_list + i);
		}
		delete* d_world;
		//delete* d_camera; im already deleting it earlier
	}
}