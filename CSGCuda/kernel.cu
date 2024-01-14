#include "kernel.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include <curand.h>
#include <curand_kernel.h>
#include "spheres_count.h"

#define THREADS_X 16
#define THREADS_Y 16
#define BVH_STACK_SIZE 64

__device__ vec3 calculateColor(hit_record rec, ray r, float3* light_postions, float3* light_colors)
{
	vec3 ambient = rec.color * 0.2f;

	vec3 total;
	for (int i = 0; i < LIGHTS_COUNT; i++) //get lights values
	{
		vec3 lightDir = unit_vector(light_postions[i] - rec.p);

		float diff = fmaxf(dot(rec.normal, lightDir), 0.0f);
		vec3 diffuse = diff * rec.color * 0.3f * light_colors[i];

		vec3 viewDir = unit_vector(r.origin() - rec.p);
		vec3 reflectDir = reflect(-lightDir, unit_vector(rec.normal));

		float spec = pow(fmaxf(dot(viewDir, reflectDir), 0.0), 40.0f);
		vec3 specular = spec * rec.color * 0.1f;
		total += diffuse + specular;
	}
	return clamp_color(total + ambient);
}


__device__ vec3 get_color(
	const ray& r, 
	const hitable_list** world, 
	BVHNode* nodes,
	uint32_t* nodes_indices,
	const uint32_t nodes_used, 
	float3* light_postions,
	float3* light_colors) 
{
	/*for (int i = 0; i < nodes_used; i++)
	{
		if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
		{
			printf("[%d] is box with n=%d spheres starting at %d\n", i, nodes[i].spheresCount, nodes[i].leftFirst);
		}
	}*/

	ray cur_ray = r;
	hit_record rec;

	BVHNode stack[BVH_STACK_SIZE];
	int stackIdx = 0;
	stack[stackIdx++] = nodes[0];

	vec3 color;
	float closest_hit = FLT_MAX;

	while (stackIdx>0)
	{
		BVHNode* current = &stack[stackIdx-1];
		stackIdx--;

		if (current->spheresCount == 0) // internal node, no primitives directly in it
		{
			if (current->intersectAABB(cur_ray))
			{
				if (current->leftFirst < nodes_used && current->leftFirst + 1 < nodes_used)
				{			
					uint32_t leftChildIdx = current->leftFirst;
					stack[stackIdx++] = nodes[leftChildIdx];
					stack[stackIdx++] = nodes[leftChildIdx + 1];
				}
			}
		}
		else
		{
			uint32_t first = current->leftFirst;
			uint32_t last = first + current->spheresCount - 1;
			if ((*world)->hit_range(cur_ray, 0.001f, closest_hit, rec, first, last, nodes_indices)) {
				closest_hit = rec.t;
				color = calculateColor(rec, cur_ray, light_postions, light_colors);
			}
		}
	}

	return color;
}

__global__ void color_grid_kernel(
	uchar3* grid, 
	const int width, 
	const int height, 
	hitable_list** world, 
	camera** camera, 
	BVHNode* bvh,
	uint32_t* indicies,
	uint32_t nodes_used,
	float3* light_postions,
	float3* light_colors)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || column >= width)
		return;

	int idx = row * width + column;

	/*if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < nodes_used; i++)
		{
			if (bvh[i].spheresCount != 0)
				printf("[%d] has %d spheres in it directly\n", i, bvh[i].spheresCount);
			else 
				printf("[%d] is an internal node and has children %d and %d \n", i, bvh[i].leftFirst, bvh[i].leftFirst+1);
		}
	}*/
	
	ray r = (*camera)->get_ray(column / (float)width, row / (float)height);
	vec3 col = get_color(r, world, bvh, indicies ,nodes_used,light_postions, light_colors);

	grid[idx].x = col.r() * 255;
	grid[idx].y = col.g() * 255;
	grid[idx].z = col.b() * 255;
}

__global__ void create_world(sphere** d_list, hitable_list** d_world, camera** d_camera,
	float originX, float originY, float originZ,
	float lookAtX, float lookAtY, float lookAtZ,
	float upX, float upY, float upZ);
__global__ void free_world(sphere** d_list, hitable_list** d_world);
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
	uint32_t* indicices,
	uint32_t nodes_used,
	float3* light_postions,
	float3* light_colors)
{
	dim3 blockDim = dim3(THREADS_X, THREADS_Y);
	dim3 gridDim = dim3((width + THREADS_X - 1) / THREADS_X, (height + THREADS_Y - 1) / THREADS_Y);

	color_grid_kernel<<<gridDim, blockDim>>>(grid, width, height, d_world, d_camera, bvh_d, indicices, nodes_used, light_postions, light_colors);

	destroy_camera << <1, 1 >> > (d_camera);

	checkCudaErrors(cudaDeviceSynchronize());
}

void create_world_on_gpu(sphere** d_list, hitable_list** d_world, camera** d_camera, Camera cpu_camera)
{
	create_world_launcher(d_list, d_world, d_camera, cpu_camera);
}

void destroy_world_resources_on_gpu(sphere** d_list, hitable_list** d_world)
{
	free_world << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
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


__global__ void free_world(sphere** d_list, hitable_list** d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < SPHERES_COUNT; i++)
		{
			delete* (d_list + i);
		}
		delete* d_world;
	}
}