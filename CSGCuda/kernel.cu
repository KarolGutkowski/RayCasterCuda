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

__device__ vec3 get_color(const ray& r, const hitable** world) {
	ray cur_ray = r;
	float cur_attenuation = 1.0f;

	hit_record rec;
	if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
		vec3 target = rec.p + rec.normal;
		cur_ray = ray(rec.p, target - rec.p);
		return cur_attenuation * rec.color;
	}
	/*else {
		vec3 unit_direction = unit_vector(cur_ray.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
		return cur_attenuation * c;
	}*/

	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void color_grid_kernel(uchar3* grid, const int width, const int height,hitable** list, hitable** world, camera** camera)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || column >= width)
		return;

	int idx = row * width + column;
	
	float u = column / (float)width;
	float v = row / (float)height;
	ray r = (*camera)->get_ray(u, v);
	vec3 col = get_color(r, world);

	grid[idx].x = col.r() * 255;
	grid[idx].y = col.g() * 255;
	grid[idx].z = col.b() * 255;
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera,
							float originX, float originY, float originZ,
							float lookAtX, float lookAtY, float lookAtZ,
							float upX, float upY, float upZ);
__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera);
__host__ void create_world_launcher(hitable** d_list, hitable** d_world, camera** d_camera, Camera cpu_camera);
__global__ void update_camera(camera** d_camera,
	float originX, float originY, float originZ,
	float lookAtX, float lookAtY, float lookAtZ,
	float upX, float upY, float upZ);

void launchKernel(uchar3* grid, const int width, const int height, hitable** d_list, hitable** d_world, camera** d_camera)
{
	/*hitable** d_list;
	hitable** d_world;
	camera** d_camera;

	checkCudaErrors(cudaMalloc((void**)&d_list, SPHERES_COUNT * sizeof(hitable*)));
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));*/


	dim3 blockDim = dim3(THREADS_X, THREADS_Y);
	dim3 gridDim = dim3((width + THREADS_X - 1) / THREADS_X, (height + THREADS_Y - 1) / THREADS_Y);

	color_grid_kernel<<<gridDim, blockDim>>>(grid, width, height, d_list, d_world, d_camera);

	checkCudaErrors(cudaDeviceSynchronize());
	/*free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));*/
}

void create_world_on_gpu(hitable** d_list, hitable** d_world, camera** d_camera, Camera cpu_camera)
{
	create_world_launcher(d_list, d_world, d_camera, cpu_camera);
}

void destroy_world_resources_on_gpu(hitable** d_list, hitable** d_world, camera** d_camera)
{
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
}


__host__ void create_world_launcher(hitable** d_list, hitable** d_world, camera** d_camera, Camera cpu_camera)
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

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera,
				float originX, float originY, float originZ,
				float lookAtX, float lookAtY, float lookAtZ,
				float upX, float upY, float upZ) {
	if (threadIdx.x == 0 && blockIdx.x == 0) 
	{
		curandState localState;
		curand_init(1984, 0, 0,&localState);
		for (int i = 0; i < SPHERES_COUNT; i++)
		{
			*(d_list+i) = new sphere(vec3(-(i%10), (curand_uniform(&localState)*2), -(i/10) - 0.5), curand_uniform(&localState), randomOnGPU(&localState));
		}
		/**(d_list) = new sphere(vec3(0, 0, -1), 0.2, vec3(0.8f, 0.3f, 0.3f));
		*(d_list + 1) = new sphere(vec3(1, 0, -1), 0.3, vec3(0.4f, 0.9f, 0.3f));
		*(d_list + 2) = new sphere(vec3(-1, 0, -1), 0.3, vec3(0.2f, 0.2f, 0.8f));*/
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


__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	/*delete* (d_list);
	delete* (d_list + 1);
	delete* (d_list + 2);*/
	for (int i = 0; i < SPHERES_COUNT; i++)
	{
		delete* (d_list + i);
	}
	delete* d_world;
	delete* d_camera;
}