#include "cpu_renderer.h"

__host__ bool intersect_bvh(const ray& r, BVHNode* nodes, uint32_t nodeIdx, float& closestHit, hit_record& hit, const hitable_list_cpu* world, uint32_t* nodes_indices);
__host__ vec3 get_color(const ray& r, const hitable_list_cpu* world, BVHNode* nodes, uint32_t* nodes_indices,
	const uint32_t nodes_used, float3* light_postions, float3* light_colors);
__host__ vec3 calculateColorCPU(hit_record rec, ray r, float3* light_postions, float3* light_colors);

void render(uchar3* grid, int width, int height, sphere* spheres, hitable_list_cpu* world, camera* camera, BVHNode* bvh, uint32_t* indices, uint32_t nodes_used, float3* light_postions, float3* light_colors)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int row = i;
			int column = j;
			ray r = camera->get_ray(column / (float)width, row / (float)height);

			vec3 col = get_color(r, world, bvh, indices, nodes_used, light_postions, light_colors);

			int idx = row * width + column;
			grid[idx].x = col.r() * 255;
			grid[idx].y = col.g() * 255;
			grid[idx].z = col.b() * 255;
		}
	}
}

bool intersect_bvh(const ray& r, BVHNode* nodes, uint32_t nodeIdx, float& closestHit, hit_record& hit, const hitable_list_cpu* world, uint32_t* nodes_indices)
{
	BVHNode& node = nodes[nodeIdx];
	if (!node.intersectAABB(r))
	{
		return false;
	}
	if (node.spheresCount != 0)
	{
		hit_record rec;
		uint32_t first = node.leftFirst;
		uint32_t last = first + node.spheresCount - 1;
		if (world->hit_range(r, 0.001f, closestHit, rec, first, last, nodes_indices))
		{
			closestHit = rec.t;
			hit = rec;
			return true;
		}
	}
	else
	{
		bool hit_left = intersect_bvh(r, nodes, node.leftFirst, closestHit, hit, world, nodes_indices);
		bool hit_right = intersect_bvh(r, nodes, node.leftFirst + 1, closestHit, hit, world, nodes_indices);

		return hit_left || hit_right;
	}
}

__host__ vec3 get_color(const ray& r, const hitable_list_cpu* world, BVHNode* nodes, uint32_t* nodes_indices,
	const uint32_t nodes_used, float3* light_postions, float3* light_colors)
{

	hit_record hit;
	float closest_hit = FLT_MAX;
	uint32_t rootIndex = 0;
	if (intersect_bvh(r, nodes, rootIndex, closest_hit, hit, world, nodes_indices))
	{
		return calculateColorCPU(hit, r, light_postions, light_colors);
	}
	return vec3(0, 0, 0);
}


__host__ vec3 calculateColorCPU(hit_record rec, ray r, float3* light_postions, float3* light_colors)
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