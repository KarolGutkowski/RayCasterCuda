#ifndef BVHNODE
#define BVHNODE

#include <cstdint>
#include "cuda_runtime.h"
#include "ray.h"

__declspec(align(32))struct BVHNode
{
    float3 aabbMin, aabbMax;
    uint32_t leftFirst, spheresCount;
    __device__ __host__ bool intersectAABB(const ray& r)
    {
        // r.dir is unit direction vector of ray
        float3 dirfrac;
        dirfrac.x = 1.0f / r.direction().x();
        dirfrac.y = 1.0f / r.direction().y();
        dirfrac.z = 1.0f / r.direction().z();
        // aabbMin is the corner of AABB with minimal coordinates - left bottom, aabbMax is maximal corner
        // r.origin() is origin() of ray
        float t1 = (aabbMin.x - r.origin().x()) * dirfrac.x;
        float t2 = (aabbMax.x - r.origin().x()) * dirfrac.x;
        float t3 = (aabbMin.y - r.origin().y()) * dirfrac.y;
        float t4 = (aabbMax.y - r.origin().y()) * dirfrac.y;
        float t5 = (aabbMin.z - r.origin().z()) * dirfrac.z;
        float t6 = (aabbMax.z - r.origin().z()) * dirfrac.z;

        float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
        float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
        if (tmax < 0)
        {
            return false;
        }

        // if tmin > tmax, ray doesn't intersect AABB
        if (tmin > tmax)
        {
            return false;
        }

        return true;
    }
};


#endif