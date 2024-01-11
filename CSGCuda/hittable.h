#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    vec3 color;
};

class hitable {
public:
    vec3 color;
    __device__ hitable(): color(vec3(0.0f, 0.0f, 0.0f)) {}
    __device__ hitable(vec3 _color) : color(_color) {}
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif