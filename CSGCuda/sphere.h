#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere /*: public hitable */
{
public:
    __device__ __host__ sphere(): sphere(vec3(0,0,0), 0, vec3(0.0, 0.0, 0.0)) {}
    __device__ __host__ sphere(vec3 cen, float r, vec3 color) : color(color),center(cen), radius(r) {};
    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const
    {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                vec3 hit_point = r.at(temp);
                    rec.t = temp;
                    rec.p = hit_point;
                    rec.normal = (rec.p - center) / radius;
                    rec.color = color;
                    return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                vec3 hit_point = r.at(temp);
                    rec.t = temp;
                    rec.p = r.at(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    rec.color = color;
                    return true;
            }
        }
        return false;
    }
    vec3 center;
    float radius;
    vec3 color;
};

#endif