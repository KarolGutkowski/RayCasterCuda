#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class camera {
public:
    __device__ __host__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture) { // vfov is top to bottom in degrees
        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width *  u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }
    __device__ __host__ ray get_ray(float s, float t) {
        return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
};

#endif