#ifndef VEC3_H
#define VEC3_H

#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#include <cassert>

class vec3 {
public:
    float e[3];
    __host__ __device__ vec3(float x, float y, float z) {
        e[0] = x;
        e[1] = y;
        e[2] = z;
    }
    __host__ __device__ vec3() : vec3(0, 0, 0) {}
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__  vec3& operator/=(double t) {
        return *this *= 1 / t;
    }

    __host__ __device__ double length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const float3& v) {
    return vec3(u.e[0] * v.x, u.e[1] * v.y, u.e[2] * v.z);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t) {
    return (1 / t) * v;
}

__host__ __device__ inline vec3 operator-(float3 u, vec3 v)
{
    return vec3(u.x - v.e[0], u.y - v.e[1], u.z - v.e[2]);
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__host__ __device__ inline float get_value_at_index(vec3 v, uint32_t idx)
{
    switch (idx)
    {
    case 0:
        return v.x();
    case 1:    
        return v.y();
    case 2:    
        return v.z();
    default:
        assert(false);
    }
}

__host__ __device__ inline vec3 reflect(vec3 vector, vec3 normal)
{
    return vector - 2 * (dot(vector, normal)) * normal;
}

__host__ __device__ inline vec3 clamp_color(vec3 color)
{
    return vec3(fminf(color.x(), 1.0f), fminf(color.y(), 1.0f), fminf(color.z(), 1.0f));
}

#endif // !VEC3_H
