#ifndef HITABLELISTH
#define HITABLELISTH

#include "hittable.h"

class hitable_list/* : public hitable*/ {
public:
    __device__ hitable_list(sphere** l, int n) { list = l; list_size = n; }
    __device__ __host__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const 
    {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if ((*(list + i))->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                temp_rec.color = list[i]->color;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ __host__ bool hit_range(const ray& r, float t_min, float t_max, hit_record& rec, uint32_t firstIdx, uint32_t lastIdx) const
    {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = firstIdx; i <= list_size && i <= lastIdx; i++) {
            if ((*(list + i))->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                temp_rec.color = list[i]->color;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
    sphere** list;
    int list_size;
};


#endif