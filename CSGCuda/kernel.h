#ifndef KERNEL_H
#define KERNEL_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include "vec3.h"
#include "ray.h"
#include "cuda_helpers/cuda_helper.h"
#include "cpu_camera.h"
void launchKernel(uchar3* grid, const int width, const int height, Camera cpu_camera);
#endif // ! KERNEL_H

