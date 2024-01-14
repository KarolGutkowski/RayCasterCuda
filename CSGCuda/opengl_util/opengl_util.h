#ifndef OPENGL_UTIL_H
#define OPENGL_UTIL_H

#include <glad/glad.h>
#include "cuda_runtime.h"
#include <iostream>
#include "screen_dimensions.h"

GLuint generatePBO();
uchar3* allocateCPUGrid();
void addDataToPBO(uchar3* data, GLuint pbo);
GLuint generateTexture();

#endif // !OPENGL_UTIL_H
