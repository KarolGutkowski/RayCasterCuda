#include "opengl_util.h"

GLuint generatePBO()
{
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    return pbo;
}

uchar3* allocateCPUGrid()
{
    uchar3* cpu_grid = (uchar3*)malloc(sizeof(uchar3) * WINDOW_WIDTH * WINDOW_HEIGHT);

    if (!cpu_grid)
    {
        std::cout << "failed to allocate cpu grid" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
    {
        cpu_grid[i].x = 0;
        cpu_grid[i].y = 0;
        cpu_grid[i].z = 0;
    }

    return cpu_grid;
}

void addDataToPBO(uchar3* data, GLuint pbo)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), data, GL_STREAM_DRAW);
}


GLuint generateTexture()
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}