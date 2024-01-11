#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "glfw_utils/glfw_functions.h"
#include "glad_utils/glad_functions.h"
#include "kernel.h"
#include "cuda_gl_interop.h"
#include <cassert>
#include "glfw_utils/interactions.h"
#include "cpu_camera.h"
#include "spheres_count.h"

#define WINDOW_WIDTH 1280 
#define WINDOW_HEIGHT 720

void displayMsPerFrame(double& lastTime);
void drawTexture(int width, int height);
void mapCudaResourcePointer(cudaGraphicsResource* cuda_pbo, GLuint pbo, uchar3* grid);

int main(int argc, char** argv)
{
    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Electrons and protons");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    // group this into a method;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    GLuint pbo = 0;;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    uchar3* cpu_grid = (uchar3*)malloc(sizeof(uchar3) * WINDOW_WIDTH * WINDOW_HEIGHT);

    if (!cpu_grid)
    {
        std::cout << "failed to allocate cpu grid" << std::endl;
        return -1;
    }

    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
    {
        cpu_grid[i].x = 0;
        cpu_grid[i].y = 0;
        cpu_grid[i].z = 0;
    }

    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), cpu_grid, GL_STREAM_DRAW);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    uchar3* grid = 0;
    int W = WINDOW_WIDTH;
    int H = WINDOW_HEIGHT;
    cudaGraphicsResource * cuda_pbo;
   /* 
   cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cuda_pbo);

    cudaGraphicsResourceGetMappedPointer((void**)&grid, NULL, cuda_pbo);*/
    
   
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    Camera cpu_camera = Camera();

    hitable** d_list;
    hitable** d_world;
    camera** d_camera;

    checkCudaErrors(cudaMalloc((void**)&d_list, SPHERES_COUNT * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world_on_gpu(d_list, d_world, d_camera, cpu_camera);

    while (!glfwWindowShouldClose(window))
    {
        //put those three calls into one function
        processInput(window, cpu_camera);
        mouse_callback(window, cpu_camera);
        glfwPollEvents();

        cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsMapResources(1, &cuda_pbo);
        cudaGraphicsResourceGetMappedPointer((void**)&grid, NULL, cuda_pbo);
        update_camera_launcher(d_camera, cpu_camera);
        launchKernel(grid, W, H, d_list, d_world, d_camera);

        cudaGraphicsUnmapResources(1, &cuda_pbo);

        drawTexture(W, H);
        glfwSwapBuffers(window);
        displayMsPerFrame(lastTime);
    }

    //cudaGraphicsUnmapResources(1, &cuda_pbo);
    destroy_world_resources_on_gpu(d_list, d_world, d_camera);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    glfwTerminate();
    return 0;
}

void drawTexture(int width, int height) 
{
    glClear(GL_COLOR_BUFFER_BIT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, -1.0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

int nbFrames = 0;
void displayMsPerFrame(double& lastTime)
{
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0) {
        double elapsed = currentTime - lastTime;
        auto timePerFrame = elapsed * 1000.0 / double(nbFrames);
        auto framesPerSecond = 1000.0 / timePerFrame;
        printf("%f ms/frame (fps=%f)\n", timePerFrame, framesPerSecond);
        nbFrames = 0;
        lastTime += 1.0;
    }
}