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
#include "bvh_structures/BVHNode.h"
#include <cstdint>

#define WINDOW_WIDTH 1280 
#define WINDOW_HEIGHT 720

void displayMsPerFrame(double& lastTime);
void drawTexture(int width, int height);
void UpdateBounds(const uint32_t nodeIdx, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices);
void build_bvh(BVHNode* bvh_nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed);
void Subdivide(uint32_t index, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed);
float3 subtract(float3 a, float3 b);
float get_value_at_index(float3 num, uint32_t idx);
float get_random_in_normalized();


int main(int argc, char** argv)
{
    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA RayCasting");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    // group this into a method;
    /*glEnable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);*/
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

    sphere* spheres = new sphere[SPHERES_COUNT];
    srand(time(NULL));
    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        spheres[i] = sphere(vec3((i % 10), get_random_in_normalized()* 30 - 15, -(i / 10) - 0.5), get_random_in_normalized()/2, vec3(get_random_in_normalized(), get_random_in_normalized(), get_random_in_normalized()));
    }
   
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    Camera cpu_camera = Camera();

    sphere** d_list;
    hitable_list** d_world;
    camera** d_camera;

    checkCudaErrors(cudaMalloc((void**)&d_list, SPHERES_COUNT * sizeof(sphere*)));

    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        sphere* sphere_device;
        checkCudaErrors(cudaMalloc((void**)&sphere_device, sizeof(sphere)));
        checkCudaErrors(cudaMemcpy(sphere_device, &spheres[i], sizeof(sphere), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void**)&(d_list[i]), &sphere_device, sizeof(sphere*), cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world_on_gpu(d_list, d_world, d_camera, cpu_camera);

    BVHNode bvhNodes[2 * SPHERES_COUNT - 1];
    uint32_t* indices = new uint32_t[SPHERES_COUNT];
    uint32_t nodes_used = 0;
    build_bvh(bvhNodes, spheres, indices, &nodes_used);

    BVHNode* bvh_d;
    checkCudaErrors(cudaMalloc((void**)&bvh_d, sizeof(BVHNode) * nodes_used));
    checkCudaErrors(cudaMemcpy(bvh_d, bvhNodes, sizeof(BVHNode) * nodes_used, cudaMemcpyHostToDevice));

    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cuda_pbo);
    cudaGraphicsResourceGetMappedPointer((void**)&grid, NULL, cuda_pbo);

    while (!glfwWindowShouldClose(window))
    {
        //put those three calls into one function
        processInput(window, cpu_camera);
        mouse_callback(window, cpu_camera);
        glfwPollEvents();

        update_camera_launcher(d_camera, cpu_camera);
        launchKernel(grid, W, H, d_list, d_world, d_camera, bvh_d, nodes_used);

        drawTexture(W, H);
        glfwSwapBuffers(window);
        displayMsPerFrame(lastTime);
    }

    cudaGraphicsUnmapResources(1, &cuda_pbo);

    destroy_world_resources_on_gpu(d_list, d_world, d_camera);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    cudaFree(bvh_d);

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

void UpdateBounds(const uint32_t nodeIdx, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices)
{
    BVHNode& node = nodes[nodeIdx];
    node.aabbMin = float3{ 1e30f, 1e30f, 1e30f };
    node.aabbMax = float3{ -1e30f, -1e30f, -1e30f };

    for (uint32_t first = node.leftFirst, i = 0; i < node.spheresCount; i++)
    {
        sphere sphere = spheres[first + i];
        float3 min{
            sphere.center.x() - sphere.radius,
            sphere.center.y() - sphere.radius,
            sphere.center.z() - sphere.radius
        };

        float3 max{
            sphere.center.x() + sphere.radius,
            sphere.center.y() + sphere.radius,
            sphere.center.z() + sphere.radius
        };

        node.aabbMax.x = fmaxf(max.x, node.aabbMax.x);
        node.aabbMax.y = fmaxf(max.y, node.aabbMax.y);
        node.aabbMax.z = fmaxf(max.z, node.aabbMax.z);

        node.aabbMin.x = fminf(min.x, node.aabbMin.x);
        node.aabbMin.y = fminf(min.y, node.aabbMin.y);
        node.aabbMin.z = fminf(min.z, node.aabbMin.z);
    }
}

void build_bvh(BVHNode* bvh_nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed)
{
    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        spheres_indices[i] = i;
    }
    uint32_t rootNodeIdx = 0;
    *nodesUsed = 1;

    BVHNode& root = bvh_nodes[rootNodeIdx];
    root.leftFirst = 0;
    root.spheresCount = SPHERES_COUNT;

    UpdateBounds(rootNodeIdx, bvh_nodes, spheres, spheres_indices);
    Subdivide(rootNodeIdx, bvh_nodes, spheres, spheres_indices, nodesUsed);
}

void Subdivide(uint32_t index, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed)
{
    BVHNode& node = nodes[index];
    float3 extent = subtract(node.aabbMax, node.aabbMin);

    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > get_value_at_index(extent, axis)) axis = 2;
    float splitPos = get_value_at_index(node.aabbMin, axis) + get_value_at_index(extent,axis) * 0.5f;

    int i = node.leftFirst;
    int j = i + node.spheresCount - 1;

    while (i <= j)
    {
        if (get_value_at_index(spheres[spheres_indices[i]].center, axis)< splitPos)
            i++;
        else
            std::swap(spheres_indices[i], spheres_indices[j--]);
    }

    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.spheresCount) return;



    int leftChildIdx = (*nodesUsed)++;
    int rightChildIdx = (*nodesUsed)++;
    nodes[leftChildIdx].leftFirst = node.leftFirst;
    nodes[leftChildIdx].spheresCount = leftCount;
    nodes[rightChildIdx].leftFirst = i;
    nodes[rightChildIdx].spheresCount = node.spheresCount - leftCount;
    node.leftFirst = leftChildIdx;
    node.spheresCount = 0;
    UpdateBounds(leftChildIdx, nodes, spheres, spheres_indices);
    UpdateBounds(rightChildIdx, nodes, spheres, spheres_indices);
    // recurse

    Subdivide(leftChildIdx, nodes, spheres, spheres_indices, nodesUsed);
    Subdivide(rightChildIdx, nodes, spheres, spheres_indices, nodesUsed);
}

float3 subtract(float3 a, float3 b)
{
    return float3
    {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

float get_value_at_index(float3 num, uint32_t idx)
{
    switch (idx)
    {
    case 0:
        return num.x;
    case 1:
        return num.y;
    case 2:
        return num.z;
    default:
        assert(false);
    }
}

float get_random_in_normalized()
{
    return rand() / (float)RAND_MAX;
}