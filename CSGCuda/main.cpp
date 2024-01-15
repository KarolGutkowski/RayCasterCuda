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
#include "lights_count.h"
#include "opengl_util/opengl_util.h"
#include "screen_dimensions.h"
#include "imgui_utils/imgui_utilities.h"
#include "cpu_renderer.h"

void displayMsPerFrame(double& lastTime);
void drawTexture(int width, int height);
void UpdateBounds(const uint32_t nodeIdx, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices);
void build_bvh(BVHNode* bvh_nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed);
void Subdivide(uint32_t index, BVHNode* nodes, sphere* spheres, uint32_t* spheres_indices, uint32_t* nodesUsed);
float3 subtract(float3 a, float3 b);
float get_value_at_index(float3 num, uint32_t idx);
float get_random_in_normalized();
sphere* generateSpheresOnCPU();
sphere** transferSpheresToGPU(sphere* spheres);
hitable_list** allocateHitableList();
camera** allocateCamera();
BVHNode* copyBVHToGPU(BVHNode* bvhNodes, uint32_t nodes_used);
void* getMappedPointer(GLuint pbo, cudaGraphicsResource* cuda_pbo);
float3* transferLightPositionsToGPU(float3* light_postions);
float3* transferLightColorsToGPU(float3* light_colors);
void generate_random_lights(float3* light_postions, float3* light_colors);
void processUserInputs(GLFWwindow* window, Camera& cpu_camera);
uint32_t* transferIndicesToGPU(uint32_t* indices);
void process_command_line_arguments(int argc, char** argv, bool& run_cpu);

int main(int argc, char** argv)
{
    bool run_cpu = false;
    bool rotate_lights = false;

    process_command_line_arguments(argc, argv, run_cpu);
    
    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA RayCasting");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    GLuint pbo = generatePBO();
    uchar3* cpu_grid = allocateCPUGrid();

    addDataToPBO(cpu_grid, pbo);

    GLuint tex = generateTexture();

    sphere* spheres = generateSpheresOnCPU();
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    Camera cpu_camera = Camera();

    sphere** d_list = transferSpheresToGPU(spheres);
    hitable_list** d_world = allocateHitableList();
    camera** d_camera = allocateCamera();

    create_world_on_gpu(d_list, d_world, d_camera, cpu_camera);

    BVHNode bvhNodes[2 * SPHERES_COUNT - 1];
    uint32_t* indices = new uint32_t[SPHERES_COUNT];
    uint32_t nodes_used = 0;
    build_bvh(bvhNodes, spheres, indices, &nodes_used);

    uint32_t* indices_d = transferIndicesToGPU(indices);

    BVHNode* bvh_d = copyBVHToGPU(bvhNodes, nodes_used);

    float3* light_postions = new float3[LIGHTS_COUNT];
    float3* light_colors = new float3[LIGHTS_COUNT];

    generate_random_lights(light_postions, light_colors);

    float3* light_postions_d = transferLightPositionsToGPU(light_postions);
    float3* light_colors_d = transferLightColorsToGPU(light_colors);

    cudaGraphicsResource* cuda_pbo;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    hitable_list_cpu* world = new hitable_list_cpu(spheres, SPHERES_COUNT);

    while (!glfwWindowShouldClose(window))
    {
        processUserInputs(window, cpu_camera);

        if (!run_cpu) {
            uchar3* grid = (uchar3*)getMappedPointer(pbo, cuda_pbo);

            update_camera_launcher(d_camera, cpu_camera);
            launchKernel(grid, WINDOW_WIDTH, WINDOW_HEIGHT, d_list, d_world, d_camera, bvh_d, indices_d, nodes_used, light_postions_d, light_colors_d);

            cudaGraphicsUnmapResources(1, &cuda_pbo);
        }
        else
        {
            vec3 up = cpu_camera.getUpVector();
            vec3 origin = cpu_camera.getOrigin();
            vec3 lookat = cpu_camera.getLookAt();

            camera* cam = new camera(origin, lookat, up, 90.0f, 16.0f / 9.0f, 1.0f);

            render(cpu_grid, WINDOW_WIDTH, WINDOW_HEIGHT, spheres, world, cam, bvhNodes, indices, nodes_used, light_postions, light_colors);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), cpu_grid, GL_STREAM_DRAW);
            delete cam;
        }

        drawTexture(WINDOW_WIDTH, WINDOW_HEIGHT);

        glfwSwapBuffers(window);
        displayMsPerFrame(lastTime);
    }

    destroy_world_resources_on_gpu(d_list, d_world);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    cudaFree(light_postions_d);
    cudaFree(light_colors_d);
    cudaFree(indices_d);
    cudaFree(bvh_d);

    delete world;
    delete[] spheres;
    //delete[] bvhNodes;
    delete[] indices;
    delete[] light_postions;
    delete[] light_colors;
    free(cpu_grid);

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
        uint32_t leafTriIdx = spheres_indices[first + i];
        sphere& sphere = spheres[leafTriIdx];
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
    if (node.spheresCount <= 5) return;

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

sphere* generateSpheresOnCPU()
{
    sphere* spheres = new sphere[SPHERES_COUNT];
    srand(time(NULL));
    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        spheres[i] = sphere(vec3((i % 10), get_random_in_normalized() * 30 -15, -(i / 100) - 0.5f), get_random_in_normalized() / 2 +0.2f, vec3(get_random_in_normalized(), get_random_in_normalized(), get_random_in_normalized()));
    }
    return spheres;
}

sphere** transferSpheresToGPU(sphere* spheres)
{
    sphere** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, SPHERES_COUNT * sizeof(sphere*)));
    for (int i = 0; i < SPHERES_COUNT; i++)
    {
        sphere* sphere_device;
        checkCudaErrors(cudaMalloc((void**)&sphere_device, sizeof(sphere)));
        checkCudaErrors(cudaMemcpy(sphere_device, &spheres[i], sizeof(sphere), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void**)&(d_list[i]), &sphere_device, sizeof(sphere*), cudaMemcpyHostToDevice));
    }
    return d_list;
}

hitable_list** allocateHitableList()
{
    hitable_list** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
    return d_world;
}

camera** allocateCamera()
{
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    return d_camera;
}

BVHNode* copyBVHToGPU(BVHNode* bvhNodes, uint32_t nodes_used)
{
    BVHNode* bvh_d;
    checkCudaErrors(cudaMalloc((void**)&bvh_d, sizeof(BVHNode) * nodes_used));
    checkCudaErrors(cudaMemcpy(bvh_d, bvhNodes, sizeof(BVHNode) * nodes_used, cudaMemcpyHostToDevice));
    return bvh_d;
}

void* getMappedPointer(GLuint pbo, cudaGraphicsResource *cuda_pbo)
{
    void* grid = 0;
    cudaGraphicsMapResources(1, &cuda_pbo);
    cudaGraphicsResourceGetMappedPointer((void**)&grid, NULL, cuda_pbo);
    return grid;
}

float3* transferLightPositionsToGPU(float3* light_postions)
{
    float3* light_postions_d;
    checkCudaErrors(cudaMalloc((void**)&light_postions_d, sizeof(float3) * LIGHTS_COUNT));
    checkCudaErrors(cudaMemcpy(light_postions_d, light_postions, sizeof(float3) * LIGHTS_COUNT, cudaMemcpyHostToDevice));
    return light_postions_d;
}

float3* transferLightColorsToGPU(float3* light_colors)
{
    float3* light_colors_d;
    checkCudaErrors(cudaMalloc((void**)&light_colors_d, sizeof(float3)* LIGHTS_COUNT));
    checkCudaErrors(cudaMemcpy(light_colors_d, light_colors, sizeof(float3)* LIGHTS_COUNT, cudaMemcpyHostToDevice));
    return light_colors_d;
}

void generate_random_lights(float3* light_postions, float3* light_colors)
{
    for (int i = 0; i < LIGHTS_COUNT; i++)
    {
        light_postions[i] = { (i / 2.0f) - 5.0f, (i / 2.0f), (float)-i };
        light_colors[i] = { get_random_in_normalized(), get_random_in_normalized(), get_random_in_normalized() };
    }

    light_postions[9] = { 16.321367, 4.202590, -3.698214 };
    light_colors[9] = { 1.0f, 0.0f, 0.0f };
}

void processUserInputs(GLFWwindow* window, Camera& cpu_camera)
{
    processInput(window, cpu_camera);
    mouse_callback(window, cpu_camera);
    glfwPollEvents();
}

uint32_t* transferIndicesToGPU(uint32_t* indices)
{
    uint32_t* indices_d;
    cudaMalloc((void**)&indices_d, SPHERES_COUNT * sizeof(uint32_t));
    cudaMemcpy(indices_d, indices, SPHERES_COUNT * sizeof(uint32_t), cudaMemcpyHostToDevice);
    return indices_d;
}

void process_command_line_arguments(int argc, char** argv, bool& run_cpu)
{
    if (argc == 1)
        return;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp("-cpu", argv[i]) == 0)
        {
            run_cpu = true;
        }
    }
}