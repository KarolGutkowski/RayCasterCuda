#ifndef IMGUI_UTILITIES_H
#define IMGUI_UTILITIES_H

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "glfw/glfw3.h"
#include "cpu_camera.h"

void initializeImGui(GLFWwindow* window);
void destroyImGuiContext();
void generateImGuiWindow(float& rotation_speed);
void ImGuiNewFrame();
void renderImGui();

#endif
