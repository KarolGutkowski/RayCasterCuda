#ifndef GLFW_FUNCTIONS_H
#define GLFW_FUNCTIONS_H

#include <GLFW/glfw3.h>
#include <iostream>
#include "interactions.h"

void initiaLizeGFLW();
void setGLFWWindowHints();
GLFWwindow* createGLFWWindow(int width, int height, const char* windowTitle);
void setupCallbacks(GLFWwindow* window);

#endif // !GLFW_FUNCTIONS_H



