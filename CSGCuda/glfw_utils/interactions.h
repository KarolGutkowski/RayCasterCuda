#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include <GLFW/glfw3.h>
#include "cpu_camera.h"

void resizeWindowCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void processInput(GLFWwindow* window, Camera& camera);
void mouse_callback(GLFWwindow* window, Camera& camera);

#endif // !INTERACTIONS_H