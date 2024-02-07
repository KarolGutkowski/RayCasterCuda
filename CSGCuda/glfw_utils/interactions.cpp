#include "interactions.h"

void resizeWindowCallback(GLFWwindow* window, int width, int height)
{
 	glfwSetWindowSize(window, width, height);
	glViewport(0, 0, width, height);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) 
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

float lastFrame = 0.0f;
int mouse_input_mode = GLFW_CURSOR_DISABLED;
void processInput(GLFWwindow* window, Camera& camera, bool& rotate_lights, float& rotation_speed_factor) 
{
    rotate_lights = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);

    float currentFrame = glfwGetTime();
    float deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.ProcessKeyboard(Camera_Movement::FORWARD, deltaTime);
    }
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.ProcessKeyboard(Camera_Movement::BACKWARD, deltaTime);
    }
    else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.ProcessKeyboard(Camera_Movement::LEFT, deltaTime);
    }
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.ProcessKeyboard(Camera_Movement::RIGHT, deltaTime);
    }
    else if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
    {
        rotation_speed_factor += 0.01f;
    }
    else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
    {
        rotation_speed_factor -= 0.01f;
    }

    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    else if(glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

bool firstMouse = true;
float lastX = 0.0f;
float lastY = 0.0f;
void mouse_callback(GLFWwindow* window, Camera& camera) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

