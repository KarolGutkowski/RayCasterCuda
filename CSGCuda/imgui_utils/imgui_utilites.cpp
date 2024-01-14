#include "imgui_utilities.h"

void initializeImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void destroyImGuiContext() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void generateImGuiWindow(Camera& camera) {
    ImGuiIO& io = ImGui::GetIO();
    ImVec4 clear_color = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);

    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Edit scene");

    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Camera")) {
            ImGui::SliderFloat("camera X", &camera.Position.x, -2.0f, 2.0f);
            ImGui::SliderFloat("camera Y", &camera.Position.y, -2.0f, 2.0f);
            ImGui::SliderFloat("camera Z", &camera.Position.z, -2.0f, 2.0f);

            ImGui::SliderFloat("yaw", &camera.Yaw, -89.0f, 89.0f);
            ImGui::SliderFloat("pitch", &camera.Pitch, -89.0f, 89.0f);
            ImGui::SliderFloat3("up", (float *) &camera.WorldUp, 0.0f, 1.0f);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Colors"))
        {
            ImGui::SliderFloat("camera X", &camera.Position.x, -2.0f, 2.0f);
            ImGui::SliderFloat("camera Y", &camera.Position.y, -2.0f, 2.0f);
            ImGui::SliderFloat("camera Z", &camera.Position.z, -2.0f, 2.0f);

            ImGui::SliderFloat("yaw", &camera.Yaw, -89.0f, 89.0f);
            ImGui::SliderFloat("pitch", &camera.Pitch, -89.0f, 89.0f);
            ImGui::SliderFloat3("up", (float *) &camera.WorldUp, 0.0f, 1.0f);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    if (ImGui::Button("Button"))
        counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();

}

void ImGuiNewFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void renderImGui() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}