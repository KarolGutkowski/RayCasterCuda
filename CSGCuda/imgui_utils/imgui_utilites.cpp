#include "imgui_utilities.h"

void initializeImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430 core");
}

void destroyImGuiContext() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void generateImGuiWindow(float& rotation_speed) {
    ImGuiIO& io = ImGui::GetIO();
    ImVec4 clear_color = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);

    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Edit scene");

    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Lights")) {
            ImGui::SliderFloat("rotation speed", &rotation_speed, 0.01f, 10.0f);
        }
        ImGui::EndTabBar();
    }
    ImGui::SameLine();

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