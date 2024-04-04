#include "window/window.h"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <imgui.h>

void framebuffer_callback(GLFWwindow* window, int width, int height){
    auto window_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if(window_) window_->resize(width, height);
}

void OnCursorPos(GLFWwindow* window, double x, double y) {
    auto window_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    window_->mouseMove(x, y);
}

void OnMouseButton(GLFWwindow* window, int button, int action, int modifier) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, modifier);
    auto window_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    window_->mouseButton(button, action, x, y);
}

void OnCharEvent(GLFWwindow* window, uint32_t ch) {
    ImGui_ImplGlfw_CharCallback(window, ch);
}

void OnScroll(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

void OnKeyEvent(GLFWwindow* window,
                int key, int scancode, int action, int mods) {

    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

WindowUPtr Window::createWindow(const uint32_t width, const uint32_t height, const char* title) {    
    auto window_ = WindowUPtr(new Window());
    Status status = window_->create(width, height, title);
    switch (status) {
    case Status::FAIL_GLFW:
        printf("Failed to initialie glfw\n");
        return nullptr;
    case Status::FAIL_GLAD:
        printf("Failed to initialize glad\n");
        return nullptr;
    case Status::FAIL_WINDOW:
        printf("Failed to create window\n");
        return nullptr;
    }
    return std::move(window_);
}

Status Window::create(const uint32_t width, const uint32_t height, const char* title) {

    if(!glfwInit()) return Status::FAIL_GLFW;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if(!window) {
        glfwTerminate();
        return Status::FAIL_WINDOW;
    }
    glfwMakeContextCurrent(window);
    
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        return FAIL_GLAD;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL Context Version : %s\n", glVersion);

    initImGui();

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_callback);
    glfwSetKeyCallback(window, OnKeyEvent);
    glfwSetCursorPosCallback(window, OnCursorPos);
    glfwSetCharCallback(window, OnCharEvent);
    glfwSetScrollCallback(window, OnScroll);
    glfwSetMouseButtonCallback(window, OnMouseButton);
    glfwSwapInterval(0);

    displayer = std::make_unique<Displayer>(width, height);

    return Status::SUCCESS;
}

void Window::update() {

    timeUpdate();
    beginFrame();
    displayer->processInput(window);
    displayer->display();
    renderImGui();
    glfwSwapBuffers(window);
    n_frames++;
    
}

void Window::resize(const uint32_t width_, const uint32_t height_) {

    glViewport(0, 0, width_, height_);
    if(displayer) displayer->resize(width_, height_);

}

Window::~Window() {
    
    ImGui_ImplOpenGL3_DestroyDeviceObjects();
    ImGui_ImplOpenGL3_DestroyFontsTexture();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    glfwTerminate();

}

void Window::beginFrame() {
    glfwPollEvents();
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Window::renderImGui() {

    if(ImGui::Begin("ui window")) {
        ImGui::Text("Elapsed Time %f", elapsedTime);
        ImGui::Text("FPS %f", FPS);
        ImGui::Text("frames %d", n_frames);
        ImGui::DragFloat3("camera pos", glm::value_ptr(displayer->m_cameraPos), 0.01f);
        ImGui::DragFloat("camera yaw", &displayer->m_cameraYaw, 0.5f);
        ImGui::DragFloat("camera pitch", &displayer->m_cameraPitch, 0.5f, -89.0f, 89.0f);
    }
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::initImGui() {

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init();
    ImGui_ImplOpenGL3_CreateFontsTexture();
    ImGui_ImplOpenGL3_CreateDeviceObjects();

}

void Window::timeUpdate() {

    elapsedTime = glfwGetTime();
    deltaTime = elapsedTime - lastTime;
    lastTime = elapsedTime;
    FPS = 1.f / deltaTime;

}