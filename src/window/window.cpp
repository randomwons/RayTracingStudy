#include "window/window.h"

void framebuffer_callback(GLFWwindow* window, int width, int height){
    auto window_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if(window_) window_->resize(width, height);
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
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_callback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        return FAIL_GLAD;
    }

    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL Context Version : %s\n", glVersion);

    glfwSwapInterval(0);

    displayer = std::make_unique<Displayer>(width, height);

    return Status::SUCCESS;
}

void Window::update() {
    glfwPollEvents();

    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    displayer->display();

    glfwSwapBuffers(window);
}

void Window::resize(const uint32_t width_, const uint32_t height_) {

    glViewport(0, 0, width_, height_);
    if(displayer) displayer->resize(width_, height_);

}