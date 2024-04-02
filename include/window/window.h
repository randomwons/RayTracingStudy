#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "common.h"
#include "window/displayer.h"

enum Status {
    SUCCESS,
    FAIL_GLFW,
    FAIL_GLAD,
    FAIL_WINDOW
};

CLASS_PTR(Window);
class Window { 
public:
    static WindowUPtr createWindow(const uint32_t width, const uint32_t height, const char* title);
    ~Window() { glfwTerminate(); }
    bool shouldClose() const { return glfwWindowShouldClose(window); }
    void update();
    void resize(const uint32_t width, const uint32_t height);

private:
    Window() {}
    Status create(const uint32_t width, const uint32_t height, const char* title);  

    std::unique_ptr<Displayer> displayer;
    GLFWwindow* window;
    // uint32_t width, height;

};

#endif // __WINDOW_H__