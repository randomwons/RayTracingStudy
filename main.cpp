
#include "window/window.h"


int main() {

    WindowUPtr window = Window::createWindow(1280, 720, "Test");
    if(!window) return EXIT_FAILURE;

    double lastTime = glfwGetTime();
    int frame = 0;
    while(!window->shouldClose()){
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        if(frame % 200) {
            printf("[FPS] : %f\n", 1.f / (deltaTime));
        }

        window->update();
        frame++;
    }

    return EXIT_SUCCESS;
}