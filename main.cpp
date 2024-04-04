
#include "window/window.h"

int main() {

    WindowUPtr window = Window::createWindow(1280, 720, "Ray Tracing");
    if(!window) return EXIT_FAILURE;

    while(!window->shouldClose()){
        window->update();
    }

    return EXIT_SUCCESS;
}