#ifndef __DISPLAYER_H__
#define __DISPLAYER_H__

#include "common.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

constexpr uint32_t BLOCK_DIM_X = 32;
constexpr uint32_t BLOCK_DIM_Y = 32;

static const char* vertShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
)glsl";

static const char* fragShaderSource = R"glsl(
    #version 330 core
    out vec4 fragColor;
    in vec2 TexCoord;

    uniform sampler2D texture1;

    void main() {
        fragColor = texture(texture1, TexCoord);
    }
)glsl";

class Displayer {
public:
    Displayer(const uint32_t width, const uint32_t height);
    void display();
    void resize(const uint32_t width_, const uint32_t height_);

private:
    uint32_t width;
    uint32_t height;
    uint32_t shaderProgram;
    uint32_t vao;
    uint32_t pbo;
    uint32_t texture;
    cudaGraphicsResource_t cudaResource;

    dim3 gridLayout;
    dim3 blockLayout = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

    

};

#endif // __DISPLAYER_H__