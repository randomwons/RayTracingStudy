#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstring>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr const char* WINDOW_TITLE = "CUDA-OpenGL Interop Test";
constexpr uint32_t IMAGE_CHANNEL = 4;

constexpr uint32_t BLOCK_DIM_X = 32;
constexpr uint32_t BLOCK_DIM_Y = 32;

const char* vertexShaderSource = 
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "void main() {\n"
    "   gl_Position = vec4(aPos, 1.0);\n"
    "   TexCoord = aTexCoord;\n"
    "}\0";

const char* fragmentShaderSource = 
    "#version 330 core\n"
    "out vec4 fragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main() {\n"
    "   fragColor = texture(texture1, TexCoord);\n"
    "}\0";

__global__ void generateRandomImage(uchar4* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int pid = y * width + x;
    curandState state;
    curand_init((unsigned long long)clock() + pid, 0, 0, &state);
    data[pid] = make_uchar4(
        curand_uniform(&state) * 255,
        curand_uniform(&state) * 255,
        curand_uniform(&state) * 255,
        255
    );
}

enum Mode {
    CPU,
    CUDA,
    CUDA_OPENGL
};

class Displayer {
public:
    Displayer(Mode mode) : mode(mode) {

        int success;
        uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
        uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if(!success){
            char infoLog[1024];
            glGetShaderInfoLog(vertexShader, 1024, nullptr, infoLog);
            printf("Failed to compile vertex shader\n");
            printf("reason : %s\n", infoLog);
        }
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if(!success){
            char infoLog[1024];
            glGetShaderInfoLog(fragmentShader, 1024, nullptr, infoLog);
            printf("Failed to compile vertex shader\n");
            printf("reason : %s\n", infoLog);
        }
        program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        float vertices[] = {
            // Position // Tex coords
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // left-lower
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, // left-upper
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // right-lower
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f  // right-upper
        };

        uint32_t indices[] = {
            0, 1, 2,
            1, 2, 3
        };

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        uint32_t texture;
        glGenTextures(1, &texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        switch (mode) {
        case Mode::CPU:
            imageCPU = new uchar4[width * height];
            break;
        case Mode::CUDA:
            imageCPU = new uchar4[width * height];
            cudaMalloc((void**)&imageGPU, sizeof(uchar4) * width * height);
            gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);
            break;
        case Mode::CUDA_OPENGL:
            glGenBuffers(1, &pbo);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * IMAGE_CHANNEL, NULL, GL_DYNAMIC_COPY);
            cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);
            gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);
            break;
        }
    }
    ~Displayer() {
        glDeleteBuffers(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteProgram(program);

        switch (mode) {
        case Mode::CPU:
            delete[] imageCPU;
            break;
        case Mode::CUDA:
            delete[] imageCPU;
            cudaFree(imageGPU);
            break;
        case Mode::CUDA_OPENGL:
            glDeleteBuffers(1, &pbo);
            cudaGraphicsUnregisterResource(cudaResource);
            break;
        }

    }
    void display() {

        switch (mode) {
        case Mode::CPU:
            for(int y = 0; y < height; y++){
                for(int x = 0; x < width; x++){
                    int index = width * y + x;
                    imageCPU[index] = make_uchar4(
                        rand() % 256,
                        rand() % 256,
                        rand() % 256,
                        255
                    );
                }
            }
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, imageCPU);
            break;
        case Mode::CUDA:
            generateRandomImage<<<gridLayout, blockLayout>>>(imageGPU, width, height);
            cudaMemcpy(imageCPU, imageGPU, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, imageCPU);
            break;
        case Mode::CUDA_OPENGL:
            if(cudaResource == nullptr) return;
            cudaGraphicsMapResources(1, &cudaResource, 0);
            uchar4 *devPtr;
            size_t size;
            cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
            generateRandomImage<<<gridLayout, blockLayout>>>(devPtr, width, height);
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
            cudaGraphicsUnmapResources(1, &cudaResource, 0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            break;
        }

        glUseProgram(program);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    }

    void resize(int width_, int height_) {

        width = width_;
        height = height_;

        switch (mode) {
        case Mode::CPU:
            if(imageCPU != nullptr) {
                delete[] imageCPU;
                imageCPU = new uchar4[width_ * height_];
            }
            break;
        case Mode::CUDA:
            if(imageCPU != nullptr) {
                delete[] imageCPU;
                imageCPU = new uchar4[width_ * height_];
            }
            if(imageGPU != nullptr) {
                cudaFree(imageGPU);
                gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);
                cudaMalloc((void**)&imageGPU, sizeof(uchar4) * width_ * height_);
            }
            break;
        case Mode::CUDA_OPENGL:
            
            if(cudaResource != nullptr) {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * IMAGE_CHANNEL, NULL, GL_DYNAMIC_COPY);
                gridLayout = dim3(width_ / BLOCK_DIM_X + 1, height_ / BLOCK_DIM_Y + 1);    
                cudaGraphicsUnregisterResource(cudaResource);
                cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);
            }
            break;
        }
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

private:
    uint32_t width { WINDOW_WIDTH };
    uint32_t height { WINDOW_HEIGHT };

    Mode mode;
    uint32_t program;
    uint32_t vao, vbo, ebo;
    uint32_t texture;

    uchar4* imageCPU;
    uchar4* imageGPU;

    dim3 gridLayout;
    dim3 blockLayout = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
    cudaGraphicsResource_t cudaResource;
    uint32_t pbo;

    

};

void framebufferCallback(GLFWwindow* window, int width, int height){
    auto displayer = reinterpret_cast<Displayer*>(glfwGetWindowUserPointer(window));
    glViewport(0, 0, width, height);
    if(displayer) displayer->resize(width, height);

}

int main(int argc, char* argv[]){

    if(argc != 2) {
        printf("Usage : mode <--cpu, --cuda, --cuda-opengl>\n");
        return EXIT_FAILURE;
    }
    
    Mode mode;
    if(strcmp(argv[1], "--cpu") == 0) {
        mode = CPU;
        printf("Mode CPU\n");
    } else if (strcmp(argv[1], "--cuda") == 0) {
        mode = CUDA;
        printf("Mode CUDA\n");
    } else if (strcmp(argv[1], "--cuda-opengl") == 0) {
        mode = CUDA_OPENGL;
        printf("Mode CUDA-OpenGL Interop\n");
    } else {
        printf("Usage : mode <--cpu, --cuda, --cuda-opengl>\n");
        return EXIT_FAILURE;
    }

    if(!glfwInit()){
        printf("Failed to initialize glfw\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if(!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OPENGL CONTEXT VERSION : %s\n", glVersion);

    std::unique_ptr<Displayer> displayer = std::make_unique<Displayer>(mode);
    glfwSetWindowUserPointer(window, displayer.get());
    glfwSetFramebufferSizeCallback(window, framebufferCallback);

    double deltaTime = 0;
    double lastFrameTime = 0;
    double currentFrameTime = 0;
    int frames = 0;
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glDisable(GL_DEPTH_TEST);
    while(!glfwWindowShouldClose(window)){

        currentFrameTime = glfwGetTime();
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        frames++;
        if(frames % 200) {
            printf("[FPS] : %f\n", 1.f / (deltaTime));
        }


        glfwPollEvents();
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        displayer->display();

        glfwSwapBuffers(window);
    }

    glfwTerminate();

    return EXIT_SUCCESS;

}