#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr const char* WINDOW_TITLE = "CUDA-OpenGL Interop Test";

// RANDOM IMAGE SIZE
constexpr uint32_t IMAGE_WIDTH = 1920;
constexpr uint32_t IMAGE_HEIGHT = 1080;
constexpr uint32_t IMAGE_CHANNEL = 4;

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

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OPENGL CONTEXT VERSION : %s\n", glVersion);

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
        return EXIT_FAILURE;
    }
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetShaderInfoLog(fragmentShader, 1024, nullptr, infoLog);
        printf("Failed to compile vertex shader\n");
        printf("reason : %s\n", infoLog);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    uint32_t shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
         0.0f,  1.0f, 0.0f, 0.5f, 1.0f
    };

    uint32_t vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    uint32_t texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    uchar4* imageCPU;
    uchar4* imageCUDA;
    cudaGraphicsResource *cudaResource;
    uint32_t pbo;
    
    if(mode == Mode::CPU) {
        imageCPU = new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT];
    } else if (mode == Mode::CUDA) {
        imageCPU = new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT];
        cudaMalloc((void**)&imageCUDA, sizeof(uchar4) * IMAGE_WIDTH * IMAGE_HEIGHT);
    } else if (mode == Mode::CUDA_OPENGL) {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL, NULL, GL_DYNAMIC_COPY);
        cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);
    }

    double deltaTime = 0;
    double lastFrameTime = 0;
    double currentFrameTime = 0;
    int frames = 0;
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

        if(mode == Mode::CPU) {
            for(int y = 0; y < IMAGE_HEIGHT; y++){
                for(int x = 0; x < IMAGE_WIDTH; x++) {
                    int index = IMAGE_WIDTH * y + x;
                    imageCPU[index] = make_uchar4(
                        rand() % 256,
                        rand() % 256,
                        rand() % 256,
                        255
                    );
                }
            }
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, imageCPU);
        } else if (mode == Mode::CUDA) {
            dim3 blocks(IMAGE_WIDTH / 16, IMAGE_HEIGHT / 16);
            dim3 threads(16, 16);
            generateRandomImage<<<blocks, threads>>>(imageCUDA, IMAGE_WIDTH, IMAGE_HEIGHT);
            cudaMemcpy(imageCPU, imageCUDA, sizeof(uchar4) * IMAGE_WIDTH * IMAGE_HEIGHT, cudaMemcpyDeviceToHost);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, imageCPU);
        } else if (mode == Mode::CUDA_OPENGL) {
            cudaGraphicsMapResources(1, &cudaResource, 0);
            uchar4 *devPtr;
            size_t size;
            cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
            dim3 blocks(IMAGE_WIDTH / 16, IMAGE_HEIGHT / 16);
            dim3 threads(16, 16);
            generateRandomImage<<<blocks, threads>>>(devPtr, IMAGE_WIDTH, IMAGE_HEIGHT);
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
            cudaGraphicsUnmapResources(1, &cudaResource, 0);

            glBindTexture(GL_TEXTURE_2D, texture);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
    }

    if(mode == Mode::CPU) {
        delete[] imageCPU;
    } else if (mode == Mode::CUDA) {
        delete[] imageCPU;
        cudaFree(imageCUDA);
    } else if (mode == Mode::CUDA_OPENGL) {
        glDeleteBuffers(1, &pbo);
        cudaGraphicsUnregisterResource(cudaResource);
    }


    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &vao);

    glfwTerminate();

    return EXIT_SUCCESS;

}