#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

// #define CPU
#define GPU


constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr uint32_t IMAGE_WIDTH = 1920;
constexpr uint32_t IMAGE_HEIGHT = 1080;

constexpr const char* WINDOW_TITLE = "Learn OpenGL";

__global__ void generateRandomImage(uchar4 *image, uint32_t width, uint32_t height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        
        int pid = y * width + x;
        curandState state;
        curand_init((unsigned long long)clock() + pid, 0, 0, &state);


        unsigned char r = curand_uniform(&state) * 255;
        unsigned char g = curand_uniform(&state) * 255;
        unsigned char b = curand_uniform(&state) * 255;
        image[pid] = make_uchar4(r, g, b, 255);
    }

}

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


int main(){
    // cudaGLSetGLDevice(1);

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

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL Context Verion : %s\n", reinterpret_cast<const char*>(glVersion));

    int success = 0;

    uint32_t vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertShader);

    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetShaderInfoLog(vertShader, 1024, nullptr, infoLog);
        printf("Failed to compile vertex shader\n");
        printf("reason : %s\n", infoLog);
        return EXIT_FAILURE;
    }

    uint32_t fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragShader);

    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetShaderInfoLog(fragShader, 1024, nullptr, infoLog);
        printf("Failed to compile vertex shader\n");
        printf("reason : %s\n", infoLog);

        glfwTerminate();
        return EXIT_FAILURE;
    }

    uint32_t shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        char infoLog[1024];
        glGetProgramInfoLog(shaderProgram, 1024, nullptr, infoLog);
        printf("Failed to link shader program\n");
        printf("reason : %s\n", infoLog);

        glfwTerminate();
        return EXIT_FAILURE;
    }


    float vertices[] = {
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
         0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
         0.0f,  0.5f, 0.0f, 0.5f, 1.0f,
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

#ifdef GPU
    cudaGraphicsResource *cudaResource;
    uint32_t pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, IMAGE_WIDTH * IMAGE_HEIGHT * 4, NULL, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);
#endif

    int nbFrames = 0;
    double lastTime = glfwGetTime();

#ifdef CPU
    unsigned char* imageData = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
#endif

    while(!glfwWindowShouldClose(window)){
        nbFrames++;

        double currentTime = glfwGetTime();
        if(nbFrames > 10){
            printf("FPS : %f\n", (float)nbFrames / (currentTime - lastTime));
            nbFrames = 0;
        }

        glfwPollEvents();
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

#ifdef CPU
        for(int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * 3; i++){
            imageData[i] = rand() % 255;
        }
#endif

#ifdef GPU
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
#endif

#ifdef CPU
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, imageData);
#endif

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);

        lastTime = currentTime;
    }

#ifdef CPU
    delete[] imageData;
#endif CPU

    glfwTerminate();
    return EXIT_SUCCESS;

}