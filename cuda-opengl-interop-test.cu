#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr const char* WINDOW_TITLE = "CUDA-OpenGL Interop Test";

// RANDOM IMAGE
constexpr uint32_t IMAGE_WIDTH = 1280;
constexpr uint32_t IMAGE_HEIGHT = 720;

const char* vertexShaderSource = 
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "void main() {\n"
    "   gl_Position = vec4(aPos, 1);\n"
    "   TexCoord = aTexCoord;\n"
    "}\0";

const char* fragmentShaderSource =
    "#version 330 core\n"
    "out vec4 fragColor;"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main() {\n"
    "   fragColor = texture(texture1, TexCoord);\n"
    "}\0";


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
    } else if (strcmp(argv[1], "--cuda") == 0) {
        mode = CUDA;
    } else if (strcmp(argv[1], "--cuda-opengl") == 0) {
        mode = CUDA_OPENGL;
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

    uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
    uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    uint32_t shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
         0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
         0.0f,  0.5f, 0.0f, 0.5f, 1.0f
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

    uint8_t* imageCPU;
    uchar4* imageCUDA;
    cudaGraphicsResource_t cudaResource;
    uint32_t pbo;
    
    if(mode == Mode::CPU) {
        imageCPU = new uint8_t[IMAGE_WIDTH * IMAGE_HEIGHT * 4];
    } else if (mode == Mode::CUDA) {

    } else if 









    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
    }

    if(mode == )


    glfwTerminate();

    return EXIT_SUCCESS;

}