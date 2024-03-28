#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

constexpr GLuint WINDOW_WIDTH = 1920;
constexpr GLuint WINDOW_HEIGHT = 1080;
constexpr const char* WINDOW_TITLE = "Ray Tracing Study";

struct Ray {
    float3 origin, dir;
};

__device__ float3 normalize(float3 v){
    float invLen = rsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ float3 color(const Ray& r) {
    float t = 0.5f*(r.dir.y + 1.0f);
    return make_float3(1.0f-t + 0.5 * t, 1.0f-t + 0.7 * t, 1.0f-t + 1.0 * t);
}

__global__ void generateRays(Ray* rays, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int pid = y * width + x;

        float normX = (x - width / 2.0f);
        float normY = (y - height / 2.0f);

        Ray ray;
        ray.origin = {0, 0, 0};
        ray.dir = {normX, normY, 1.0f};
        ray.dir = normalize(ray.dir);
        rays[pid] = ray;

    }

}

__global__ void traceRays(Ray* rays, float3* frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height) {
        int pid = y * width + x;
        frame[pid] = color(rays[pid]);
    }
}


__global__ void FrameKernel(uchar3* result, float3* frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int pid = y * width + x;
        result[pid] = {
            static_cast<unsigned char>(__saturatef(frame[pid].x) * 255.0f),
            static_cast<unsigned char>(__saturatef(frame[pid].y) * 255.0f),
            static_cast<unsigned char>(__saturatef(frame[pid].z) * 255.0f)
        };
    }
}

int main(){

    if(!glfwInit()){
        const char* desc = nullptr;
        glfwGetError(&desc);
        printf("Faeild to initialize glfw : %s\n", desc);
        return EXIT_FAILURE;
    }

    // TODO: I don't know why it doesn't work
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    auto window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if(!window) {
        printf("Failed to create glfw window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        printf("Failed to initialize glad\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    auto glVersion = glGetString(GL_VERSION);
    printf("OpenGL Context Version : %s\n", reinterpret_cast<const char*>(glVersion));

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


    uint32_t threadLayoutX = 2;
    uint32_t threadLayoutY = 16;
    uint32_t blockLayoutX = (WINDOW_WIDTH - threadLayoutX + 1) / threadLayoutX;
    uint32_t blockLayoutY = (WINDOW_HEIGHT - threadLayoutY + 1) / threadLayoutY;
    dim3 threadLayout = dim3(threadLayoutX, threadLayoutY);
    dim3 blockLayout = dim3(blockLayoutX, blockLayoutY);

    GLuint texture;
    GLuint pbo;
    cudaGLSetGLDevice(0);
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    size_t size = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);


    uchar3* d_pixelBuffer;
    cudaMalloc(&d_pixelBuffer, sizeof(uchar3) * WINDOW_WIDTH * WINDOW_HEIGHT);

    Ray* rays;
    cudaMalloc(&rays, sizeof(Ray) * WINDOW_WIDTH * WINDOW_HEIGHT);

    float3* d_frame;
    cudaMalloc(&d_frame, sizeof(float3) * WINDOW_WIDTH * WINDOW_HEIGHT);

    while(!glfwWindowShouldClose(window)) {
        
        generateRays<<<blockLayout, threadLayout>>>(rays, WINDOW_WIDTH, WINDOW_HEIGHT);
        traceRays<<<blockLayout, threadLayout>>>(rays, d_frame, WINDOW_WIDTH, WINDOW_HEIGHT);
        FrameKernel<<<blockLayout, threadLayout>>>(d_pixelBuffer, d_frame, WINDOW_WIDTH, WINDOW_HEIGHT);
    
        void *d_ptr = nullptr;
        cudaGLMapBufferObject((void**)&d_ptr, pbo);
        cudaMemcpy2D(d_ptr, WINDOW_WIDTH * 3, (void*)d_pixelBuffer, WINDOW_WIDTH * 3, WINDOW_WIDTH * 3, WINDOW_HEIGHT, cudaMemcpyDeviceToDevice);
        cudaGLUnmapBufferObject(pbo);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 0.0f);
        glEnd();

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    cudaFree(d_pixelBuffer);

    //

    glfwTerminate();

    return EXIT_SUCCESS;

}