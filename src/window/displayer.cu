#include "window/displayer.h"

#include <curand_kernel.h>


__global__ void generateRays(Ray* rays, int width, int height, float* intrinsic, float* extrinsic) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    float fx = intrinsic[0];
    float fy = intrinsic[4];
    float cx = intrinsic[2];
    float cy = intrinsic[5];

    float dx = (x - cx) / fx;
    float dy = (y - cy) / fy;
    float dz = 1.0;

    float worldDx = extrinsic[0] * dx + extrinsic[1] * dy + extrinsic[2] * dz + extrinsic[3];
    float worldDy = extrinsic[4] * dx + extrinsic[5] * dy + extrinsic[6] * dz + extrinsic[7];
    float worldDz = extrinsic[8] * dx + extrinsic[9] * dy + extrinsic[10] * dz + extrinsic[11];

    float worldOx = extrinsic[3];
    float worldOy = extrinsic[7];
    float worldOz = extrinsic[11];

    int index = y * width + x;
    rays[index].o = make_float3(worldOx, worldOy, worldOz);
    rays[index].d = make_float3(worldDx, worldDy, worldDz);

}

__global__ void raytracing(Ray* rays, uchar4* data, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;

    int pid = y * width + x;
    float t = 0.5f * (rays[pid].d.y + 1.0f);

    data[pid].x = static_cast<unsigned char>(__saturatef(1.0f - t + 0.5 * t) * 255.0f);
    data[pid].y = static_cast<unsigned char>(__saturatef(1.0f - t + 0.7 * t) * 255.0f);
    data[pid].z = static_cast<unsigned char>(__saturatef(1.0f - t + 1.0 * t) * 255.0f);
    data[pid].w = 255;
}

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

Displayer::Displayer(const uint32_t width, const uint32_t height) : width(width), height(height) {

    gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);

    int success;
    uint32_t vertShader = glCreateShader(GL_VERTEX_SHADER);
    uint32_t fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertShader, 1, &vertShaderSource, nullptr);
    glCompileShader(vertShader);
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetShaderInfoLog(vertShader, 1024, nullptr, infoLog);
        printf("Failed to compile vertex shader\n");
        printf("reason : %s\n", infoLog);
    }
    glShaderSource(fragShader, 1, &fragShaderSource, nullptr);
    glCompileShader(fragShader);
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if(!success){
        char infoLog[1024];
        glGetShaderInfoLog(fragShader, 1024, nullptr, infoLog);
        printf("Failed to compile vertex shader\n");
        printf("reason : %s\n", infoLog);
    }
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

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

    uint32_t vbo, ebo;

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
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    intrinsic[0] = 1000.;
    intrinsic[1] = 0;
    intrinsic[2] = width / 2.;
    intrinsic[3] = 0;
    intrinsic[4] = 1000.;
    intrinsic[5] = height / 2.;
    intrinsic[6] = 0;
    intrinsic[7] = 0;
    intrinsic[8] = 1;

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            if (i == j) extrinsic[i * 4 + j] = 1.;
            else extrinsic[i * 4 + j] = 0.;
        }
    }
    cudaMalloc((void**)&rays, sizeof(Ray) * width * height);
    cudaMalloc((void**)&d_intrinsic, sizeof(float) * 9);
    cudaMalloc((void**)&d_extrinsic, sizeof(float) * 16);
    cudaMemcpy(d_intrinsic, intrinsic, sizeof(float) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_extrinsic, extrinsic, sizeof(float) * 16, cudaMemcpyHostToDevice);
}

void Displayer::display() {

    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    cudaGraphicsMapResources(1, &cudaResource, 0);
    uchar4 *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
    generateRays<<<gridLayout, blockLayout>>>(rays, width, height, d_intrinsic, d_extrinsic);
    raytracing<<<gridLayout, blockLayout>>>(rays, devPtr, width, height);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Displayer::resize(const uint32_t width_, const uint32_t height_) {

    width = width_;
    height = height_;

    if(cudaResource != nullptr) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * 4, NULL, GL_DYNAMIC_COPY);
        gridLayout = dim3(width_ / BLOCK_DIM_X + 1, height_ / BLOCK_DIM_Y + 1);    
        cudaGraphicsUnregisterResource(cudaResource);
        cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone); 
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }
    
    if(rays != nullptr) {
        cudaFree(rays);
        cudaMalloc((void**)&rays, sizeof(Ray) * width_ * height_);
    }
    
}

Displayer::~Displayer() {

    if(cudaResource != nullptr) cudaGraphicsUnregisterResource(cudaResource);
    glDeleteBuffers(1, &pbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &texture);
    if(rays != nullptr) cudaFree(rays);
    if(!d_intrinsic) cudaFree(d_intrinsic);
    if(!d_extrinsic) cudaFree(d_extrinsic);
    // if(!frame) cudaFree(frame);
    

}