#include "window/displayer.h"
#include <curand_kernel.h>

#define XCOORD blockDim.x * blockIdx.x + threadIdx.x;
#define YCOORD blockDim.y * blockIdx.y + threadIdx.y;

__device__ bool hit_sphere(glm::vec3& center, double radius, const Ray& r) {

    glm::vec3 oc = r.o - center;
    auto a = glm::dot(r.d, r.d);
    auto b = 2.0 * glm::dot(oc, r.d);
    auto c = glm::dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    return (discriminant >= 0);

}


__global__ void generateRays(Ray* rays, int width, int height, glm::mat3* intrinsic, glm::mat4* extrinsic) {

    int x = XCOORD;
    int y = YCOORD;
    if(x >= width || y >= height) return;

    glm::mat3 intrinsic_ = intrinsic[0];
    glm::mat4 extrinsic_ = extrinsic[0];

    float fx = intrinsic_[0][0];
    float fy = intrinsic_[1][1];
    float cx = intrinsic_[0][2];
    float cy = intrinsic_[1][2];

    float dx = (x - cx) / fx;
    float dy = (y - cy) / fy;
    float dz = 1.0;
    float length = sqrt(dx * dx + dy * dy + dz * dz);
    dx /= length;
    dy /= length;
    dz /= length;
    
    float worldDx = extrinsic_[0][0] * dx + extrinsic_[1][0] * dy + extrinsic_[2][0] * dz;
    float worldDy = extrinsic_[0][1] * dx + extrinsic_[1][1] * dy + extrinsic_[2][1] * dz;
    float worldDz = extrinsic_[0][2] * dx + extrinsic_[1][2] * dy + extrinsic_[2][2] * dz;

    float worldOx = extrinsic_[3][0];
    float worldOy = extrinsic_[3][1];
    float worldOz = extrinsic_[3][2];

    int index = y * width + x;
    rays[index].o = glm::vec3(worldOx, worldOy, worldOz);
    rays[index].d = glm::vec3(worldDx, worldDy, worldDz);

}

__global__ void raytracing(Ray* rays, uchar4* data, int width, int height) {

    int x = XCOORD;
    int y = YCOORD;
    if(x >= width || y >= height) return;

    int pid = y * width + x;
    if(hit_sphere(glm::vec3(0, 0, -1), 0.5, rays[pid])) {
        data[pid] = make_uchar4(100, 200, 10, 255);
        return;
    }
    if(hit_sphere(glm::vec3(1, 0, -1), 0.5, rays[pid])) {
        data[pid] = make_uchar4(100, 200, 10, 255);
        return;
    }

    data[pid].x = static_cast<unsigned char>(__saturatef(rays[pid].d.x) * 255.0f);
    data[pid].y = static_cast<unsigned char>(__saturatef(rays[pid].d.y) * 255.0f);
    data[pid].z = static_cast<unsigned char>(__saturatef(rays[pid].d.z) * 255.0f);
    data[pid].w = 255;
}

__global__ void generateRandomImage(uchar4* data, int width, int height) {

    int x = XCOORD;
    int y = YCOORD;
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
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // left-lower
        -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, // left-upper
         1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // right-lower
         1.0f,  1.0f, 0.0f, 1.0f, 0.0f  // right-upper
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

    intrinsic = glm::mat3(0.0f);
    intrinsic[0][0] = 1000.;
    intrinsic[1][1] = 1000.;
    intrinsic[0][2] = width / 2.;
    intrinsic[1][2] = height / 2.;

    cudaMalloc((void**)&rays, sizeof(Ray) * width * height);
    cudaMalloc((void**)&d_intrinsic, sizeof(glm::mat3));
    cudaMalloc((void**)&d_extrinsic, sizeof(float) * 16);
    
}

void Displayer::display() {

    m_cameraFront = 
        glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraYaw), glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraPitch), glm::vec3(1.0f, 0.0f, 0.0f)) *
        glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);

    view = glm::lookAt(
        m_cameraPos,
        m_cameraPos + m_cameraFront,
        m_cameraUp);

    view = glm::inverse(view) * glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    cudaMemcpy(d_intrinsic, (void*)&intrinsic, sizeof(glm::mat3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_extrinsic, (void*)&view, sizeof(glm::mat4), cudaMemcpyHostToDevice);

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

    intrinsic[0][2] = width_ / 2.;
    intrinsic[1][2] = height_ / 2.;

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