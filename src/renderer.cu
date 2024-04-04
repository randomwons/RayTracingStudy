#include "renderer.cuh"

// __device__ bool hit_sphere(glm::vec3& center, double radius, const Ray& r) {

//     glm::vec3 oc = r.o - center;
//     auto a = glm::dot(r.d, r.d);
//     auto b = 2.0 * glm::dot(oc, r.d);
//     auto c = glm::dot(oc, oc) - radius*radius;
//     auto discriminant = b*b - 4*a*c;
//     return (discriminant >= 0);

// }

// __global__ void generateRays(Ray* rays, int width, int height, glm::mat3* intrinsic, glm::mat4* extrinsic) {

//     int x = XCOORD;
//     int y = YCOORD;
//     if(x >= width || y >= height) return;

//     glm::mat3 intrinsic_ = intrinsic[0];
//     glm::mat4 extrinsic_ = extrinsic[0];

//     float fx = intrinsic_[0][0];
//     float fy = intrinsic_[1][1];
//     float cx = intrinsic_[0][2];
//     float cy = intrinsic_[1][2];

//     float dx = (x - cx) / fx;
//     float dy = (y - cy) / fy;
//     float dz = 1.0;
//     float length = sqrt(dx * dx + dy * dy + dz * dz);
//     dx /= length;
//     dy /= length;
//     dz /= length;
    
//     float worldDx = extrinsic_[0][0] * dx + extrinsic_[1][0] * dy + extrinsic_[2][0] * dz;
//     float worldDy = extrinsic_[0][1] * dx + extrinsic_[1][1] * dy + extrinsic_[2][1] * dz;
//     float worldDz = extrinsic_[0][2] * dx + extrinsic_[1][2] * dy + extrinsic_[2][2] * dz;

//     float worldOx = extrinsic_[3][0];
//     float worldOy = extrinsic_[3][1];
//     float worldOz = extrinsic_[3][2];

//     int index = y * width + x;
//     rays[index].o = glm::vec3(worldOx, worldOy, worldOz);
//     rays[index].d = glm::vec3(worldDx, worldDy, worldDz);

// }

// __global__ void raytracing(Ray* rays, uchar4* data, int width, int height) {

//     int x = XCOORD;
//     int y = YCOORD;
//     if(x >= width || y >= height) return;

//     int pid = y * width + x;
//     if(hit_sphere(glm::vec3(0, 0, -1), 0.5, rays[pid])) {
//         data[pid] = make_uchar4(100, 200, 10, 255);
//         return;
//     }
//     if(hit_sphere(glm::vec3(1, 0, -1), 0.5, rays[pid])) {
//         data[pid] = make_uchar4(100, 200, 10, 255);
//         return;
//     }

//     data[pid].x = static_cast<unsigned char>(__saturatef(rays[pid].d.x) * 255.0f);
//     data[pid].y = static_cast<unsigned char>(__saturatef(rays[pid].d.y) * 255.0f);
//     data[pid].z = static_cast<unsigned char>(__saturatef(rays[pid].d.z) * 255.0f);
//     data[pid].w = 255;
// }

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

KernelRenderer::KernelRenderer(int width, int height) : width(width), height(height) {

    gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);

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
    cudaMalloc((void**)&rays, sizeof(Ray) * width * height);

}

void KernelRenderer::render() {

    cudaGraphicsMapResources(1, &cudaResource, 0);
    uchar4 *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
    generateRandomImage<<<gridLayout, blockLayout>>>(devPtr, width, height);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

}

void KernelRenderer::resize(int width_, int height_) {

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
    if(rays != nullptr){
        cudaFree(rays);
        cudaMalloc((void**)&rays, sizeof(Ray) * width * height);
    }

}

KernelRenderer::~KernelRenderer() {
    if(!cudaResource){
        cudaGraphicsUnregisterResource(cudaResource);
    }
    if(!rays) {
        cudaFree(rays);
    }
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
}