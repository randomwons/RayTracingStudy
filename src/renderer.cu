#include "renderer.cuh"

__device__ bool hit_sphere(glm::vec3& center, double radius, const Ray& r) {

    glm::vec3 oc = r.origin - center;
    auto a = glm::dot(r.direction, r.direction);
    auto b = 2.0 * glm::dot(oc, r.direction);
    auto c = glm::dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    return (discriminant >= 0);

}

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

__global__ void raytracing(thrust::device_ptr<Camera*> camera, uchar4* data, int width, int height) {

    int x = XCOORD;
    int y = YCOORD;
    if(x >= width || y >= height) return;

    int pid = y * width + x;

    Ray ray = ((Camera*)(*camera))->getRay(x, y);
    if(hit_sphere(glm::vec3(0, 0, -2), 0.5, ray)) {
        data[pid].x = 255;
        data[pid].y = 0;
        data[pid].z = 0;
        data[pid].w = 255;
        return;
    }

    data[pid].x = static_cast<unsigned char>(__saturatef(ray.direction.x) * 255.0f);
    data[pid].y = static_cast<unsigned char>(__saturatef(ray.direction.y) * 255.0f);
    data[pid].z = static_cast<unsigned char>(__saturatef(ray.direction.z) * 255.0f);
    data[pid].w = 255;
}

__global__ void setCamera(thrust::device_ptr<Camera*> camera) {
    if(threadIdx.x != 0) return;

    glm::mat3 intrinsic = glm::mat3(1000., 0, 640, 0, 1000, 340, 0, 0, 1);

    *camera = new Camera(intrinsic, glm::mat4(1.0f));

}

__global__ void setCameraPosition(thrust::device_ptr<Camera*> camera, glm::mat4 pose) {
    if(threadIdx.x != 0) return;
    
    ((Camera*)(*camera))->setPosition(pose);   
}

__global__ void setCameraIntrinsic(thrust::device_ptr<Camera*> camera, glm::mat3 intrinsic) {
    if(threadIdx.x != 0) return;

    ((Camera*)(*camera))->setIntrinsic(intrinsic);
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

void KernelRenderer::setPosition(glm::mat4 pose) {
    setCameraPosition<<<1, 1>>>(d_camera, pose);
}

void KernelRenderer::setIntrinsic(glm::mat3 intrinsic) {
    setCameraIntrinsic<<<1, 1>>>(d_camera, intrinsic);
}


KernelRenderer::KernelRenderer(cudaGraphicsResource_t cudaResource, int width, int height) 
    : cudaResource(cudaResource), width(width), height(height) {

    gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);

    d_camera = thrust::device_new<Camera*>();
    setCamera<<<1, 1>>>(d_camera);

}

void KernelRenderer::render() {

    cudaGraphicsMapResources(1, &cudaResource, 0);
    uchar4 *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
    raytracing<<<gridLayout, blockLayout>>>(d_camera, devPtr, width, height);
    // generateRandomImage<<<gridLayout, blockLayout>>>(devPtr, width, height);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

}

void KernelRenderer::resize(int width_, int height_) {

    width = width_;
    height = height_;
    gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);
    

    glm::mat3 intrinsic = glm::mat3(1.0f);

    double f = width / (2 * glm::tan(glm::radians(80.f) / 2));

    intrinsic[0][0] = f;
    intrinsic[1][1] = f;
    intrinsic[0][2] = width / 2;
    intrinsic[1][2] = height / 2;
    setIntrinsic(intrinsic);

    // if(cudaResource != nullptr) {
    //     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    //     glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * 4, NULL, GL_DYNAMIC_COPY);
    //     gridLayout = dim3(width_ / BLOCK_DIM_X + 1, height_ / BLOCK_DIM_Y + 1);    
    //     cudaGraphicsUnregisterResource(cudaResource);
    //     cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone); 
    //     glBindTexture(GL_TEXTURE_2D, texture);
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    // }
    // if(rays != nullptr){
    //     thrust::device_free(rays);
    //     rays = thrust::device_new<Ray*>(width * height);
    //     // cudaMalloc((void**)&rays, sizeof(Ray) * width * height);
    // }

}

KernelRenderer::~KernelRenderer() {
    // if(!cudaResource){
    //     cudaGraphicsUnregisterResource(cudaResource);
    // }
    // if(!rays) {
    //     thrust::device_free(rays);
    // }
    // glDeleteBuffers(1, &pbo);
    // glDeleteTextures(1, &texture);
}