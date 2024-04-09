#include "renderer.cuh"

__device__ bool hit_sphere(glm::vec3& center, double radius, const Ray& r) {

    // glm::vec3 oc = r.origin - center;
    // auto a = glm::dot(r.direction, r.direction);
    // auto b = 2.0 * glm::dot(oc, r.direction);
    // auto c = glm::dot(oc, oc) - radius*radius;
    // auto discriminant = b*b - 4*a*c;
    // return (discriminant >= 0);

    double rayorix;
    double rayoriy;
    double rayoriz;

    double rayInvDirx;
    double rayInvDiry;
    double rayInvDirz;
    
    glm::vec3 min(0, 0, 0);
    glm::vec3 max(1.28, 1.28, 1.28);

    if(r.direction.x < 0.0f){
        rayorix = center.x * 2.0f - r.origin.x;
        rayInvDirx = -(1 / r.direction.x);
    } else {
        rayorix = r.origin.x;
        rayInvDirx = 1 / r.direction.x; 
    }
    if(r.direction.y < 0.0f){
        rayoriy = center.y * 2.0f - r.origin.y;
        rayInvDiry = -(1 / r.direction.y);
    } else {
        rayoriy = r.origin.y;
        rayInvDiry = 1 / r.direction.y; 
    }
    if(r.direction.z < 0.0f){
        rayoriz = center.z * 2.0f - r.origin.z;
        rayInvDirz = -(1 / r.direction.z);
    } else {
        rayoriz = r.origin.z;
        rayInvDirz = 1 / r.direction.z; 
    }

    const float tx0 = (min.x - rayorix) * rayInvDirx;
    const float tx1 = (max.x - rayorix) * rayInvDirx;
    const float ty0 = (min.y - rayoriy) * rayInvDiry;
    const float ty1 = (max.y - rayoriy) * rayInvDiry;
    const float tz0 = (min.z - rayoriz) * rayInvDirz;
    const float tz1 = (max.z - rayoriz) * rayInvDirz;

    if(fmaxf(fmaxf(tx0, ty0), tz0) < fminf(fminf(tx1, ty1), tz1)) return true;
    return false;

}

__global__ void raytracing(thrust::device_ptr<Camera*> camera, thrust::device_ptr<Octree*> octree, uchar4* data, int width, int height) {
    
    int x = XCOORD;
    int y = YCOORD;
    if(x >= width || y >= height) return;

    int pid = y * width + x;

    Ray ray = ((Camera*)(*camera))->getRay(x, y);
    double value = ((Octree*)(*octree))->traverse(ray);

    data[pid].x = (unsigned char)value;
    // value = octree.traverse(ray, ..)
    if(hit_sphere(glm::vec3(0.64, 0.64, 0.64), 0.5, ray)) {
        data[pid].x = 255;
        data[pid].y = 255;
        data[pid].z = 255;
        data[pid].w = 255;
        return;
    }
    
    // data[pid].x = static_cast<unsigned char>(__saturatef(ray.direction.x) * 255.0f);
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

__global__ void setOctreeKernel(thrust::device_ptr<Octree*> octree, glm::vec3 min, glm::vec3 max, float resolution) {
    if(threadIdx.x != 0) return;

    *octree = new Octree(min, max, resolution);
}

void KernelRenderer::setPosition(glm::mat4 pose) {
    setCameraPosition<<<1, 1>>>(d_camera, pose);
}

void KernelRenderer::setIntrinsic(glm::mat3 intrinsic) {
    setCameraIntrinsic<<<1, 1>>>(d_camera, intrinsic);
}

// void KernelRenderer::setOctree(glm::vec3 min, glm::vec3 max, float resolution) {
//     setOctreeKernel<<<1, 1>>>(d_octree, min, max, resolution);
// }


KernelRenderer::KernelRenderer(cudaGraphicsResource_t cudaResource, int width, int height) 
    : cudaResource(cudaResource), width(width), height(height) {

    gridLayout = dim3(width / BLOCK_DIM_X + 1, height / BLOCK_DIM_Y + 1);

    d_camera = thrust::device_new<Camera*>();
    setCamera<<<1, 1>>>(d_camera);

    d_octree = thrust::device_new<Octree*>();

    glm::vec3 min = glm::vec3(0.0f);
    glm::vec3 max = glm::vec3(1.28f);
    float resolution = 0.01;

    setOctreeKernel<<<1, 1>>>(d_octree, min, max, resolution); 


}

void KernelRenderer::render() {

    cudaGraphicsMapResources(1, &cudaResource, 0);
    uchar4 *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);
    raytracing<<<gridLayout, blockLayout>>>(d_camera, d_octree, devPtr, width, height);
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