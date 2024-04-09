#ifndef __RENDERER_CUH__
#define __RENDERER_CUH__

#include "common.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_free.h>

#include "camera.h"
#include "octree.h"

constexpr uint32_t BLOCK_DIM_X = 32;
constexpr uint32_t BLOCK_DIM_Y = 32;

#define XCOORD blockDim.x * blockIdx.x + threadIdx.x;
#define YCOORD blockDim.y * blockIdx.y + threadIdx.y;

CLASS_PTR(KernelRenderer);
class KernelRenderer {
public:
    KernelRenderer() {}
    ~KernelRenderer();
    KernelRenderer(cudaGraphicsResource_t cudaResource, int width, int height);
    void render();
    void resize(int width, int height);
    void setPosition(glm::mat4 pose);
    void setIntrinsic(glm::mat3 intrinsic);

    void setOctree(glm::vec3 min, glm::vec3 max, float resolution);



    cudaGraphicsResource_t cudaResource;

private:
    thrust::device_ptr<Camera*> d_camera;
    thrust::device_ptr<Octree*> d_octree;

    dim3 gridLayout;
    dim3 blockLayout = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
    uint32_t width, height;


};

#endif // __RENDERER_CUH__