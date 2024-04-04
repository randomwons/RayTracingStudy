#ifndef __RENDERER_CUH__
#define __RENDERER_CUH__

#include "common.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

constexpr uint32_t BLOCK_DIM_X = 32;
constexpr uint32_t BLOCK_DIM_Y = 32;

#define XCOORD blockDim.x * blockIdx.x + threadIdx.x;
#define YCOORD blockDim.y * blockIdx.y + threadIdx.y;

struct Ray {

    glm::vec3 o, d;

};

CLASS_PTR(KernelRenderer);
class KernelRenderer {
public:
    KernelRenderer() {}
    ~KernelRenderer();
    KernelRenderer(int width, int height);
    void render();
    void resize(int width, int height);

private:
    Ray* rays;
    // glm::mat3 intrinsic;
    // glm::mat4 extrinsic;

    dim3 gridLayout;
    dim3 blockLayout = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
    uint32_t width, height;
    uint32_t texture;
    uint32_t pbo;
    cudaGraphicsResource_t cudaResource;


};

#endif // __RENDERER_CUH__