#ifndef __RAY_H__
#define __RAY_H__

#include "device_launch_parameters.h"
#include <glm/vec3.hpp>

class Ray {
public:
    __device__ __host__ Ray() {}
    __device__ __host__ Ray(const glm::vec3& origin_, const glm::vec3& direction_) {
        origin = origin_;
        direction = direction_;
    }
    glm::vec3 origin;
    glm::vec3 direction;
};

#endif // __RAY_H__