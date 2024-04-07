#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "common.h"
#include "ray.h"


// Define pinhole camera
class Camera {
public:
    
    glm::vec3 origin;
    glm::mat3 rot;
    glm::mat3 intrinsic;
    
    __device__ Camera(glm::mat3 intrinsic_, glm::mat4 pose_) {

        intrinsic = intrinsic_;
        origin = glm::vec3(pose_[3]);
        rot = glm::mat3(pose_);

    }

    __device__ Ray getRay(uint32_t u, uint32_t v) {
        
        float dx = (u - intrinsic[0][2]) / intrinsic[0][0];
        float dy = (v - intrinsic[1][2]) / intrinsic[1][1];
        float dz = 1.0;
        

        float wdx = rot[0][0] * dx + rot[1][0] * dy + rot[2][0] * dz;
        float wdy = rot[0][1] * dx + rot[1][1] * dy + rot[2][1] * dz;
        float wdz = rot[0][2] * dx + rot[1][2] * dy + rot[2][2] * dz;
        float length = sqrtf(wdx * wdx + wdy * wdy + wdz * wdz);
        wdx /= length;
        wdy /= length;
        wdz /= length;

        glm::vec3 dir = glm::vec3(wdx, wdy, wdz);
        return Ray(origin, dir);
    }

    __device__ void setPosition(glm::mat4 pose_) {
        origin = glm::vec3(pose_[3]);
        rot = glm::mat3(pose_);
    }

    __device__ void setIntrinsic(glm::mat3 intrinsic_) {
        intrinsic = intrinsic_;
    }

    // __device__ void setLookAt(glm::vec3 forward, glm::vec) {

    // }

};





#endif // __CAMERA_H__