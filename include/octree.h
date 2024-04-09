#ifndef __OCTREE_H__
#define __OCTREE_H__

#include "common.h"
#include "ray.h"

class Octree {
public:
    glm::vec3 min, max ,center;
    float resolution;

    __device__ Octree(glm::vec3 min, glm::vec3 max, float resolution)
        : min(min), max(max), resolution(resolution) {

        center = (min + max) / glm::vec3(2.0f);

    }

    __device__ double traverse(Ray& ray) {
        return 200;
    }

};



#endif // __OCTREE_H__