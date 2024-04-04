#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <memory>
#include <string>
#include <optional>
#include <fstream>
#include <sstream>

#include "glad/glad.h"
#include "glfw/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define CLASS_PTR(klassName) \
class klassName; \
using klassName ## UPtr = std::unique_ptr<klassName>; \
using klassName ## Ptr = std::shared_ptr<klassName>; \
using klassName ## WPtr = std::weak_ptr<klassName>;


inline std::optional<std::string> loadTextFile(const std::string& filename) {

    std::ifstream fin(filename);
    if(!fin.is_open()){
        printf("Failed to open file : %s\n", filename.c_str());
        return {};
    }
    std::stringstream text;
    text << fin.rdbuf();
    return text.str();
}

#endif // __COMMON_H__