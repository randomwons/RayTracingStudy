#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <memory>
#include <string>
#include <optional>

#include "glad/glad.h"
#include "glfw/glfw3.h"

#define CLASS_PTR(klassName) \
class klassName; \
using klassName ## UPtr = std::unique_ptr<klassName>; \
using klassName ## Ptr = std::unique_ptr<klassName>; \
using klassName ## WPtr = std::unique_ptr<klassName>;


#endif // __COMMON_H__