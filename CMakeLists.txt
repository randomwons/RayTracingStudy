cmake_minimum_required(VERSION 3.24)
project(GL_TEST2 CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
    main.cpp include/common.h
    src/window/window.cpp include/window/window.h
    src/window/displayer.cu include/window/displayer.h

)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include(Dependency.cmake)

target_include_directories(${PROJECT_NAME} PUBLIC include ${DEP_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${DEP_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEP_LIBS})

add_dependencies(${PROJECT_NAME} ${DEP_LIST})

add_executable(TEST
    test_code/cuda-opengl-interop-test.cu
)

# include(Dependency.cmake)

target_include_directories(TEST PUBLIC include ${DEP_INCLUDE_DIR})
target_link_directories(TEST PUBLIC ${DEP_LIB_DIR})
target_link_libraries(TEST PUBLIC ${DEP_LIBS})

add_dependencies(TEST ${DEP_LIST})