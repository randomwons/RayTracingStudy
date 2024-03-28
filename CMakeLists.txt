cmake_minimum_required(VERSION 3.24)
project(GL_TEST2 CXX CUDA)

add_executable(${PROJECT_NAME}
    main.cu
)

include(Dependency.cmake)

target_include_directories(${PROJECT_NAME} PUBLIC include ${DEP_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${DEP_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEP_LIBS})

add_dependencies(${PROJECT_NAME} ${DEP_LIST})