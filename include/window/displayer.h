#ifndef __DISPLAYER_H__
#define __DISPLAYER_H__

#include "common.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "shader/shader.h"
#include "shader/program.h"
#include "renderer.cuh"

class Displayer {
public:
    Displayer(const uint32_t width, const uint32_t height);
    ~Displayer();
    void display();
    void resize(const uint32_t width_, const uint32_t height_);

    void processInput(GLFWwindow* window) {
        
        const float cameraSpeed = 0.01f;
        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            m_cameraPos += cameraSpeed * m_cameraFront;
        }
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            m_cameraPos -= cameraSpeed * m_cameraFront;

        auto cameraRight = glm::normalize(glm::cross(m_cameraUp, -m_cameraFront));
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            m_cameraPos += cameraSpeed * cameraRight;
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            m_cameraPos -= cameraSpeed * cameraRight;

        auto cameraUp = glm::normalize(glm::cross(-m_cameraFront, cameraRight));
        if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                m_cameraPos -= cameraSpeed * cameraUp;
            else
                m_cameraPos += cameraSpeed * cameraUp;

        m_cameraFront = 
            glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraYaw), glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraPitch), glm::vec3(1.0f, 0.0f, 0.0f)) *
            glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);

        view = glm::lookAt(
            m_cameraPos,
            m_cameraPos + m_cameraFront,
            m_cameraUp);

        view = glm::inverse(view) * glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        renderer->setPosition(view);

    }

    void mouseButton(int button, int action, double x, double y){
        if(button == GLFW_MOUSE_BUTTON_RIGHT) {
            if(action == GLFW_PRESS) {
                m_prevMousePos = glm::vec2((float)x, (float)y);
                m_cameraControl = true;
            }
            else if (action == GLFW_RELEASE) {
                m_cameraControl = false;
            }
        }
    }

    void mouseMove(double x, double y) {
        if(!m_cameraControl) return;

        auto pos = glm::vec2((float)x, (float)y);
        auto deltaPos = pos - m_prevMousePos;

        const float cameraRotSpped = 0.3f;
        m_cameraYaw -= deltaPos.x * cameraRotSpped;
        m_cameraPitch -= deltaPos.y * cameraRotSpped;
        
        if(m_cameraYaw < 0.0f) m_cameraYaw += 360.f;
        if(m_cameraYaw > 360.0f) m_cameraYaw -= 360.0f;
        if(m_cameraPitch > 89.0f) m_cameraPitch = 89.0f;
        if(m_cameraPitch < -89.0f) m_cameraPitch = -89.0f;

        m_prevMousePos = pos;
    }

    bool m_cameraControl { false };

    float m_cameraPitch { 0.0f };
    float m_cameraYaw { 0.0f };
    glm::vec2 m_prevMousePos { glm::vec2(0.0f) };
    glm::vec3 m_cameraPos { glm::vec3(0.0f, 0.0f, 3.0f) };
    glm::vec3 m_cameraFront { glm::vec3(0.0f, 0.0f, -1.0f) };
    glm::vec3 m_cameraUp { glm::vec3(0.0f, 1.0f, 0.0f) };
    glm::mat4 extrinsic;
    glm::mat4 view { glm::mat4(1.0f) };

private:
    KernelRendererUPtr renderer;

    uint32_t width, height;
    uint32_t vao, pbo;
    uint32_t shaderProgram;
    uint32_t texture;
    
    cudaGraphicsResource_t cudaResource;

    dim3 gridLayout;
    dim3 blockLayout = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

    ProgramUPtr program;

    
};

#endif // __DISPLAYER_H__