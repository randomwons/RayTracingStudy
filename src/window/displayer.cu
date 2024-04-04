#include "window/displayer.h"

Displayer::Displayer(const uint32_t width, const uint32_t height) : width(width), height(height) {

    shaderProgram = glCreateProgram();
    ShaderPtr vertShader = Shader::createFromFile("shader/simple.vs", GL_VERTEX_SHADER);
    ShaderPtr fragShader = Shader::createFromFile("shader/simple.fs", GL_FRAGMENT_SHADER);
    program = Program::create({vertShader, fragShader});

    glGenVertexArrays(1, &vao);
    glBindVertexArray(0);

    renderer = KernelRendererUPtr(new KernelRenderer(width, height));

}

void Displayer::display() {

    // m_cameraFront = 
    //     glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraYaw), glm::vec3(0.0f, 1.0f, 0.0f)) *
    //     glm::rotate(glm::mat4(1.0f), glm::radians(m_cameraPitch), glm::vec3(1.0f, 0.0f, 0.0f)) *
    //     glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);

    // view = glm::lookAt(
    //     m_cameraPos,
    //     m_cameraPos + m_cameraFront,
    //     m_cameraUp);

    // view = glm::inverse(view) * glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    // cudaMemcpy(d_intrinsic, (void*)&intrinsic, sizeof(glm::mat3), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_extrinsic, (void*)&view, sizeof(glm::mat4), cudaMemcpyHostToDevice);

    program->use();
    glBindVertexArray(vao);
    renderer->render();
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Displayer::resize(const uint32_t width_, const uint32_t height_) {

    width = width_;
    height = height_;

    renderer->resize(width_, height_);
}

Displayer::~Displayer() {

    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    
}