#include "window/displayer.h"

Displayer::Displayer(const uint32_t width, const uint32_t height) : width(width), height(height) {

    shaderProgram = glCreateProgram();
    ShaderPtr vertShader = Shader::createFromFile("shader/simple.vs", GL_VERTEX_SHADER);
    ShaderPtr fragShader = Shader::createFromFile("shader/simple.fs", GL_FRAGMENT_SHADER);
    program = Program::create({vertShader, fragShader});

    glGenVertexArrays(1, &vao);
    glBindVertexArray(0);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    renderer = KernelRendererUPtr(new KernelRenderer(cudaResource, width, height));

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
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    renderer->render();
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Displayer::resize(const uint32_t width_, const uint32_t height_) {

    width = width_;
    height = height_;
    renderer->resize(width, height);

    if(cudaResource != nullptr) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * 4, NULL, GL_DYNAMIC_COPY);  
        cudaGraphicsUnregisterResource(cudaResource);
        cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsNone); 
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        renderer->cudaResource = cudaResource;
    }
}

Displayer::~Displayer() {

    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    
}