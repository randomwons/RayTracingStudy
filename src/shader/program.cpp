#include "shader/program.h"

ProgramUPtr Program::create(const std::vector<ShaderPtr>& shaders) {

    auto program = ProgramUPtr(new Program());
    if(!program->link(shaders)) return nullptr;

    return std::move(program);

}

bool Program::link(const std::vector<ShaderPtr>& shaders) {

    m_program = glCreateProgram();
    for(auto& shader : shaders)
        glAttachShader(m_program, shader->get());
    glLinkProgram(m_program);

    int success = 0;
    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if(!success) {
        char infoLog[1024];
        glGetProgramInfoLog(m_program, 1024, nullptr, infoLog);
        printf("Failed to link program : %s\n", infoLog);
        return false;
    }
    return true;
}

Program::~Program() {
    if(m_program) glDeleteProgram(m_program);
}

void Program::use() const {
    glUseProgram(m_program);
}