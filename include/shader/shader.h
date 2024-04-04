#ifndef __SHADER_H__
#define __SHADER_H__

#include "common.h"

CLASS_PTR(Shader);
class Shader {
public:
    static ShaderUPtr createFromFile(const std::string& filename, GLenum shaderType);
    ~Shader();

    uint32_t get() const { return m_shader; }

private:
    Shader() {}
    bool loadFile(const std::string& filename, GLenum shaderType);
    uint32_t m_shader { 0 };

};


#endif // __SHADER_H__