#ifndef __PROGRAM_H__
#define __PROGRAM_H__

#include "common.h"
#include "shader/shader.h"
#include <vector>

CLASS_PTR(Program);
class Program {
public:
    static ProgramUPtr create(const std::vector<ShaderPtr>& shaders);
    ~Program();

    uint32_t get() const { return m_program; }
    void use() const;

private:
    Program() {}
    bool link(const std::vector<ShaderPtr>& shaders);
    uint32_t m_program { 0 };

};


#endif // __PROGRAM_H__