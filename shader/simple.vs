#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    TexCoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(TexCoord * 2.0 - 1.0, 0.0, 1.0);
    // gl_Position = vec4(aPos, 1.0);
    // TexCoord = aTexCoord;
}