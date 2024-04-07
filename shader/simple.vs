#version 330 core

out vec2 TexCoord;

void main() {
    
    TexCoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(TexCoord.x * 2.0 - 1.0, TexCoord.y * -2.0 + 1.0, 0.0, 1.0);

}