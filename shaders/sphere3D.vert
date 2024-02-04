#version 330 core

// Uniforms
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 center;
uniform float radius;

in vec3 vertex;
out vec2 coord;

void main() {
    coord = vertex.xy;

    vec3 disp = transpose(mat3(viewMatrix)) * vertex;
    vec3 position = center + disp * radius;

    gl_Position = projectionMatrix * viewMatrix 
    * modelMatrix * vec4(position, 1);
}