#version 330 core

// Uniforms
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

// Data in
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

// Data out
out vec3 fragColor;

void main() {
  gl_Position = projectionMatrix * viewMatrix
  * modelMatrix * vec4(position, 0, 1);
  fragColor = color;
}

