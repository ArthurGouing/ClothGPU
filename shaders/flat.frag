#version 330 core

// Data in
in vec3 fragColor;

// Data out
out vec4 outColor;

void main() {
  outColor = vec4(fragColor, 1);
}