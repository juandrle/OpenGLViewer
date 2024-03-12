#version 330

precision mediump float;
in vec4 vertexColor;
out vec4 fragColor;

void main() {
  fragColor = vertexColor;
}