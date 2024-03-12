#version 330

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
uniform mat4 modelview_projection_matrix;
uniform mat4 model_matrix;
uniform mat3 normal_matrix;
out vec3 normalInterp;
out vec3 v2f_color;
out vec3 vertPos;

void main() {
    vec4 vertPos4 = model_matrix * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = normalize(normal_matrix * normal);
    v2f_color = vec3(0.0, 0.0, 0.0);
    gl_Position = modelview_projection_matrix * vec4(position, 1.0);
}