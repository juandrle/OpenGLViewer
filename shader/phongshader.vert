#version 330

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

uniform mat4 modelview_projection_matrix;
uniform mat4 model_matrix;
uniform mat3 normal_matrix;
out vec3 normalInterp;
out vec3 vertPos;

void main()
{
    vec4 vertPos4 = model_matrix * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = normalize(normal_matrix * vec3(normal));
    gl_Position = modelview_projection_matrix * vertPos4;
}