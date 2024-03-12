#version 330

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

uniform mat4 modelview_projection_matrix;
uniform mat4 modelview_matrix;
uniform mat3 normal_matrix;
uniform vec3 light_position;
uniform float shininess;
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;

out vec3 normalInterp;
out vec3 vertPos;
out vec4 vertexColor;

void main()
{
    vec4 vertPos4 = modelview_matrix * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = normalize(normal_matrix * normal);
    gl_Position = modelview_projection_matrix * vertPos4;

    vec3 N = normalize(normalInterp);
    vec3 L = normalize(light_position - vertPos);

    // Lambert's cosine law
    float lambertian = max(dot(N, L), 0.0);
    float specular = 0.0;

    if (lambertian > 0.0) {
        vec3 R = reflect(-L, N);      // Reflected light vector
        vec3 V = normalize(-vertPos); // Vector to viewer
        // Compute the specular term
        float specAngle = max(dot(R, V), 0.0);
        specular = pow(specAngle, shininess);
    }

    vertexColor = vec4(ambientColor +
                 lambertian * diffuseColor +
                 specular * specularColor, 1.0);
}
