#version 330

in vec3 vertPos;
in vec3 normalInterp;

out vec4 FragColor;

uniform float shininess;
uniform vec3 ambient_color;
uniform vec3 diffuse_color;
uniform vec3 light_color;
uniform mat4 view_matrix;
uniform vec3 light_position;

void main()
{
    vec3 transformedLightPos = vec3(view_matrix * vec4(light_position, 1.0));

    vec3 N = normalize(normalInterp);
    vec3 L = normalize(transformedLightPos - vertPos);

    float lambertian = max(dot(N, L), 0.0);
    float specular = 0.0;
    if (lambertian > 0.0) {
        vec3 R = reflect(-L, N);
        vec3 V = normalize(-vertPos);
        float specAngle = max(dot(R, V), 0.0);
        specular = pow(specAngle, shininess);
    }
    FragColor = vec4(ambient_color + lambertian * diffuse_color + specular * light_color, 1.0);
}
