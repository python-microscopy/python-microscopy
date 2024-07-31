#version 330

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec4 light_position;
uniform float shininess;
uniform vec4 view_vector;

in vec4 FrontColor;
in vec3 normal;
in float vis;
in float depth;

out vec4 FragColor;

void main(void)
{
    if (vis < .5) discard;

    vec4 inputColor = FrontColor;
    vec4 white = vec4(1.0, 1.0, 1.0, FrontColor.a);
    vec4 ambient = inputColor * light_ambient;
    //direction to the lightsource
    vec3 lightsource = normalize(vec3(light_position));
    //normal between object and eye space
    vec3 normal = normalize(normal);

    float diffuseLight = abs(dot(lightsource, normal));//, 0.0);
    vec4 diffuse = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    //vec4 diff = vec4(0.0, 0.0, 0.0, 1.0);
    if (diffuseLight > 0){
        diffuse = diffuseLight * inputColor * light_diffuse;
        vec3 H = normalize(light_position.xyz + view_vector.xyz);
        float specLight = pow(abs(dot(H, normal)), shininess);
        vec4 spec = white * light_specular;
        specular = specLight * spec;
    }

    FragColor = ambient + diffuse + specular;
    FragColor.a = inputColor.a;
}
