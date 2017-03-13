#version 120

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec4 light_position;
uniform float shininess;
uniform vec4 view_vector;

varying vec4 vertexColor;

void main(void){

    vec4 ambient = gl_Color * light_ambient;
    //direction to the lightsource
    vec3 lightsource = normalize(vec3(light_position));
    //normal between object and eye space
    vec3 normal = gl_NormalMatrix*gl_Normal;

    float diffuseLight = max(dot(lightsource, normal), 0.0);
    vec4 diffuse = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);

    if (diffuseLight > 0){
        vec4 diff = gl_Color * light_diffuse;
        diffuse = diffuseLight * diff;
        vec3 H = normalize(light_position.xyz + view_vector.xyz);
        float specLight = pow(max(dot(H, normal),0), shininess);
        vec4 spec = gl_Color * light_specular;
        specular = specLight * spec;
    }
    vertexColor = ambient + diffuse + specular;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}