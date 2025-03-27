#version 120

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec4 light_position;
uniform float shininess;
uniform vec4 view_vector;

varying vec4 FrontColor;

varying float vis;
varying float depth;

uniform float x_min;
uniform float x_max;
uniform float y_min;
uniform float y_max;
uniform float z_min;
uniform float z_max;
uniform float v_min;
uniform float v_max;

void main(void){
    bool visible;

    vec4 inputColor = gl_Color;
    vec4 white = vec4(1.0, 1.0, 1.0, gl_Color.a);
    vec4 ambient = inputColor * light_ambient;
    //direction to the lightsource
    vec3 lightsource = normalize(vec3(light_position));
    //normal between object and eye space
    vec3 normal = normalize(gl_NormalMatrix*gl_Normal);

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
    FrontColor = ambient + diffuse + specular;
    FrontColor.a = inputColor.a;

    visible = gl_Vertex.x > x_min && gl_Vertex.x < x_max;
    visible = visible && gl_Vertex.y > y_min && gl_Vertex.y < y_max;
    visible = visible && gl_Vertex.z > z_min && gl_Vertex.z < z_max;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    vec4 eye_frame = gl_ModelViewMatrix*gl_Vertex; //just do model view part for OIT calcs

    visible = visible && gl_Position.z > v_min && gl_Position.z < v_max;
    vis = (float(visible));

    // depth n eye/camera space coordinates
    depth = (eye_frame.z);
}

#version 330

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec4 light_position;
uniform float shininess;
uniform vec4 view_vector;


uniform float x_min;
uniform float x_max;
uniform float y_min;
uniform float y_max;
uniform float z_min;
uniform float z_max;
uniform float v_min;
uniform float v_max;

uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform float point_size_px;
layout (location = 0) in vec3 Vertex;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec4 Color;

out vec4 FrontColor;
out float vis;
out float depth;

void main(void){
    bool visible;

    gl_PointSize = point_size_px;

    vec4 inputColor = Color;
    vec4 white = vec4(1.0, 1.0, 1.0, Color.a);
    vec4 ambient = inputColor * light_ambient;
    //direction to the lightsource
    vec3 lightsource = normalize(vec3(light_position));
    //normal between object and eye space
    vec3 normal = normalize(NormalMatrix*Normal);

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
    FrontColor = ambient + diffuse + specular;
    FrontColor.a = inputColor.a;

    visible = Vertex.x > x_min && Vertex.x < x_max;
    visible = visible && Vertex.y > y_min && Vertex.y < y_max;
    visible = visible && Vertex.z > z_min && Vertex.z < z_max;

    gl_Position = ModelViewProjectionMatrix * vec4(Vertex, 1.0);

    vec4 eye_frame = ModelViewMatrix*vec4(Vertex, 1.0); //just do model view part for OIT calcs

    visible = visible && gl_Position.z > v_min && gl_Position.z < v_max;
    vis = (float(visible));

    // depth n eye/camera space coordinates
    depth = (eye_frame.z);
}