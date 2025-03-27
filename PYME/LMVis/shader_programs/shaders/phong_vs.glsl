#version 330

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
out vec3 normal;
out float vis;
out float depth;

void main(void){
    bool visible;

    FrontColor = Color;
    normal = normalize(NormalMatrix*Normal);

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