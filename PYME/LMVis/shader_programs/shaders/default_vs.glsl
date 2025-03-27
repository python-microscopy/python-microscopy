#version 120
//This is one of the simplest vertex shader possible.
//It simply passes the given color to the fragment shader and
//transforms the vertex into screen space.

uniform float x_min;
uniform float x_max;
uniform float y_min;
uniform float y_max;
uniform float z_min;
uniform float z_max;
uniform float v_min;
uniform float v_max;
uniform mat4 clip_rotation_matrix;

varying float vis;


void main() {
    bool visible;
    vec4 v_position;
    gl_PointSize = gl_Point.size;
    gl_FrontColor = gl_Color;

    visible = gl_Vertex.x > x_min && gl_Vertex.x < x_max;
    visible = visible && gl_Vertex.y > y_min && gl_Vertex.y < y_max;
    visible = visible && gl_Vertex.z > z_min && gl_Vertex.z < z_max;

    v_position = clip_rotation_matrix*gl_Vertex;
    visible = visible && v_position.z > v_min && v_position.z < v_max;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    vis = float(visible);

    gl_FrontColor.a = gl_FrontColor.a*(float(visible));
}

#version 330 core
//This is one of the simplest vertex shader possible.
//It simply passes the given color to the fragment shader and
//transforms the vertex into screen space.

uniform float x_min;
uniform float x_max;
uniform float y_min;
uniform float y_max;
uniform float z_min;
uniform float z_max;
uniform float v_min;
uniform float v_max;
uniform mat4 clip_rotation_matrix;

uniform mat4 ModelViewProjectionMatrix;
uniform float point_size_px;

layout (location = 0) in vec3 Vertex;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec4 Color;

out float vis;
out vec4 FrontColor;
//out float PointSize;
//out vec4 Position;


void main() {
    bool visible;
    vec4 v_position;
    vec4 poss;
    //PointSize = gl_Point.size;
    gl_PointSize = point_size_px;
    FrontColor = Color;

    visible = Vertex.x > x_min && Vertex.x < x_max;
    visible = visible && Vertex.y > y_min && Vertex.y < y_max;
    visible = visible && Vertex.z > z_min && Vertex.z < z_max;

    poss = vec4(Vertex, 1.0);
    v_position = clip_rotation_matrix*poss;
    visible = visible && v_position.z > v_min && v_position.z < v_max;

    gl_Position = ModelViewProjectionMatrix * poss;

    vis = float(visible);

    FrontColor.a = FrontColor.a*(float(visible));
}