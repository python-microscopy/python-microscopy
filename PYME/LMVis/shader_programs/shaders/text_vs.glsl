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
varying vec2 tex_coords;

void main() {
    bool visible;
    vec4 v_position;
    //gl_PointSize = gl_Point.size;
    gl_FrontColor = gl_Color;

    visible = gl_Vertex.x > x_min && gl_Vertex.x < x_max;
    visible = visible && gl_Vertex.y > y_min && gl_Vertex.y < y_max;
    visible = visible && gl_Vertex.z > z_min && gl_Vertex.z < z_max;

    //v_position = clip_rotation_matrix*gl_Vertex;
    //visible = visible && v_position.z > v_min && v_position.z < v_max;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    vis = float(visible);
    tex_coords = gl_MultiTexCoord0.xy;

    //gl_FrontColor.a = gl_FrontColor.a*(float(visible));
}