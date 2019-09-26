#version 120
//This vertex shader simply transforms the position in eye space and forwards it.
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

varying float vis;

void main() {
    bool visible;
    gl_PointSize = gl_Point.size;
    gl_FrontColor = gl_Color;

    visible = gl_Vertex.x > x_min && gl_Vertex.x < x_max;
    visible = visible && gl_Vertex.y > y_min && gl_Vertex.y < y_max;
    visible = visible && gl_Vertex.z > z_min && gl_Vertex.z < z_max;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    visible = visible && gl_Position.z > v_min && gl_Position.z < v_max;

    vis = float(visible);

    gl_FrontColor.a = gl_FrontColor.a*(vis);
}