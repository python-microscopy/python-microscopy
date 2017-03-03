#version 330
//This vertex shader simply transforms the position in eye space and forwards it.
layout(location = 0) in vec4 aVert;
layout(location = 1) in vec4 color;

uniform mat4 mvpMatrix;

out vec4 position;
out vec4 vColor;

void main()
{
    position = mvpMatrix * aVert;
    vColor = color;
    gl_PointSize = 40;
    gl_Position = position;
}