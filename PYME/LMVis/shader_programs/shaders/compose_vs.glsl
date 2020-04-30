#version 120

attribute vec3 vPosition;

varying vec2 vTexcoord;

void main(void)
{
    gl_Position = vec4(vPosition,1.0);

    vTexcoord = (vPosition.xy+1.0)/2.0;
}