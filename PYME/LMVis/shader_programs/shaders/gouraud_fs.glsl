#version 120

varying vec4 vertexColor;
varying float vis;

void main(void)
{
   if (vis < .5) discard;

   gl_FragColor = vertexColor;
}
