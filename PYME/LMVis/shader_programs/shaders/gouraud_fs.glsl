#version 120

varying vec4 vertexColor;
varying float vis;

void main(void)
{
   if (vis < .5) discard;

   gl_FragColor = vertexColor;
}


#version 330

in vec4 vertexColor;
in float vis;

out vec4 FragColor;

void main(void)
{
   if (vis < .5) discard;

   FragColor = vertexColor;
}
