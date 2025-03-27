#version 120

varying vec4 FrontColor;
varying float vis;

void main(void)
{
   if (vis < .5) discard;

   gl_FragColor = FrontColor;
}


#version 330

in vec4 FrontColor;
in float vis;

out vec4 FragColor;

void main(void)
{
   if (vis < .5) discard;

   FragColor = FrontColor;
}
