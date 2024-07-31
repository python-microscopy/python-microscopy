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

   vec2 coord = gl_PointCoord.xy*2 - vec2(1.0);
   float m = dot(coord, coord);
   if (m > 1.0) discard;

   FragColor = vertexColor;
}
