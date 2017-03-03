#version 330

// This fragmentshader takes the given color and displays it.
// It's transparency is determined by a submitted texture. That way the brighness is decreasing with the distance to the \
// center of the point.



// This shader isn't compatible to our code and not used so far.

in vec4 position;
in vec4 vColor;
out vec4 fragColor;

uniform sampler2D tex2D;

void main()
{
    fragColor = vec4(vColor.xyz, texture(tex2D, gl_PointCoord));
}