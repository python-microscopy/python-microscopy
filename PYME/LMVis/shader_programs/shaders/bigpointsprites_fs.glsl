#version 330

// This fragmentshader takes the given color and displays it.
// It's transparency is determined by a submitted texture. That way the brightness is decreasing with the distance to the \
// center of the point.

uniform sampler2D tex2D;

in vec4 frontColor;
in float visibility;
in vec2 texCoord;

layout (location = 0) out vec4 fragColor;

void main()
{
    if (visibility <0.5) discard;
    float alpha = frontColor.a * texture(tex2D, texCoord).x;
    fragColor = vec4(frontColor.xyz, alpha);
}