#version 330

// This fragmentshader takes the given color and displays it, ignoring texture coordinates
// TODO - use texture coordinates to render a 3D cylinder

//uniform sampler2D tex2D;

in vec4 frontColor;
in float visibility;
in vec2 texCoord;

layout (location = 0) out vec4 fragColor;

void main()
{
    if (visibility <0.5) discard;
    //float alpha = frontColor.a * texture(tex2D, texCoord).x;
    fragColor = frontColor;
}