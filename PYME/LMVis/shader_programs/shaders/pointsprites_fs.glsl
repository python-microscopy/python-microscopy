#version 120

// This fragmentshader takes the given color and displays it.
// It's transparency is determined by a submitted texture. That way the brightness is decreasing with the distance to the \
// center of the point.

uniform sampler2D tex2D;

void main()
{
    float alpha = gl_Color.a * texture2D(tex2D, gl_PointCoord).x;
    gl_FragColor = vec4(gl_Color.xyz, alpha);
}

#version 330

// This fragmentshader takes the given color and displays it.
// It's transparency is determined by a submitted texture. That way the brightness is decreasing with the distance to the \
// center of the point.

uniform sampler2D tex2D;

in vec4 frontColor;
in float vis;

layout (location = 0) out vec4 fragColor;

void main()
{
    if (vis <0.5) discard;
    float alpha = frontColor.a * texture(tex2D, gl_PointCoord).x;
    fragColor = vec4(frontColor.xyz, alpha);
}