#version 120
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
varying float vis;

void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    if (vis < .5) discard;
    gl_FragColor = gl_Color;
}

#version 330 core
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
in float vis;
in vec4 FrontColor;

layout(location = 0) out vec4 FragColor;

void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    if (vis < .5) discard;

    vec2 circ_coord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circ_coord, circ_coord) > 1) discard;
    FragColor = FrontColor;
}