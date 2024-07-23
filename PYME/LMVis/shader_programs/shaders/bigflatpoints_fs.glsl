#version 330 core
// Like the flatpoints shader, but using texture coordinates rather than point coordinates
in float visibility;
in vec4 frontColor;
in vec2 texCoord;

layout(location = 0) out vec4 FragColor;

void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    if (visibility < .5) discard;

    vec2 circ_coord = 2.0 * texCoord - 1.0;
    if (dot(circ_coord, circ_coord) > 1) discard;
    FragColor = frontColor;
}