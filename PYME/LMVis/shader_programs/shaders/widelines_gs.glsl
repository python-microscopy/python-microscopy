#version 330 core
// convert points to quads by using a geometry shader

uniform float line_width_px;
uniform vec2 viewport_size;

layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 FrontColor[];
in vec3 normal[]; // this is not atually the normal, but a vector which is the average of the line perpendiculars for the current and previous/next line segments.
in float vis[];

out vec4 frontColor;
out vec2 texCoord;
out float visibility;

void main(){
    float half_size =  line_width_px/ (2.0*viewport_size.x);
    vec4 pos0 = gl_in[0].gl_Position;
    vec4 pos1 = gl_in[1].gl_Position;

    float aspect = 1.0;//viewport_size.x / viewport_size.y;

    vec2 dir = normalize(pos1.xy - pos0.xy);

    vec2 ortho = vec2(-dir.y, dir.x*aspect);

    //vec2 norm0 = normalize(normal[0].xy);
    //vec2 norm1 = normalize(normal[1].xy);

    vec2 norm0 = ortho;
    vec2 norm1 = ortho;

    vec2 offset0 = norm0 * half_size;// / dot(norm0, ortho);
    vec2 offset1 = norm1 * half_size;// / dot(norm1, ortho);

    // pass through colour and visibility
    frontColor = FrontColor[0];
    visibility = vis[0];

    gl_Position = pos0 - vec4(offset0, 0, 0);    // 1:bottom-left
    texCoord = vec2(0.0, 0.0);
    EmitVertex();   
    gl_Position = pos1 - vec4(offset1, 0.0, 0.0);    // 2:bottom-right
    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = pos0 + vec4(offset0, 0.0, 0.0);    // 3:top-left
    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = pos1 + vec4(offset1, 0.0, 0.0);    // 4:top-right
    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

}
