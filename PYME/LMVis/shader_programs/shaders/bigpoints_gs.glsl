#version 330 core
// convert points to quads by using a geometry shader

uniform float point_size_vp;
uniform vec2 viewport_size;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 FrontColor[];
in float vis[];

out vec4 frontColor;
out vec2 texCoord;
out float visibility;

void main(){
    float half_size = point_size_vp / 2.0;
    float aspect_ratio = viewport_size.x / viewport_size.y;
    vec4 position = gl_in[0].gl_Position;

    // pass through colour and visibility
    frontColor = FrontColor[0];
    visibility = vis[0];

    gl_Position = position + vec4(-half_size, -half_size*aspect_ratio, 0.0, 0.0);    // 1:bottom-left
    texCoord = vec2(0.0, 0.0);
    EmitVertex();   
    gl_Position = position + vec4( half_size, -half_size*aspect_ratio, 0.0, 0.0);    // 2:bottom-right
    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4(-half_size,  half_size*aspect_ratio, 0.0, 0.0);    // 3:top-left
    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    gl_Position = position + vec4( half_size,  half_size*aspect_ratio, 0.0, 0.0);    // 4:top-right
    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    EndPrimitive();

}
