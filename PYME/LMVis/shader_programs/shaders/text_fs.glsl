#version 120
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
varying float vis;
varying vec2 tex_coords;

uniform sampler2D im_sampler;

void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    //if (vis < .5) discard;

    float I;

    I = texture2D(im_sampler, tex_coords).r;
    I = clamp(I, 0.0, 1.0);

    gl_FragColor = vec4(gl_Color.xyz, I);
}