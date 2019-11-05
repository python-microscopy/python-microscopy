#version 120
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
varying float vis;
varying vec2 tex_coords;

uniform vec2 clim;

uniform sampler2D im_sampler;
uniform sampler1D lut;

void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    //if (vis < .5) discard;

    float I;
    float a = clim.x;
    float b = clim.y;
    vec4 c;

    I = texture2D(im_sampler, tex_coords).r;
    I = clamp((I-a)/(b-a), 0.0, 1.0);

    c = texture1D(lut, I);

    gl_FragColor = c;

    //gl_FragColor = vec4(I, I, I, 1.0);
}