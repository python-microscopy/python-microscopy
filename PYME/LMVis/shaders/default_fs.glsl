#version 120
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
void main() {
    //gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    gl_FragColor = gl_Color;
}