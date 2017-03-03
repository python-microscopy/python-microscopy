#version 330
//This is a very simple shader. It simply forwards the given color to the fragment.
//There's no shading involved.
void main() {
    gl_FragColor = gl_Color;
}