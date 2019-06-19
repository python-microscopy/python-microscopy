precision mediump float;
uniform sampler2D u_tile;
uniform vec2 u_tile_size;
varying vec2 v_tile_pos;

//
// Sample the color at offset
//
vec4 rgba(float dx, float dy) {
  // calculate the color of sampler at an offset from position
  return texture2D(u_tile, v_tile_pos+vec2(dx,dy));
}

void main() {

  gl_FragColor = rgba(0.,0.);
}