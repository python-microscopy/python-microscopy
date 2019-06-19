precision mediump float;
varying vec2 v_tile_pos;

void main() {

  gl_FragColor = vec4(0., v_tile_pos.y, v_tile_pos.x, 1.);
}