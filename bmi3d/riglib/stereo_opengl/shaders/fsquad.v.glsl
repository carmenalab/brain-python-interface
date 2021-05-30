#version 120

attribute vec4 position;

varying vec2 uv;

void main(void) {
    gl_Position = position;
    uv = (vec2( position.x, position.y ) + vec2( 1.0 ) ) * 0.5;
}