#version 110

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform mat4 modelview;
uniform vec4 basecolor;
uniform float shininess;

attribute vec4 position;
attribute vec2 texcoord;
attribute vec4 normal;

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;
varying float vshininess;

void main(void) {
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;
    
    vposition = eye_position.xyz;
    vnormal   = (xfm * vec4(normal.xyz, 0.0)).xyz;
    vtexcoord = texcoord;
    vshininess = shininess;
}
