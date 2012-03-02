#version 110

varying vec3 vnormal;
varying vec3 vposition;

void main() {
    gl_FragData[0] = vec4(normalize(vnormal), 1.);
}