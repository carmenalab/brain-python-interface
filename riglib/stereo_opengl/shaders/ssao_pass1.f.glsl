#version 110

uniform float nearclip;
uniform float farclip;

varying vec3 vnormal;
varying vec3 vposition;

float lindepth(float z) {
   float n = nearclip; // camera z near
   float f = farclip; // camera z far
   return (2.0 * n) / (f + n - z * (f - n));
}

void main() {
    gl_FragColor = vec4(normalize(vnormal), lindepth(gl_FragCoord.z));
}