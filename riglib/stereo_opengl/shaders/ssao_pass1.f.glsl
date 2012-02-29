#version 110

varying vec3 vnormal;

void main() {
    gl_FragColor = vec4(normalize(vnormal), 1.0);
}