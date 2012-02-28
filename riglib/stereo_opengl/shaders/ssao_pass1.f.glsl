#version 110

uniform float nearclip;
uniform float farclip;

void main() {
    gl_FragColor[0] = phong();
    gl_FragColor[1] = vec4(normalize(vnormal), (gl_FragCoord.z - nearclip) / farclip);
}