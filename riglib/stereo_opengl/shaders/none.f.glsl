#version 110

uniform vec4 basecolor;

void main() {
    //gl_FragColor = vec4(basecolor.rgb, 1.);
    gl_FragColor = basecolor;
}