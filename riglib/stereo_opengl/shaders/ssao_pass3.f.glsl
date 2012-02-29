#version 110

uniform sampler2D shadow;
uniform float width, height;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(width, height);
    float shade = texture2D(shadow, uv).r;
    gl_FragColor = phong() * vec4(shade, shade, shade, 1.0);
}