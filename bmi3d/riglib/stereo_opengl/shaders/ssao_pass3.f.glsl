#version 110

uniform sampler2D shadow;
uniform vec4 window;

void main() {
    vec2 uv = (gl_FragCoord.xy - window.xy) / window.zw;
    //vec2 uv = 0.5*(vppos + 1.);
    //gl_FragColor = texture2D(shadow, uv);
    float shade = texture2D(shadow, uv).r;
    gl_FragColor = phong() * vec4(shade, shade, shade, 1.0);
    //gl_FragColor = vec4(uv, 0., 1.);
}