#version 120
uniform sampler2D tex; // the texture with the scene you want to blur
varying vec2 uv;
 
const float blurSize = 1.0/512.0; // I've chosen this size because this will result in that every step will be one pixel wide if the tex texture is of size 512x512
 
void main(void)
{
   vec4 sum = vec4(0.0);
 
   // blur in y (vertical)
   // take nine samples, with the distance blurSize between them
   sum += texture2D(tex, vec2(uv.x - 4.0*blurSize, uv.y)) * 0.05;
   sum += texture2D(tex, vec2(uv.x - 3.0*blurSize, uv.y)) * 0.09;
   sum += texture2D(tex, vec2(uv.x - 2.0*blurSize, uv.y)) * 0.12;
   sum += texture2D(tex, vec2(uv.x - blurSize, uv.y)) * 0.15;
   sum += texture2D(tex, vec2(uv.x, uv.y)) * 0.16;
   sum += texture2D(tex, vec2(uv.x + blurSize, uv.y)) * 0.15;
   sum += texture2D(tex, vec2(uv.x + 2.0*blurSize, uv.y)) * 0.12;
   sum += texture2D(tex, vec2(uv.x + 3.0*blurSize, uv.y)) * 0.09;
   sum += texture2D(tex, vec2(uv.x + 4.0*blurSize, uv.y)) * 0.05;
   
   gl_FragColor = sum;
}