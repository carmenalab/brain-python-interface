#version 120
/*
 * Screen-space ambient occlusion code modified from 
 * http://www.gamerendering.com/2009/01/14/ssao/
 *
 */

uniform sampler2D rnm;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

uniform float nearclip;
uniform float farclip;

varying vec2 uv;
const float totStrength = 1.4;
const float strength = 0.07;
const float offset = 18.0;
const float falloff = 0.000001;
const float rad = 0.006;
#define SAMPLES 10 // 10 is good
const float invSamples = -totStrength/10.;

float LinearizeDepth(vec2 uv) {
   float n = nearclip; // camera z near
   float f = farclip; // camera z far
   float z = texture2D(depthMap, uv).x;
   return (2.0 * n) / (f + n - z * (f - n));
}

void main(void) {
   vec3 pSphere[10];
   pSphere[0] = vec3(-0.010735935, 0.01647018, 0.0062425877);
   pSphere[1] = vec3(-0.06533369, 0.3647007, -0.13746321);
   pSphere[2] = vec3(-0.6539235, -0.016726388, -0.53000957);
   pSphere[3] = vec3(0.40958285, 0.0052428036, -0.5591124);
   pSphere[4] = vec3(-0.1465366, 0.09899267, 0.15571679);
   pSphere[5] = vec3(-0.44122112, -0.5458797, 0.04912532);
   pSphere[6] = vec3(0.03755566, -0.10961345, -0.33040273);
   pSphere[7] = vec3(0.019100213, 0.29652783, 0.066237666);
   pSphere[8] = vec3(0.8765323, 0.011236004, 0.28265962);
   pSphere[9] = vec3(0.29264435, -0.40794238, 0.15964167);

   // grab a normal for reflecting the sample rays later on
   vec3 fres = normalize((texture2D(rnm,uv*offset).xyz*2.0) - vec3(1.0));
   
   vec4 currentPixelSample = texture2D(normalMap,uv);
   float currentPixelDepth = LinearizeDepth(uv);
 
   // current fragment coords in screen space
   vec3 ep = vec3(uv.xy,currentPixelDepth);
   // get the normal of current fragment
   vec3 norm = currentPixelSample.xyz;
 
   float bl = 0.0;
   // adjust for the depth ( not sure if this is good..)
   float radD = rad/currentPixelDepth;
 
   //vec3 ray, se, occNorm;
   float occluderDepth, depthDifference;
   vec4 occluderFragment;
   vec3 ray;
   vec2 occUV;

   for(int i=0; i<SAMPLES;i++) {
      // get a vector (randomized inside of a sphere with radius 1.0) from a texture and reflect it
      ray = radD*reflect(pSphere[i],fres);
 
      // get the depth of the occluder fragment
      occUV = ep.xy + sign(dot(ray,norm))*ray.xy;
      // if depthDifference is negative = occluder is behind current fragment
      depthDifference = currentPixelDepth-LinearizeDepth(occUV);
 
      // calculate the difference between the normals as a weight of
      // the falloff equation, starts at falloff and is kind of 1/x^2 falling
      bl += step(falloff,depthDifference)*
            (1.0-dot(texture2D(normalMap,occUV).xyz,norm))*
            (1.0-smoothstep(falloff,strength,depthDifference));
   }
   //gl_FragColor = texture2D(normalMap, uv);
   // output the result
   gl_FragColor.r = 1.0+totStrength*bl*invSamples;
   //gl_FragColor = vec4(LinearizeDepth(uv));
}