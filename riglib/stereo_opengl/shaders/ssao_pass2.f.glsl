#version 120
/*
 * Screen-space ambient occlusion code modified from 
 * http://www.gamerendering.com/2009/01/14/ssao/
 *
 */

uniform sampler2D rnm;
uniform sampler2D normalMap;

varying vec2 uv;
const float totStrength = 1.38;
const float strength = 0.07;
const float offset = 17.0;
const float falloff = 0.000005;
const float rad = .008;
#define SAMPLES 16 // 10 is good
const float invSamples = -totStrength/16.;

vec3 pSphere[16] = vec3[](vec3(0.53812504, 0.18565957, -0.43192),vec3(0.13790712, 0.24864247, 0.44301823),vec3(0.33715037, 0.56794053, -0.005789503),vec3(-0.6999805, -0.04511441, -0.0019965635),vec3(0.06896307, -0.15983082, -0.85477847),vec3(0.056099437, 0.006954967, -0.1843352),vec3(-0.014653638, 0.14027752, 0.0762037),vec3(0.010019933, -0.1924225, -0.034443386),vec3(-0.35775623, -0.5301969, -0.43581226),vec3(-0.3169221, 0.106360726, 0.015860917),vec3(0.010350345, -0.58698344, 0.0046293875),vec3(-0.08972908, -0.49408212, 0.3287904),vec3(0.7119986, -0.0154690035, -0.09183723),vec3(-0.053382345, 0.059675813, -0.5411899),vec3(0.035267662, -0.063188605, 0.54602677),vec3(-0.47761092, 0.2847911, -0.0271716));


void main(void) {   
   // grab a normal for reflecting the sample rays later on
   vec3 fres = normalize((texture2D(rnm,uv*offset).xyz*2.0) - vec3(1.0));
   
   vec4 currentPixelSample = texture2D(normalMap,uv);
   float currentPixelDepth = currentPixelSample.w;
   vec3 norm = currentPixelSample.xyz;
 
   // current fragment coords in screen space
   vec3 ep = vec3(uv.xy,currentPixelDepth);
 
   float bl = 0.0;
   // adjust for the depth ( not sure if this is good..)
   float radD = rad/currentPixelDepth;
   
   //vec3 ray, se, occNorm;
   float occluderDepth, depthDifference;
   vec3 ray;
   vec4 occFrag;

   for(int i=0; i<SAMPLES;i++) {
      // get a vector (randomized inside of a sphere with radius 1.0) from a texture and reflect it
      ray = radD*reflect(pSphere[i],fres);
 
      // get the coordinate of the occluder fragment
      occFrag = texture2D(normalMap, ep.xy + sign(dot(ray,norm))*ray.xy);

      // if depthDifference is negative = occluder is behind current fragment
      depthDifference = currentPixelDepth-occFrag.w;
      
      // calculate the difference between the normals as a weight of
      // the falloff equation, starts at falloff and is kind of 1/x^2 falling
      bl += step(falloff,depthDifference)*
            (1.0-dot(occFrag.xyz,norm))*
            (1.0-smoothstep(falloff,strength,depthDifference));
   }
   // output the result
   gl_FragColor = vec4(1.0+bl*invSamples);
   //gl_FragColor = vec4(dot(norm, norm));
}
