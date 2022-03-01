#version 110
uniform mat4 modelview;
uniform vec4 basecolor;
uniform vec4 spec_color;

uniform sampler2D texture;

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;
varying float vshininess;

const vec4 light_direction = vec4(-1, 2, -2, 0.0);
const vec4 light_diffuse = vec4(0.6, 0.6, 0.6, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);

vec4 phong() {
    vec3 mv_light_direction = (modelview * light_direction).xyz,
         normal = normalize(vnormal),
         eye = normalize(-vposition),
         reflection = normalize(-reflect(mv_light_direction, normal));
    
    vec4 frag_diffuse = vec4(texture2D(texture, vtexcoord).rgb + basecolor.rgb, basecolor.a);

    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    vec4 specular_factor
        = pow(max(dot(-reflection, eye), 0.0), vshininess) * light_specular;
    
    return ambient_diffuse_factor*frag_diffuse + specular_factor*spec_color;
}
