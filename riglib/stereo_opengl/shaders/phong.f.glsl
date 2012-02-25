#version 120

uniform mat4 modelview;
uniform sampler2D texture;
uniform vec4 basecolor;

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;

const vec4 light_direction = vec4(-1, 2, -2, 0.0);
const vec4 light_diffuse = vec4(0.8, 0.8, 0.8, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);


void main() {
    vec3 mv_light_direction = (modelview * light_direction).xyz,
         normal = normalize(vnormal),
         eye = normalize(vposition),
         reflection = reflect(mv_light_direction, normal);
    
    vec4 frag_diffuse = texture2D(texture, vtexcoord) + basecolor;
    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    //vec4 specular_factor
    //    = max(pow(-dot(reflection, eye), frag_shininess), 0.0) * light_specular;
    
    gl_FragColor = ambient_diffuse_factor * frag_diffuse;
    //gl_FragColor = vec4(gl_FragCoord.z+1)/2;
    //gl_FragColor = vec4(-dot(normal, light_direction.xyz));
}