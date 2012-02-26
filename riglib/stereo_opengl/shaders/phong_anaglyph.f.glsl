#version 110

uniform mat4 modelview;
uniform vec4 basecolor;
uniform vec4 texweight[20];
uniform sampler2D textures[20];
uniform sampler2D fbo;

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;
varying float vshininess;

const vec4 light_direction = vec4(-1, 2, -2, 0.0);
const vec4 light_diffuse = vec4(0.6, 0.6, 0.6, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);

const vec4 spec_color = vec4(1.0);

const vec4 ltint = vec4(1, 0, 0.5, 1);
const vec4 rtint = vec4(0, 1, 0.5, 1);

void main() {
    int i;
    vec3 mv_light_direction = (modelview * light_direction).xyz,
         normal = normalize(vnormal),
         eye = normalize(-vposition),
         reflection = normalize(-reflect(mv_light_direction, normal));
    
    vec4 frag_diffuse = texweight[0] * texture2D(textures[0], vtexcoord) + basecolor;

    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    vec4 specular_factor
        = pow(max(dot(-reflection, eye), 0.0), vshininess) * light_specular;
    
    gl_FragColor = ltint * texture2D(fbo,normalize(gl_FragCoord).xy) + 
        rtint * ( ambient_diffuse_factor * frag_diffuse + 
                  specular_factor * spec_color );
}