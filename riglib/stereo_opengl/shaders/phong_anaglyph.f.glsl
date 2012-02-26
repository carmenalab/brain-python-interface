#version 110

uniform mat4 modelview;
uniform vec4 basecolor;
uniform float shininess;
uniform vec4 spec_color = vec4(1.0);
uniform int ntex;
uniform vec4 texweight[20];
uniform sampler2D textures[20];

uniform sampler2D fbo;

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;

const vec4 light_direction = vec4(-1, 2, -2, 0.0);
const vec4 light_diffuse = vec4(0.8, 0.8, 0.8, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);

const vec4 ltint = vec4(1, 0, 0.5, 1);
const vec4 rtint = vec4(0, 1, 0.5, 1);

void main() {
    int i;
    vec3 mv_light_direction = (modelview * light_direction).xyz,
         normal = normalize(vnormal),
         eye = normalize(vposition),
         reflection = reflect(mv_light_direction, normal);
    
    vec4 frag_diffuse = basecolor;
    for (i=0; i<ntex; i++) {
        frag_diffuse += texweight[i] * texture2D(textures[i], vtexcoord);
    }

    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    vec4 specular_factor
        = max(pow(-dot(reflection, eye), frag_shininess), 0.0) * light_specular;
    
    gl_FragColor = ltint * texture2D(fbo,gl_FragCoord.xy) + 
        rtint * ( ambient_diffuse_factor * frag_diffuse + 
                  specular_factor * spec_color );
}