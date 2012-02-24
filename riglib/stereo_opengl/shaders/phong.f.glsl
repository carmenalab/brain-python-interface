#version 150


uniform mat4 xfm;
uniform sampler2D texture;
uniform vec4 basecolor;

in VertexData {
    vec3 position;
    vec2 texcoord;
    vec3 normal;
} VertexIn;

const vec3 light_direction = vec3(-1, 2, -5);
const vec4 light_diffuse = vec4(0.8, 0.8, 0.8, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);

out vec4 color_output;

void main()
{
    
    vec3 mv_light_direction = (xfm * vec4(light_direction, 0.0)).xyz,
         normal = normalize(VertexIn.normal),
         eye = normalize(VertexIn.position),
         reflection = reflect(mv_light_direction, normal);
    
    vec4 frag_diffuse = texture2D(texture, VertexIn.texcoord) + basecolor;
    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    //vec4 specular_factor
    //    = max(pow(-dot(reflection, eye), frag_shininess), 0.0) * light_specular;

    //gl_FragColor = ambient_diffuse_factor * frag_diffuse;
    
    color_output = ambient_diffuse_factor * frag_diffuse;
}
