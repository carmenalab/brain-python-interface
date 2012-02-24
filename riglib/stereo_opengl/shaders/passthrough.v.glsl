#version 150

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform sampler2D texture;
uniform vec4 basecolor;

in vec4 position;
in vec2 texcoord;
in vec4 normal;

/*
out float shininess;
out vec4 specular;
*/
out VertexData {
    vec3 position;
    vec3 normal;
    vec2 texcoord;
} VertexOut;

void main(void)
{
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;

    VertexOut.position = eye_position.xyz;
    VertexOut.texcoord = texcoord;
    VertexOut.normal   = (xfm * vec4(normal.xyz,0)).xyz;
}
