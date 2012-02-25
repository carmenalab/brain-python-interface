#version 110

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform sampler2D texture;
uniform vec4 basecolor;

attribute vec4 position;
attribute vec2 texcoord;
attribute vec4 normal;

/*
out float shininess;
out vec4 specular;
*/

varying vec3 vposition;
varying vec3 vnormal;
varying vec2 vtexcoord;

void main(void)
{
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;
    
    vposition = eye_position.xyz;
    vnormal   = (xfm * vec4(normal.xyz, 0.0)).xyz;
    vtexcoord = texcoord;
}
