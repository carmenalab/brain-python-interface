#version 110

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform sampler2D texture;

attribute vec4 position;
attribute vec2 texcoord;
/*
attribute vec4 normal;
attribute float shininess;
attribute vec4 specular;
*/
varying vec3 frag_position, frag_normal;
varying vec2 frag_texcoord;
varying float frag_shininess;
varying vec4 frag_specular;

void main()
{
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;

    /*
    frag_position = eye_position.xyz;
    frag_normal   = (xfm * vec4(normal, 0.0)).xyz;
    frag_texcoord = texcoord;
    frag_shininess = shininess;
    frag_specular = specular;*/
}
