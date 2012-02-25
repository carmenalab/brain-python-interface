#version 120
#extension GL_ARB_geometry_shader4 : enable

varying in vec4 position[3];
varying in vec3 normal[3];
varying in vec2 texcoord[3];

varying out vec4 vposition;
varying out vec3 vnormal;
varying out vec2 vtexcoord;

void main() {
    int i;
    vec4 vec1 = gl_PositionIn[1] - gl_PositionIn[0];
    vec4 vec2 = gl_PositionIn[2] - gl_PositionIn[0];
    vec3 normal = cross(vec1.xyz, vec2.xyz);

    for(i = 0; i < gl_VerticesIn; i++) {
        // copy attributes
        gl_Position = gl_PositionIn[i];
        vposition = position[i];
        vnormal = normal[i];
        vtexcoord = texcoord[i];

        // done with the vertex
        EmitVertex();
    }
}