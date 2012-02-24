#version 150
#extension GL_EXT_geometry_shader4 : enable

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in VertexData {
    vec3 position;
    vec2 texcoord;
} VertexIn[3];

out VertexData {
    vec2 texcoord;
    vec3 normal;
    vec3 position;
} VertexOut;
 
void main() {
    vec4 vec1 = gl_in[1].gl_Position - gl_in[0].gl_Position;
    vec4 vec2 = gl_in[2].gl_Position - gl_in[0].gl_Position;
    vec3 normal = cross(vec1.xyz, vec2.xyz);

    for(int i = 0; i < gl_VerticesIn; i++) {
        // copy attributes
        gl_Position = gl_in[i].gl_Position;
        VertexOut.normal = normal;
        VertexOut.texcoord = VertexIn[i].texcoord;
        VertexOut.position = VertexIn[i].position;

        // done with the vertex
        EmitVertex();
    }
}