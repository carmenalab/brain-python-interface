#version 110

uniform float nearclip;
uniform float farclip;

void main() {
    //gl_FragColor[0] = phong();
    float A = -(farclip + nearclip) / (farclip - nearclip);
    float B = -2.*farclip*nearclip / (farclip - nearclip);
    float z = (-A*vposition.z + B) / -vposition.z;

    gl_FragColor = vec4(normalize(vnormal).xyz, (z+1.0)*0.5);
}