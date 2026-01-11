#version 120

// Uniforms provided by Panda3D
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;

// Inputs
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;

// Outputs to Fragment Shader
varying vec2 texcoord;
varying vec3 normal;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
    normal = normalize(p3d_NormalMatrix * p3d_Normal);
}
