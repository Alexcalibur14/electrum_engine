#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUV;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view;
    mat4 proj;
    vec3 position;
} camera;

layout(set = 1, binding = 0) uniform ModelViewProjection {
    mat4 model_matrix;
    mat4 normal_matrix;
};

void main() {
    gl_Position = camera.proj * camera.view * model_matrix * vec4(inPosition, 1.0);
}
