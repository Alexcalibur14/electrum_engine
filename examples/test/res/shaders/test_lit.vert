#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform ModelViewProjection {
    mat4 model_matrix;
    mat4 normal_matrix;
};

layout(set = 2, binding = 0) uniform CameraData {
    vec3 position;
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 outNormals;
layout(location = 3) out vec3 camera_position;

void main() {
    gl_Position = camera.proj * camera.view * model_matrix * vec4(inPosition, 1.0);
    
    outPosition = (model_matrix * vec4(inPosition, 1.0)).xyz;
    fragTexCoord = inTexCoord;
    outNormals = (normal_matrix * vec4(inNormal, 0.0)).xyz;
    camera_position = camera.position;
}
