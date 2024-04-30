#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform ModelViewProjection {
    mat4 model_matrix;
    mat4 mvp_matrix;
    mat4 normal_matrix;
} model_data;

layout(set = 1, binding = 3) uniform CameraData {
    vec3 position;
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 outNormals;

void main() {
    gl_Position = model_data.mvp_matrix * vec4(inPosition, 1.0);
    
    outPosition = (model_data.model_matrix * vec4(inPosition, 1.0)).xyz;
    fragTexCoord = inTexCoord;
    outNormals = (model_data.normal_matrix * vec4(inNormal, 0.0)).xyz;
}
