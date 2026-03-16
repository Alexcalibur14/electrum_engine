#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColour;
layout(location = 3) out vec2 outUV;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_matrix;
    mat4 projection_matrix;
    vec3 position;
};

layout(set = 1, binding = 0) uniform ModelViewProjection {
    mat4 model_matrix;
    mat4 normal_matrix;
};

void main() {
    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(inPosition, 1.0);
    outPosition = (model_matrix * vec4(inPosition, 1.0)).xyz;
    outNormal = (normal_matrix * vec4(inNormal, 0.0)).xyz;
    outColour = inColour;
    outUV = inUV;
}
