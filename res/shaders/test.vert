#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColour;
layout(location = 2) out vec2 outUV;

layout(set = 0, binding = 0) uniform ModelViewProjection {
    mat4 model_matrix;
};

void main() {
    gl_Position = model_matrix * vec4(inPosition, 1.0);
    outNormal = inNormal;
    outColour = inColour;
    outUV = inUV;
}
