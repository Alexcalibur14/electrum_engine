#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColour, 1.0);
}
