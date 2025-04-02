# version 430

layout(location = 0) out vec4 color;

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, set = 1, binding = 0) uniform subpassInput inputDepth;

void main() {
    float depth_sq = subpassLoad(inputDepth).r * subpassLoad(inputDepth).r;
    color = subpassLoad(inputColor) * depth_sq;
}
