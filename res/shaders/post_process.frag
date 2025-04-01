# version 430

layout(location = 0) out vec4 color;

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;

void main() {
    color = subpassLoad(inputColor);
}
