#version 450

layout(location = 0) in vec2 in_position;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 0) uniform sampler2D in_texture;

struct Levels {
    float in_black;
    float in_white;
    float out_black;
    float out_white;
    float gamma;
};

layout(set = 1, binding = 0) uniform ColourCorrectionData {
    Levels levels;
};

vec3 calc_levels(vec3 in_colour, Levels levels) {
    vec3 norm_colour = ((in_colour - levels.in_black) / (levels.in_white - levels.in_black));
    // not sure about the 255s here but it does make a big difference
    vec3 colour = pow(norm_colour * 255, vec3(levels.gamma)) / 255;
    return colour * (levels.out_white - levels.out_black) + levels.out_black;
}

void main() {
    vec2 uv = (in_position + 1) / 2;
    vec3 in_colour = texture(in_texture, uv).rgb;

    vec3 mapped = calc_levels(in_colour, levels);

    out_colour = vec4(mapped, 1.0);
}
