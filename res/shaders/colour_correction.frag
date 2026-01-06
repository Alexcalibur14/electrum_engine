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
    float hue_shift;
};

layout(set = 1, binding = 0) uniform ColourCorrectionData {
    Levels levels;
};

vec3 calc_levels(vec3 in_colour, Levels levels) {
    vec3 norm_colour = ((in_colour - levels.in_black) / (levels.in_white - levels.in_black));
    vec3 colour = pow(norm_colour, vec3(levels.gamma));
    return colour * (levels.out_white - levels.out_black) + levels.out_black;
}

vec3 rgb_to_oklab(vec3 rgb) {
    const mat3 rgb_to_cone = mat3(
        0.4121656120, 0.2118591070, 0.0883097947,
        0.5362752080, 0.6807189584, 0.2818474174,
        0.0514575653, 0.1074065790, 0.6302613616
    );
    const mat3 cone_to_oklab = mat3(
         0.2104542553,  1.9779984951,  0.0259040371,
         0.7936177850, -2.4285922050,  0.7827717662,
        -0.0040720468,  0.4505937099, -0.8086757660
    );

    vec3 cone = rgb_to_cone * rgb;

    vec3 nl_cone = sign(cone) * pow(abs(cone), vec3(1.0 / 3.0));

    return cone_to_oklab * nl_cone;
}

vec3 oklab_to_rgb(vec3 oklab) {
    mat3 oklab_to_cone = mat3(
        1.000000000,  1.000000000,  1.000000000,
        0.396337777, -0.105561346, -0.089484178,
        0.215803757, -0.063854173, -1.291485548
    );
    mat3 cone_to_rgb = mat3(
         4.076724529, -1.268143773, -0.004111989,
        -3.307216883,  2.609332323, -0.703476310,
         0.230759054, -0.341134429,  1.706862569
    );

    vec3 nl_cone = oklab_to_cone * oklab;
    vec3 cone = nl_cone * nl_cone * nl_cone;
    return cone_to_rgb * cone;
}

vec3 oklab_to_oklch(vec3 oklab) {
    float c = sqrt((oklab.y * oklab.y) + (oklab.z * oklab.z));
    float h = atan(oklab.z, oklab.g);
    return vec3(oklab.r, c, h);
}

vec3 oklch_to_oklab(vec3 oklch) {
    float a = oklch.g * cos(oklch.b);
    float b = oklch.g * sin(oklch.b);
    return vec3(oklch.r, a, b);
}

void main() {
    vec2 uv = (in_position + 1) / 2;
    vec3 in_colour = texture(in_texture, uv).rgb;

    vec3 mapped = calc_levels(in_colour, levels);

    vec3 oklch = oklab_to_oklch(rgb_to_oklab(mapped));
    oklch.b += levels.hue_shift;
    vec3 final = oklab_to_rgb(oklch_to_oklab(oklch));

    out_colour = vec4(final, 1.0);
}
