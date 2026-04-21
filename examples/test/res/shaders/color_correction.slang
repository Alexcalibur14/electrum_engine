struct VertexOut {
    float4 position : SV_Position;
    float2 frag_position;
}

[shader("vertex")]
VertexOut main(uint vertex_id: SV_VulkanVertexID) {
    static const float2 positions[3] = float2[](
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    );

    float2 pos = positions[vertex_id];
    VertexOut vs_out;
    vs_out.frag_position = pos;
    vs_out.position = float4(pos, 0.0, 1.0);
    return vs_out;
}

struct Levels {
    float in_black;
    float in_white;
    float out_black;
    float out_white;
    float gamma;
}

struct ColorCorrectionData {
    Levels levels;
    float hue_shift;
}

[shader("fragment")]
float4 main(
    VertexOut vs_out,
    ParameterBlock<ColorCorrectionData> color_correction_data,
    Sampler2D<float4> in_texture,
) : SV_Target {
    float2 uv = (vs_out.frag_position + 1.0) / 2.0;
    float3 in_color = in_texture.Sample(uv).rgb;

    float3 mapped = claculate_levels(in_color, color_correction_data.levels);

    float3 oklch = oklab_to_oklch(rgb_to_oklab(mapped));
    oklch.b += color_correction_data.hue_shift;
    float3 final = oklab_to_rbg(oklch_to_oklab(oklch));

    return float4(final, 1.0);
}

float3 claculate_levels(float3 in_color, Levels levels) {
    float3 norm_color = ((in_color - levels.in_black) / (levels.in_white - levels.in_black));
    float3 color = pow(norm_color, float3(levels.gamma));
    return color * (levels.out_white - levels.out_black) + levels.out_black;
}

float3 rgb_to_oklab(float3 rgb) {
    static const float3x3 rgb_to_cone = float3x3(
        0.4121656120, 0.2118591070, 0.0883097947,
        0.5362752080, 0.6807189584, 0.2818474174,
        0.0514575653, 0.1074065790, 0.6302613616
    );

    static const float3x3 cone_to_oklab = float3x3(
        0.2104542553, 1.9779984951, 0.0259040371,
        0.7936177850, -2.4285922050, 0.7827717662,
        -0.0040720468, 0.4505937099, -0.8086757660
    );

    float3 cone = mul(rgb, rgb_to_cone);

    float3 nl_cone = sign(cone) * pow(abs(cone), float3(1.0 / 3.0));

    return mul(nl_cone, cone_to_oklab);
}

float3 oklab_to_rbg(float3 oklab) {
    static const float3x3 oklab_to_cone = float3x3(
        1.000000000, 1.000000000, 1.000000000,
        0.396337777, -0.105561346, -0.089484178,
        0.215803757, -0.063854173, -1.291485548
    );

    static const float3x3 cone_to_rgb = float3x3(
        4.076724529, -1.268143773, -0.004111989,
        -3.307216883, 2.609332323, -0.703476310,
        0.230759054, -0.341134429, 1.706862569
    );

    float3 nl_cone = mul(oklab, oklab_to_cone);
    float3 cone = nl_cone * nl_cone * nl_cone;
    return mul(cone, cone_to_rgb);
}

float3 oklab_to_oklch(float3 oklab) {
    float c = sqrt((oklab.y * oklab.y) + (oklab.z * oklab.z));
    float h = atan2(oklab.z, oklab.g);
    return float3(oklab.r, c, h);
}

float3 oklch_to_oklab(float3 oklch) {
    float a = oklch.g * cos(oklch.b);
    float b = oklch.g * sin(oklch.b);
    return float3(oklch.r, a, b);
}