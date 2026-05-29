import common;

struct VertexOut {
    float4 position : SV_Position;
    float2 frag_position;
}

[shader("vertex")]
VertexOut frag(uint vertex_id: SV_VulkanVertexID) {
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

struct InTexture {
    Sampler2D<float4> in_texture;
}

[shader("fragment")]
float4 frag(
    VertexOut vs_out,
    ParameterBlock<ColorCorrectionData> color_correction_data,
    ParameterBlock<InTexture> in_texture,
) : SV_Target {
    float2 uv = (vs_out.frag_position + 1.0) / 2.0;
    float3 in_color = in_texture.in_texture.Sample(uv).rgb;

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
