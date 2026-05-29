module common;

public struct CameraData {
    public float4x4 view;
    public float4x4 projection;
    public float3 position;
}

public struct ObjectData {
    public float4x4 model;
    public float4x4 normal;
}

public struct MaterialData {
    public float3 color;
    public float specular;
}

public struct LightCameraData {
    public float4x4 view;
    public float4x4 projection;
    public float3 position;
}

public struct LightData {
    public float3 position;
    public float strength;
    public float3 direction;
    public int light_type;
    public float3 color;
}

public struct Levels {
    public float in_black;
    public float in_white;
    public float out_black;
    public float out_white;
    public float gamma;
}

public struct ColorCorrectionData {
    public Levels levels;
    public float hue_shift;
}

public float3 rgb_to_oklab(float3 rgb) {
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

public float3 oklab_to_rbg(float3 oklab) {
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

public float3 oklab_to_oklch(float3 oklab) {
    float c = sqrt((oklab.y * oklab.y) + (oklab.z * oklab.z));
    float h = atan2(oklab.z, oklab.g);
    return float3(oklab.r, c, h);
}

public float3 oklch_to_oklab(float3 oklch) {
    float a = oklch.g * cos(oklch.b);
    float b = oklch.g * sin(oklch.b);
    return float3(oklch.r, a, b);
}