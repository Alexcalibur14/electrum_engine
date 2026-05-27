struct VertexIn {
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
}

struct VertexOut {
    float4 position : SV_Position;
    float3 color;
}

struct CameraData {
    float4x4 view;
    float4x4 projection;
    float3 position;
}

struct ObjectData {
    float4x4 model;
    float4x4 normal;
}

ParameterBlock<CameraData> camera_data;

[shader("vertex")]
VertexOut vert(
    VertexIn vs_in,
    ParameterBlock<ObjectData> object_data,
) {
    VertexOut vs_out;
    vs_out.position = mul(mul(mul(float4(vs_in.position, 1.0), object_data.model), camera_data.view), camera_data.projection);
    vs_out.color = vs_in.color;
    return vs_out;
}

[shader("fragment")]
float4 frag(
    VertexOut frag_in,
) : SV_Target {
    return float4(frag_in.color, 1.0);
}