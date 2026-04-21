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

[shader("vertex")]
VertexOut main(
    VertexIn vs_in,
    ParameterBlock<CameraData> camera_data,
    ParameterBlock<ObjectData> object_data,
) {
    VertexOut vs_out;
    vs_out.position = mul(mul(mul(float4(vs_in.position, 1.0), object_data.model), camera_data.view), camera_data.projection);
    vs_out.color = vs_in.color;
    return vs_out;
}