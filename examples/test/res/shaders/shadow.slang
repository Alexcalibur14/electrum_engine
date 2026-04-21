struct VertexIn {
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
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
float4 main(
    VertexIn vs_in,
    ParameterBlock<CameraData> camera_data,
    ParameterBlock<ObjectData> object_data,
) : SV_Position {
    return mul(mul(mul(float4(vs_in.position, 1.0), object_data.model), camera_data.view), camera_data.projection);
}