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
float4 vert(
    VertexIn vs_in,
    ParameterBlock<CameraData> light_camera_data,
    ParameterBlock<ObjectData> object_data,
) : SV_Position {
    return mul(mul(mul(float4(vs_in.position, 1.0), object_data.model), light_camera_data.view), light_camera_data.projection);
}