import common;

struct VertexIn {
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
}

[shader("vertex")]
float4 vert(
    VertexIn vs_in,
    ParameterBlock<CameraData> light_camera_data,
    ParameterBlock<ObjectData> object_data,
) : SV_Position {
    return mul(mul(mul(float4(vs_in.position, 1.0), object_data.model), light_camera_data.view), light_camera_data.projection);
}