struct VertexIn {
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
}

struct VertexOut {
    float4 position: SV_Position;
    float3 frag_position;
    float3 normal;
    float3 color;
    float2 uv;
    float4 light_space_vert;
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

struct MaterialData {
    float3 color;
    float specular;
}

struct LightCameraData {
    float4x4 view;
    float4x4 projection;
    float3 position;
}

struct LightData {
    float3 position;
    float strength;
    float3 direction;
    int light_type;
    float3 color;
}

struct ShadowMap {
    Sampler2DShadow shadow_map;
}

ParameterBlock<CameraData> camera_data;

[shader("vertex")]
VertexOut main(
    VertexIn vs_in,
    ParameterBlock<ObjectData> object_data,
    ParameterBlock<LightCameraData> light_camera,
) {
    VertexOut vs_out;

    float4 world_space = mul(float4(vs_in.position, 1.0), object_data.model);
    vs_out.position = mul(mul(world_space, camera_data.view), camera_data.projection);
    
    vs_out.frag_position = world_space.xyz;
    vs_out.normal = mul(object_data.normal, float4(vs_in.normal, 0.0)).xyz;
    vs_out.color = vs_in.color;
    vs_out.uv = vs_in.uv;
    vs_out.light_space_vert = mul(mul(world_space, light_camera.view), light_camera.projection);

    return vs_out;
}


[shader("fragment")]
float4 main(
    VertexOut frag_in,
    ParameterBlock<LightData> light,
    ParameterBlock<MaterialData> material,
    ParameterBlock<ShadowMap> shadow_map,
) : SV_Target {
    float light_distance = distance(light.position, frag_in.frag_position);
    float attenuation = 1.0 / pow(light_distance, 2.0);

    // Ambiant
    float ambiant_strength = 0.1;
    float3 ambiant_color = light.color * ambiant_strength;

    // Diffuse
    float diffuse_strength = max(dot(frag_in.normal, light.direction), 0.0);
    float3 diffuse_color = diffuse_strength * light.color;

    // Specular
    float3 view_dir = normalize(camera_data.position - frag_in.frag_position);
    float3 half_dir = normalize(light.direction + view_dir);
    float specular_strength = pow(max(dot(frag_in.normal, half_dir), 0.0), 32);
    float3 specular_color = specular_strength * material.specular * light.color;

    float shadow = get_shadow(shadow_map.shadow_map, frag_in.light_space_vert, frag_in.normal, light.direction);

    return float4(material.color * (ambiant_color + (diffuse_color + specular_color) * shadow) * light.strength * attenuation, 1.0);
}

float get_shadow(Sampler2DShadow shadow_map, float4 vert, float3 normal, float3 light_dir) {
    float3 proj_coords = vert.xyz / vert.w;
    float2 uv = proj_coords.xy * 0.5 + 0.5;

    float closest_depth = shadow_map.Sample(uv);
    float current_depth = proj_coords.z;

    float shadow = closest_depth > current_depth ? 0.0 : 1.0;

    return shadow;
}
