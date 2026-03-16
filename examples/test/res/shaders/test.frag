#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 in_light_space_vert;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_matrix;
    mat4 projection_matrix;
    vec3 position;
} camera;

layout(set = 2, binding = 0) uniform MaterialData {
    vec3 colour;
    float specular;
} material;

layout(set = 3, binding = 0) uniform LightData {
    vec3 position;
    float strength;
    vec3 direction;
    int light_type;
    vec3 colour;
} light;

layout(set = 4, binding = 0) uniform sampler2D shadow_map;

float get_shadow(vec4 light_space_vert, vec3 normal, vec3 light_dir) {
    vec3 proj_coords = light_space_vert.xyz / light_space_vert.w;
    vec2 uv = proj_coords.xy * 0.5 + 0.5;

    float closest_depth = texture(shadow_map, uv).r;
    float current_depth = proj_coords.z;

    float shadow = closest_depth > current_depth ? 0.0 : 1.0;

    return shadow;
}

void main() {
    float light_distance = distance(light.position, inPosition);
    float attenuation = 1.0 / pow(light_distance, 2.0);

    // Ambiant Component
    float ambiant_strength = 0.1;
    vec3 ambiant_colour = light.colour * ambiant_strength;

    // Diffuse Component
    float diffuse_strength = max(dot(inNormal, light.direction), 0.0);
    vec3 diffuse_colour = diffuse_strength * light.colour;

    // Specular Component
    vec3 view_dir = normalize(camera.position - inPosition);
    vec3 half_dir = normalize(light.direction + view_dir);
    float specular_strength = pow(max(dot(inNormal, half_dir), 0.0), 32);
    vec3 specular_colour = specular_strength * material.specular * light.colour;

    float shadow = get_shadow(in_light_space_vert, inNormal, light.direction);

    outColor = vec4(material.colour * (ambiant_colour + (diffuse_colour + specular_colour) * shadow) * light.strength * attenuation, 1.0);
}
