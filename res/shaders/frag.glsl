#version 450

layout(location = 0) in vec3 vert_position;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(set = 0, binding = 2) uniform PointLight {
    vec3 position;
    vec3 colour;
    float strength;
} point_light;

void main() {
    float gamma = 2.2;

    // ambiant lighting
    float ambiant_strength = 0.005;
    vec3 ambiant_colour = point_light.colour * ambiant_strength;

    vec4 obj_colour = texture(texSampler, fragTexCoord);

    // diffuse lighting
    vec3 light_dir = normalize(point_light.position - vert_position);

    float diffuse_strength = max(dot(inNormal, light_dir), 0.0);
    vec3 diffuse_colour = point_light.colour * diffuse_strength;


    vec3 result = (ambiant_colour + diffuse_colour) * obj_colour.rgb;

    outColor = vec4(result * gamma, obj_colour.a);
}
