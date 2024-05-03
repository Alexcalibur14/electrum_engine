#version 450

layout(location = 0) in vec3 vert_position;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 camera_position;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(set = 0, binding = 2) uniform PointLight {
    vec3 position;
    vec3 colour;
    float strength;
} point_light;

const float GAMMA = 2.2;
const float AMBIANT_STRENGTH = 0.005;

void main() {
    float light_distance = distance(point_light.position, vert_position);
    float attenuation = 1.0 / pow(light_distance, 2.0);

    // ambiant lighting
    vec3 ambiant_colour = point_light.colour * AMBIANT_STRENGTH;

    vec4 obj_colour = texture(texSampler, fragTexCoord);

    // diffuse lighting
    vec3 light_dir = normalize(point_light.position - vert_position);

    float diffuse_strength = max(dot(inNormal, light_dir), 0.0);
    vec3 diffuse_colour = point_light.colour * diffuse_strength;

    // specular lighting
    vec3 view_dir = normalize(camera_position - vert_position);
    vec3 half_dir = normalize(view_dir + light_dir);

    float specular_strength = pow(max(dot(view_dir, half_dir), 0.0), 32.0);
    vec3 specular_colour = specular_strength * point_light.colour;

    // output
    vec3 result = (ambiant_colour + diffuse_colour + specular_colour) * obj_colour.rgb;

    vec3 gamma_correction = pow(result, vec3(1.0 / GAMMA));

    outColor = vec4(gamma_correction, obj_colour.a);
}
