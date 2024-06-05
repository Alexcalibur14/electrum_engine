#version 450

layout(location = 0) in vec3 vert_position;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 camera_position;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D texSampler;

struct PointLight {
    vec3 position;
    vec3 colour;
    float strength;
};

layout(set = 2, binding = 0) buffer PointLights {
    PointLight lights[];
} point_lights;

const float GAMMA = 2.2;
const float AMBIANT_STRENGTH = 0.005;

void main() {
    vec3 total_light_colour = vec3(0.0);

    for(int i = 0; i < point_lights.lights.length(); i++) {
        PointLight point_light = point_lights.lights[i];
        if (point_light.strength == 0.0) {continue;}

        float light_distance = distance(point_light.position, vert_position);
        float attenuation = 1.0 / pow(light_distance, 2.0);

        // ambiant lighting
        vec3 ambiant_colour = point_light.colour * AMBIANT_STRENGTH;


        // diffuse lighting
        vec3 light_dir = normalize(point_light.position - vert_position);

        float diffuse_strength = max(dot(inNormal, light_dir), 0.0);
        vec3 diffuse_colour = point_light.colour * diffuse_strength;

        // specular lighting
        vec3 view_dir = normalize(camera_position - vert_position);
        vec3 half_dir = normalize(view_dir + light_dir);

        float specular_strength = pow(max(dot(view_dir, half_dir), 0.0), 32.0);
        vec3 specular_colour = specular_strength * point_light.colour;

        total_light_colour += (ambiant_colour + diffuse_colour + specular_colour) * point_light.strength * attenuation;
    }

    vec4 obj_colour = texture(texSampler, fragTexCoord);

    // output
    vec3 result = total_light_colour * obj_colour.rgb;

    vec3 gamma_correction = pow(result, vec3(1.0 / GAMMA));

    outColor = vec4(gamma_correction, obj_colour.a);
}
