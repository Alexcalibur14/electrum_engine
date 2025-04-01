# version 430

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec3 normal;

void main()
{
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
