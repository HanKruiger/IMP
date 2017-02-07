#version 410

uniform float u_radius;
uniform vec2 u_center;

in vec2 a_position;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
