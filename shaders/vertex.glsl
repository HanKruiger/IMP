#version 410

in vec2 position;
in float color;

uniform float point_size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out float f_color;

void main() {
   gl_Position = projection * view * model * vec4(position, 0.0, 1.0);
   gl_PointSize = point_size;
   f_color = color;
}