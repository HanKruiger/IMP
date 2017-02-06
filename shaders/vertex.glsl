#version 410

in float position_x;
in float position_y;
in float color;

uniform float point_size;
uniform float opacity;
uniform mat4 view;
uniform mat4 projection;

out float f_color;
out float f_opacity;

void main() {
	vec2 position = vec2(position_x, position_y);
	gl_Position = projection * view * vec4(position, 0.0, 1.0);
	gl_PointSize = point_size;
	f_color = color;
	f_opacity = opacity;
}