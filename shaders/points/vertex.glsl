#version 410

in float position_x;
in float position_y;
in float color;

uniform float f_view_transition;
uniform float point_size;
uniform highp int observation_type;
uniform float opacity;
uniform mat4 view;
uniform mat4 view_new;
uniform mat4 projection;

out float f_color;
out float f_opacity;
out vec2 vertex_coordinate;

void main() {
	vec2 position = vec2(position_x, position_y);
	vec4 p_view_old = projection * view * vec4(position, 0.0, 1.0);
	vec4 p_view_new = projection * view_new * vec4(position, 0.0, 1.0);
	gl_Position = mix(p_view_old, p_view_new, smoothstep(0.0, 1.0, f_view_transition));
	
	gl_PointSize = point_size;
	
	f_color = color;
	f_opacity = opacity;
}