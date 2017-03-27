#version 410

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;
uniform bool u_is_representative;
uniform float u_opacity_regular;
uniform float u_opacity_representatives;
uniform float u_fadein_interpolation;

in vec2 v_position;
in vec2 v_position_new;
in uint v_has_old;
in uint v_has_new;
in float v_color;
in float v_dist_from_repr;

out float f_color;
out float f_opacity;
out float f_dist_from_repr;

void main() {
	vec2 position = v_position;
	// if (v_has_old != 0 && v_has_new == 0){
	// 	position = v_position;
	// } else if (v_has_old != 0 && v_has_new != 0) {
	// 	position = mix(v_position, v_position_new, u_fadein_interpolation);
	// } else if (v_has_old == 0 && v_has_new != 0) {
	// 	position = v_position_new;
	// }
	
	gl_Position = u_projection * u_view * vec4(position, 0.0, 1.0);
	
	gl_PointSize = u_point_size;
	
	f_color = v_color;
	if (!u_is_representative)
		f_opacity = u_opacity_regular;
	else
		f_opacity = u_opacity_representatives;

	f_dist_from_repr = v_dist_from_repr;
}