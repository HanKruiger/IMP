#version 410

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;
uniform float u_opacity_regular;
uniform float u_opacity_representatives;
uniform float u_fadein_interpolation;

in vec2 v_position_old;
in vec2 v_position_new;
in uint v_has_old;
in uint v_has_new;
in uint v_is_repr;
in uint v_is_repr_new;
in float v_color;
in float v_dist_from_repr_old;
in float v_dist_from_repr_new;

out float f_color;
flat out float f_dist_from_repr;
flat out uint f_has_old;
flat out uint f_has_new;
flat out uint f_is_repr;
flat out uint f_is_repr_new;

void main() {
	vec2 position;
	if (v_has_old != 0 && v_has_new == 0){
		position = v_position_old;
	} else if (v_has_old != 0 && v_has_new != 0) {
		position = mix(v_position_old, v_position_new, smoothstep(0.0, 1.0, u_fadein_interpolation));
	} else if (v_has_old == 0 && v_has_new != 0) {
		position = v_position_new;
	} else{
		position = vec2(0.0, 0.0);
	}
	
	gl_Position = u_projection * u_view * vec4(position, 0.0, 1.0);
	
	gl_PointSize = u_point_size;
	
	f_color = v_color;

	f_is_repr = v_is_repr;
	f_is_repr_new = v_is_repr_new;
	f_has_old = v_has_old;
	f_has_new = v_has_new;
	f_dist_from_repr = mix(v_dist_from_repr_old, v_dist_from_repr_new, smoothstep(0.0, 1.0, u_fadein_interpolation));
}