#version 410

uniform mat4 u_view_old;
uniform mat4 u_view_new;
uniform mat4 u_projection;
uniform float u_point_size;
uniform float u_opacity_regular;
uniform float u_opacity_representatives;
uniform float u_time;

in vec2 v_position_old;
in vec2 v_position_new;
in uint v_has_old;
in uint v_has_new;
in uint v_is_repr_old;
in uint v_is_repr_new;
in float v_colour;

out float f_colour;
flat out uint f_has_old;
flat out uint f_has_new;
flat out uint f_is_repr;
flat out uint f_is_repr_new;

void main() {
	vec2 position;
	if (v_has_old != 0 && v_has_new == 0){
		position = v_position_old;
	} else if (v_has_old != 0 && v_has_new != 0) {
		position = mix(v_position_old, v_position_new, smoothstep(0.0, 1.0, u_time));
	} else if (v_has_old == 0 && v_has_new != 0) {
		position = v_position_new;
	} else{
		position = vec2(0.0, 0.0);
	}
	
	vec4 projected_old = u_projection * u_view_old * vec4(position, 0.0, 1.0);
	vec4 projected_new = u_projection * u_view_new * vec4(position, 0.0, 1.0);
	
	gl_Position = mix(projected_old, projected_new, smoothstep(0.0, 1.0, u_time));

	
	gl_PointSize = u_point_size;
	
	f_colour = v_colour;

	f_is_repr = v_is_repr_old;
	f_is_repr_new = v_is_repr_new;
	f_has_old = v_has_old;
	f_has_new = v_has_new;
}