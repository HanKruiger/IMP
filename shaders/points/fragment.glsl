#version 410

float colormap_red(float x);
float colormap_green(float x);
float colormap_blue(float x);
vec3 colormap(float x);

uniform float u_fadein_interpolation;
uniform float u_opacity_regular;
uniform float u_opacity_representatives;

in float f_color;
flat in float f_dist_from_repr;
flat in uint f_is_repr;
flat in uint f_is_repr_new;
flat in uint f_has_old;
flat in uint f_has_new;

out vec4 o_color;

void main() {

	float dist_from_border;
	if (f_is_repr_new == 0) {
		vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
		float radius = dot(circCoord, circCoord);

		/* Discard fragment if outside circle */
		if (radius > 1.0) {
			discard;
		}

		dist_from_border = 1.0 - radius;
	} else {
		dist_from_border = min(min(gl_PointCoord.s, 1.0 - gl_PointCoord.s), min(gl_PointCoord.t, 1.0 - gl_PointCoord.t));
	}

	/* Smooth transition on the border of the disk */
	float delta_dist_from_border = fwidth(dist_from_border);
	float alpha = smoothstep(0.0, delta_dist_from_border, dist_from_border);

	float old_opacity = u_opacity_regular;
	float new_opacity = u_opacity_regular;
	if (f_is_repr != 0)
		old_opacity = u_opacity_representatives;
	if (f_is_repr_new != 0)
		new_opacity = u_opacity_representatives;

	alpha *= mix(old_opacity, new_opacity, u_fadein_interpolation);

	/* Fade in new points */
	if (f_has_new != 0 && f_has_old == 0)
		alpha *= clamp(u_fadein_interpolation, 0.0, 1.0);
	else if (f_has_new == 0 && f_has_old != 0)
		alpha *= clamp(1.0 - u_fadein_interpolation, 0.0, 1.0);

	o_color = vec4(colormap(f_color), alpha);
}

float colormap_red(float x) {
	if (x < 0.7) {
		return 4.0 * x - 1.5;
	} else {
		return -4.0 * x + 4.5;
	}
}

float colormap_green(float x) {
	if (x < 0.5) {
		return 4.0 * x - 0.5;
	} else {
		return -4.0 * x + 3.5;
	}
}

float colormap_blue(float x) {
	if (x < 0.3) {
	   return 4.0 * x + 0.5;
	} else {
	   return -4.0 * x + 2.5;
	}
}

vec3 colormap(float x) {
	float r = clamp(colormap_red(x), 0.0, 1.0);
	float g = clamp(colormap_green(x), 0.0, 1.0);
	float b = clamp(colormap_blue(x), 0.0, 1.0);

	return vec3(r, g, b);
}