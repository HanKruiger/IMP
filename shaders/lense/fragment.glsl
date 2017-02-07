#version 410

const float lens_thickness = 2;

uniform float u_radius;
uniform vec2 u_center;

/* Because Qt uses this origin too */
layout(origin_upper_left) in vec4 gl_FragCoord;

out vec4 out_color;

void main() {
    float dist_from_center = distance(gl_FragCoord.xy, u_center);
    if (dist_from_center > u_radius || dist_from_center < u_radius - lens_thickness) {
        discard;
    }

    float delta_dist_from_center = fwidth(dist_from_center);
    float alpha =
        smoothstep(u_radius - lens_thickness, u_radius - lens_thickness + delta_dist_from_center, dist_from_center) *
        (1 - smoothstep(u_radius - delta_dist_from_center, u_radius, dist_from_center));
	out_color = vec4(0.0, 0.0, 0.0, alpha);
}
