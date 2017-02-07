#version 410

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

in float f_color;
in float f_opacity;
in vec2 vertex_coordinate;
out vec4 out_color;

void main() {
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	float radius = dot(circCoord, circCoord);

	/* Discard fragment if outside circle */
	if (radius > 1.0) {
		discard;
	}

    /* Smooth transition on the border of the disk */
    float dist_from_border = 1.0 - radius;
    float delta_dist_from_border = fwidth(dist_from_border);
    float alpha = f_opacity * smoothstep(0.0, delta_dist_from_border, dist_from_border);
	out_color = vec4(colormap(f_color), alpha);
}
