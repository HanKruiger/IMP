#version 410

in vec4 f_color;
out vec4 out_color;

void main()
{
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	float radius = dot(circCoord, circCoord);

	// Discard fragment if outside circle
	if (radius > 1.0) {
		discard;
	}
	out_color = f_color;
}
