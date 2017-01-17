#version 410

in vec2 position;
in vec4 color;
uniform float point_size;
out vec4 f_color;

void main()
{
   gl_Position = vec4(position, 0, 1);
   gl_PointSize = point_size;
   f_color = color;
}