#version 410

in vec2 position;
in vec4 color;
uniform float point_size;
uniform mat4 camera;
out vec4 f_color;

void main()
{
   gl_Position = camera * vec4(position, 0, 1);
   gl_PointSize = point_size;
   f_color = color;
}