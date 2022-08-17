#version 460

layout (location = 0) out vec4 color;

void main()
{
	// this endless lopp results in a device lost error on the host
	while (true)
		color = vec4(1, 0, 0, 1);
}
