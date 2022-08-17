#version 460

void main()
{
	// quad on the lower right
	gl_Position = vec4((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2, 0, 1);
}
