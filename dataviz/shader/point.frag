
precision mediump float;

varying float value;

void main() {
	// set pixels in points to green
    gl_FragColor.r = 0.0;
    gl_FragColor.g = 1.0;
	gl_FragColor.b = 0.0;
    gl_FragColor.a = 1.0;
}