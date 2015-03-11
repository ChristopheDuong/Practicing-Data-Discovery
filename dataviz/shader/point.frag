
precision mediump float;

varying float time;
varying float transitNearby;

void main() {
	// set pixels in points to green
	if (transitNearby > 0.5) {
	    gl_FragColor.r = 0.0;
	    gl_FragColor.g = 0.0;
		gl_FragColor.b = 0.7;
	    gl_FragColor.a = time;
    } 
}