
attribute vec4 worldCoord;
uniform vec2 pointSize;
uniform vec2 timeInterval;

uniform mat4 mapMatrix;
varying float value;

void main() {
	value = worldCoord.z;
	
	if (timeInterval.x < value && value < timeInterval.y) {
		// transform world coordinate by matrix uniform variable
		gl_Position = mapMatrix * worldCoord;
		float factor = (value - timeInterval.x) / (timeInterval.y - timeInterval.x); 
		gl_PointSize = mapMatrix[0].x * factor * pointSize.x;
    }
}
