
attribute vec4 worldCoord;
uniform vec2 pointSize;
uniform vec2 timeInterval;

uniform mat4 mapMatrix;
varying float time;
varying float transitNearby;

void main() {
	time = worldCoord.z;
	transitNearby = worldCoord.a;
	vec4 position = vec4(worldCoord.x, worldCoord.y, 1.0, 1.0);
	if (timeInterval.x < time && time < timeInterval.y) {
		// transform world coordinate by matrix uniform variable
		gl_Position = mapMatrix * position;
		//float factor = (time - timeInterval.x) / (timeInterval.y - timeInterval.x); 
		float factor = 1.0;
		gl_PointSize = mapMatrix[0].x * factor * pointSize.x;
    }
}
