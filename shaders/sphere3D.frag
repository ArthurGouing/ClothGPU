#version 330 core

// Uniforms
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 colors[8];
uniform vec3 center;
uniform float radius;
uniform mat3 rotation;

// Data in
in vec2 coord;

// Data out
out vec4 color;

void main() {
    //color = vec4(coord, 0, 1);
    float coord_2 = dot( coord, coord );
    if (coord_2 > 1.0) discard;   // discard pixels outside circle

    vec3 N = vec3( coord, sqrt(1 - coord_2) ) ;

    vec3 normal_w = transpose(rotation) * mat3( transpose(viewMatrix) ) * N ;
    vec3 pos_w = center + transpose(rotation) * N * radius ;

    vec3 pos = ( viewMatrix * vec4( pos_w, 1)).xyz ;
    vec3 normal = normalize( mat3(viewMatrix) * rotation * normal_w );
	
	vec4 ambientColor = vec4(0.4 , 0.1, 0.1, 1. );
	
	if( normal_w.z > 0 ) {
        if( normal_w.y > 0 ) {
            if( normal_w.x > 0) {
                ambientColor = vec4(colors[0], 1.);
            }
            else {
                ambientColor = vec4(colors[1], 1.);
            }
        }
        else {
            if( normal_w.x > 0) {
                ambientColor = vec4(colors[2], 1.);
            }
            else {
                ambientColor = vec4(colors[3], 1.);
            }
        }
	}
    else {
        if( normal_w.y > 0 ) {
            if( normal_w.x > 0) {
                ambientColor = vec4(colors[4], 1.);
            }
            else {
                ambientColor = vec4(colors[5], 1.);
            }
        }
        else {
            if( normal_w.x > 0) {
                ambientColor = vec4(colors[6], 1.);
            }
            else {
                ambientColor = vec4(colors[7], 1.);
            }
        }
    }

    color = ambientColor;
}