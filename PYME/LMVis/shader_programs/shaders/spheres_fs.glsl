#version 120

// This fragmentshader takes the given color and displays it.
// It's transparency is determined by a submitted texture. That way the brightness is decreasing with the distance to the \
// center of the point.

uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec4 light_position;
uniform float shininess;
uniform vec4 view_vector;

varying float vis;

void main()
{
    vec3 N;
    float m;

    if (vis < 0.5) discard;

    N.xy = gl_PointCoord.xy*2.0 -vec2(1.0);
    m = dot(N.xy, N.xy);
    if (m > 1.0) discard;
    N.z = sqrt(1.0 - m);

    vec4 inputColor = gl_Color;
    vec4 white = vec4(1.0, 1.0, 1.0, gl_Color.a);
    vec4 ambient = inputColor * light_ambient;
    //direction to the lightsource
    vec3 lightsource = normalize(vec3(light_position));

    float diffuseLight = abs(dot(lightsource, N));//, 0.0);
    vec4 diffuse = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    //vec4 diff = vec4(0.0, 0.0, 0.0, 1.0);
    if (diffuseLight > 0){
        diffuse = diffuseLight * inputColor * light_diffuse;
        vec3 H = normalize(light_position.xyz + view_vector.xyz);
        float specLight = pow(abs(dot(H, N)), shininess);
        vec4 spec = white * light_specular;
        specular = specLight * spec;
    }
    gl_FragColor = ambient + diffuse + specular;

    //float alpha = gl_Color.a * texture2D(tex2D, gl_PointCoord).x;
    //gl_FragColor = vec4(gl_Color.xyz, alpha);
}