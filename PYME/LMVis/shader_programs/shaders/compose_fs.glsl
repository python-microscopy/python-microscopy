#version 120

uniform sampler2D accum_t; // sum(rgb * a, a)
uniform sampler2D reveal_t; // prod(1 - a)

varying vec2 vTexcoord;

void main(void)
{
    vec4 accum = texture2D(accum_t, vTexcoord);
    float alpha = accum.a;
    accum.a = texture2D(reveal_t, vTexcoord).r; // GL_RED
    //if (alpha >= 1.0f) { // Save the blending and color texture fetch cost
    //    discard;
    //}
    //gl_FragColor = vec4(accum.rgb / clamp(accum.a, 1e-4f, 5e4f), (1.0f - alpha) );
    gl_FragColor = vec4(accum.rgb / clamp(accum.a, 1e-4f, 5e4f), (alpha) );
    //gl_FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

#version 330

uniform sampler2D accum_t; // sum(rgb * a, a)
uniform sampler2D reveal_t; // prod(1 - a)

in vec2 vTexcoord;

out vec4 fragColor;

void main(void)
{
    vec4 accum = texture(accum_t, vTexcoord);
    float alpha = accum.a;
    accum.a = texture(reveal_t, vTexcoord).r; // GL_RED
    //if (alpha >= 1.0f) { // Save the blending and color texture fetch cost
    //    discard;
    //}
    //gl_FragColor = vec4(accum.rgb / clamp(accum.a, 1e-4f, 5e4f), (1.0f - alpha) );
    fragColor = vec4(accum.rgb / clamp(accum.a, 1e-4f, 5e4f), (alpha) );
    //gl_FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}