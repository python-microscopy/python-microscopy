#version 120

varying vec4 vertexColor;
varying float depth;
varying float vis;

void main(void)
{
    if (vis < .5) discard;

    float alpha = vertexColor.a;

    // the object lies between -40 and -60 z coordinates
    //float weight = pow(alpha + 0.01f, 4.0f) + max(0.01f, min(3000.0f, 0.3f / (0.00001f + pow(abs(depth) / 200.0f, 4.0f))));

    //float weight = pow(alpha + 0.01f, 4.0f) + max(0.01f, min(3000.0f, 0.3f / (0.00001f + pow(abs(10*depth-50) / 200.0f, 4.0f))));

    //dz = ((znear*zfar/z -zfar)/(znear-zfar)
    // znear = 8, zfar = 12

    float dz = ((100.0f/(-12.0f + depth)) + 0.0f)/4.0f;
    //float dz = ((1.0f/(depth)) - 1.0f)/2.0f;
    float dz1 = 1.0f - dz;
    float weight = alpha*max(0.01f, 3000.0f*dz1*dz1*dz1);

    // RGBA32F texture (accumulation)
    gl_FragData[0] = vec4(vertexColor.rgb * alpha * weight, alpha); // GL_COLOR_ATTACHMENT0, synonym of gl_FragColor

    // R32F texture (revealage)
    // Make sure to use the red channel (and GL_RED target in your texture)
    gl_FragData[1].r = alpha * weight; // GL_COLOR_ATTACHMENT1
}
