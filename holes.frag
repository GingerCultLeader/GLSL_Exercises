#version 120

varying vec3 var_V;
varying vec3 var_L;

struct scm
{
    vec2  r;
    vec2  b[16];
    float a[16];
    float k0;
    float k1;
};

uniform sampler2D scalar_sampler;
uniform sampler2D normal_sampler;

uniform scm scalar;
uniform scm normal;

uniform vec2 A[16];
uniform vec2 B[16];

//------------------------------------------------------------------------------

vec4 sample_scalar(vec2 t)
{
    vec4   c = texture2D(scalar_sampler, (t * A[ 0] + B[ 0]) * scalar.r + scalar.b[ 0]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 1] + B[ 1]) * scalar.r + scalar.b[ 1]), scalar.a[ 1]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 2] + B[ 2]) * scalar.r + scalar.b[ 2]), scalar.a[ 2]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 3] + B[ 3]) * scalar.r + scalar.b[ 3]), scalar.a[ 3]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 4] + B[ 4]) * scalar.r + scalar.b[ 4]), scalar.a[ 4]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 5] + B[ 5]) * scalar.r + scalar.b[ 5]), scalar.a[ 5]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 6] + B[ 6]) * scalar.r + scalar.b[ 6]), scalar.a[ 6]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 7] + B[ 7]) * scalar.r + scalar.b[ 7]), scalar.a[ 7]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 8] + B[ 8]) * scalar.r + scalar.b[ 8]), scalar.a[ 8]);
    c = mix(c, texture2D(scalar_sampler, (t * A[ 9] + B[ 9]) * scalar.r + scalar.b[ 9]), scalar.a[ 9]);
    c = mix(c, texture2D(scalar_sampler, (t * A[10] + B[10]) * scalar.r + scalar.b[10]), scalar.a[10]);
    c = mix(c, texture2D(scalar_sampler, (t * A[11] + B[11]) * scalar.r + scalar.b[11]), scalar.a[11]);
    c = mix(c, texture2D(scalar_sampler, (t * A[12] + B[12]) * scalar.r + scalar.b[12]), scalar.a[12]);
    c = mix(c, texture2D(scalar_sampler, (t * A[13] + B[13]) * scalar.r + scalar.b[13]), scalar.a[13]);
    c = mix(c, texture2D(scalar_sampler, (t * A[14] + B[14]) * scalar.r + scalar.b[14]), scalar.a[14]);
    c = mix(c, texture2D(scalar_sampler, (t * A[15] + B[15]) * scalar.r + scalar.b[15]), scalar.a[15]);
    return c;
}

vec4 sample_normal(vec2 t)
{
    vec4   c = texture2D(normal_sampler, (t * A[ 0] + B[ 0]) * normal.r + normal.b[ 0]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 1] + B[ 1]) * normal.r + normal.b[ 1]), normal.a[ 1]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 2] + B[ 2]) * normal.r + normal.b[ 2]), normal.a[ 2]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 3] + B[ 3]) * normal.r + normal.b[ 3]), normal.a[ 3]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 4] + B[ 4]) * normal.r + normal.b[ 4]), normal.a[ 4]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 5] + B[ 5]) * normal.r + normal.b[ 5]), normal.a[ 5]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 6] + B[ 6]) * normal.r + normal.b[ 6]), normal.a[ 6]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 7] + B[ 7]) * normal.r + normal.b[ 7]), normal.a[ 7]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 8] + B[ 8]) * normal.r + normal.b[ 8]), normal.a[ 8]);
    c = mix(c, texture2D(normal_sampler, (t * A[ 9] + B[ 9]) * normal.r + normal.b[ 9]), normal.a[ 9]);
    c = mix(c, texture2D(normal_sampler, (t * A[10] + B[10]) * normal.r + normal.b[10]), normal.a[10]);
    c = mix(c, texture2D(normal_sampler, (t * A[11] + B[11]) * normal.r + normal.b[11]), normal.a[11]);
    c = mix(c, texture2D(normal_sampler, (t * A[12] + B[12]) * normal.r + normal.b[12]), normal.a[12]);
    c = mix(c, texture2D(normal_sampler, (t * A[13] + B[13]) * normal.r + normal.b[13]), normal.a[13]);
    c = mix(c, texture2D(normal_sampler, (t * A[14] + B[14]) * normal.r + normal.b[14]), normal.a[14]);
    c = mix(c, texture2D(normal_sampler, (t * A[15] + B[15]) * normal.r + normal.b[15]), normal.a[15]);
    return c;
}

//------------------------------------------------------------------------------

float peak(float k, float c)
{
    return max(0.0, 1.0 - abs(k - c) * 5.0);
}

vec3 bound(vec3 c, float k)
{
    c = mix(c, vec3(1.0), step(1.0, k));
    c = mix(c, vec3(0.2), step(k, 0.0));
    return c;
}

vec3 colormap(float k)
{
    return bound(peak(k, 0.0) * vec3(1.0, 0.0, 1.0) +
                 peak(k, 0.2) * vec3(0.0, 0.0, 1.0) +
                 peak(k, 0.4) * vec3(0.0, 1.0, 1.0) +
                 peak(k, 0.6) * vec3(1.0, 1.0, 0.0) +
                 peak(k, 0.8) * vec3(1.0, 0.0, 0.0) +
                 peak(k, 1.0) * vec3(1.0, 1.0, 1.0), k);
}

//------------------------------------------------------------------------------

void main()
{
    vec3 V = normalize(var_V);
    vec3 L = normalize(var_L);

    vec4  S =           sample_scalar(gl_TexCoord[0].xy);
    vec3  N = normalize(sample_normal(gl_TexCoord[0].xy).rgb * 2.0 - 1.0);

    float k = mix(scalar.k0, scalar.k1, S.r);
    
    float specularContribution = 0.5;
    float diffuseContribution = 1.0 - specularContribution;
    vec3 reflection = reflect(-L, N);
    float spec = pow(max(dot(reflection, L), 0.0), 64.0);
    float diff = max(dot(N, L), 0.0);
    vec3 lightIntensity = colormap(k) * vec3(diffuseContribution * diff) + vec3(specularContribution * spec);
    
    if (diff > 0.9)
        discard;

    gl_FragColor = vec4(lightIntensity, 1.0);
}
