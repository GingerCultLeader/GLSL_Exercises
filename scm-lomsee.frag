#version 120

varying vec3 var_V;
varying vec3 var_L;
varying vec3 var_N;

struct scm
{
    vec2  r;
    vec2  b[16];
    float a[16];
    float k0;
    float k1;
};

uniform sampler2D color_sampler;
uniform sampler2D normal_sampler;

uniform scm color;
uniform scm normal;

uniform vec2 A[16];
uniform vec2 B[16];

//------------------------------------------------------------------------------

vec4 sample_color(vec2 t)
{
    vec4   c = texture2D(color_sampler, (t * A[ 0] + B[ 0]) * color.r + color.b[ 0]);
    c = mix(c, texture2D(color_sampler, (t * A[ 1] + B[ 1]) * color.r + color.b[ 1]), color.a[ 1]);
    c = mix(c, texture2D(color_sampler, (t * A[ 2] + B[ 2]) * color.r + color.b[ 2]), color.a[ 2]);
    c = mix(c, texture2D(color_sampler, (t * A[ 3] + B[ 3]) * color.r + color.b[ 3]), color.a[ 3]);
    c = mix(c, texture2D(color_sampler, (t * A[ 4] + B[ 4]) * color.r + color.b[ 4]), color.a[ 4]);
    c = mix(c, texture2D(color_sampler, (t * A[ 5] + B[ 5]) * color.r + color.b[ 5]), color.a[ 5]);
    c = mix(c, texture2D(color_sampler, (t * A[ 6] + B[ 6]) * color.r + color.b[ 6]), color.a[ 6]);
    c = mix(c, texture2D(color_sampler, (t * A[ 7] + B[ 7]) * color.r + color.b[ 7]), color.a[ 7]);
    c = mix(c, texture2D(color_sampler, (t * A[ 8] + B[ 8]) * color.r + color.b[ 8]), color.a[ 8]);
    c = mix(c, texture2D(color_sampler, (t * A[ 9] + B[ 9]) * color.r + color.b[ 9]), color.a[ 9]);
    c = mix(c, texture2D(color_sampler, (t * A[10] + B[10]) * color.r + color.b[10]), color.a[10]);
    c = mix(c, texture2D(color_sampler, (t * A[11] + B[11]) * color.r + color.b[11]), color.a[11]);
    c = mix(c, texture2D(color_sampler, (t * A[12] + B[12]) * color.r + color.b[12]), color.a[12]);
    c = mix(c, texture2D(color_sampler, (t * A[13] + B[13]) * color.r + color.b[13]), color.a[13]);
    c = mix(c, texture2D(color_sampler, (t * A[14] + B[14]) * color.r + color.b[14]), color.a[14]);
    c = mix(c, texture2D(color_sampler, (t * A[15] + B[15]) * color.r + color.b[15]), color.a[15]);
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

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
{
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    
    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;
    
    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    
    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y
    
    // Permutations
    i = mod289(i);
    vec4 p = permute( permute( permute(
                                       i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                     + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    
    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;
    
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)
    
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
    
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    
    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    
    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    
    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                 dot(p2,x2), dot(p3,x3) ) );
}

float turbulantNoise(vec3 v)
{
    float noise = (abs(snoise(2048.0 * v)) / 2.0 +
                   abs(snoise(4096.0 * v)) / 4.0 +
                   abs(snoise(8192.0 * v)) / 8.0 +
                   abs(snoise(16384.0 * v)) / 16.0);
    
    return noise;
}

//------------------------------------------------------------------------------

vec4 norm(vec4 c, float k0, float k1)
{
    return vec4(mix(vec3(k0), vec3(k1), c.rgb), c.a);
}

void main()
{
    vec3 V = normalize(var_V);
    vec3 L = normalize(var_L);

    vec4 d = norm(sample_color (gl_TexCoord[0].xy),  color.k0,  color.k1);
    vec4 n = norm(sample_normal(gl_TexCoord[0].xy), normal.k0, normal.k1);

    vec3 N = normalize(n.rgb * 2.0 - 1.0);
    
    float noise_0 = turbulantNoise(var_N);
    float noise_1 = turbulantNoise(var_N.yzx);
    float noise_2 = turbulantNoise(var_N.zxy);
    
    N = normalize(N + vec3 (noise_0, noise_1, noise_2) / 8.0);

    vec3 reflection = reflect(-L, N);
    float spec = pow(max(dot(reflection, L), 0.0), 64.0);
    float diff = max(dot(N, L), 0.0);

    //gl_FragColor = vec4(d.rgb * kd, 1.0);
    gl_FragColor = vec4(d.rgb * diff, 1.0);
}

