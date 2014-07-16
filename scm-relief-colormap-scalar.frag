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

vec3 brickcolor(vec3 var_V)
{
    //Defines the color of the brick and mortar, the size of the brick and the percent that is brick.
    const vec3 BrickColor = vec3 (1.0, 0.3, 0.2);
    const vec3 MortarColor = vec3 (0.85, 0.86, 0.84);
    const vec3 BrickSize = vec3 (50000.0, 30000.0, 30000.0);
    const vec3 BrickPct = vec3 (0.9, 0.9, 0.9);
    vec3 p = (1.0 - BrickPct) / 2.0;
    
    //Variables to store color, location and whether or not to use the brick color or the mortar color.
    vec3 color;
    vec3 position;
    vec3 useBrick;
    vec3 epsilon;
    
    //Location in respect to the uniform vertex position and the size of the brick.
    position = var_V / BrickSize;
    
    //Determines the offset of the bricks.
    if (fract(position.y * 0.5) > 0.5)
        position.x += 0.5;
    
    //Gives the vertical, horizontal, and depth of a brick.
    epsilon = fwidth(position) * 3.0;
    position = fract(position);
    //Determines if the pixel will be brick or mortar.
    //useBrick = step(position, BrickPct);
    //Creates interpolation on the edges for a blended transition from brick to mortar.
    useBrick = smoothstep(p - epsilon, p + epsilon, position) - smoothstep(1.0 - p - epsilon, 1.0 - p + epsilon, position);
    color = mix(MortarColor, BrickColor, useBrick.x * useBrick.y);
    return color;
}

/*vec3 contourlines(float k)
{
    vec3 color;
    float epsilon;
    float drawline;
    float size = 1000.0;
    float position = k / size;
    float p = 0.0;
    
    epsilon = fwidth(position);
    position = fract(position);
    
    drawline = smoothstep(p, p + epsilon, position) - smoothstep(1.0 - p - epsilon, 1.0 - p, position);
    color = mix(vec3 (0.0, 1.0, 0.0), vec3 (1.0, 1.0, 1.0), drawline);
    
    return color;
}*/

vec3 contourlines(float k)
{
    vec3 color;
    float epsilon;
    float drawline;
    float size = 1000.0;
    float position = k / size;
    float p = 0.5;
    
    epsilon = fwidth(position);
    position = fract(position);
    
    drawline = smoothstep(p - epsilon, p, position) - smoothstep(p, p + epsilon, position);
    color = mix(vec3 (1.0, 1.0, 1.0), vec3 (0.0, 1.0, 0.0), drawline);
    
    return color;
}

//--------------------------------------------------------------------------------------------------

//Noise functions.
vec3 mod289(vec3 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x)
{
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

//------------------------------------------------------------------------------

float cloudyNoise(vec3 v)
{
    float noise = (snoise(8192.0 * v) / 2.0 +
                   snoise(16384.0 * v) / 4.0 +
                   snoise(32768.0 * v) / 8.0 +
                   snoise(65536.0 * v) / 16.0);
    
    return noise;
}

//------------------------------------------------------------------------------

void main()
{
    vec3 V = normalize(var_V);  //View
    vec3 L = normalize(var_L);  //Lighting

    vec4  S =           sample_scalar(gl_TexCoord[0].xy);  //Scalar
    vec3  N = normalize(sample_normal(gl_TexCoord[0].xy).rgb * 2.0 - 1.0);  //Normal

    float k = mix(scalar.k0, scalar.k1, S.r); //height
    
    //Reflection variables
    float specularContribution = 0.5;  //Light that is reflected off the surface.
    float diffuseContribution = 1.0 - specularContribution;  //Light that is absorbed by the surface,
                                                             //then released at random angles.

    /* float nl = max(0.0, dot(N, L));
    float nv = max(0.0, dot(N, V));
    float kd = 2.0 * nl / (nl + nv); */
    
    float noise_0 = cloudyNoise(var_N);
    float noise_1 = cloudyNoise(var_N.yzx);
    float noise_2 = cloudyNoise(var_N.zxy);
    
    N = normalize(N + vec3 (noise_0, noise_1, noise_2) / 8.0);
    
    vec3 reflection = reflect(-L, N);
    float spec = pow(max(dot(reflection, L), 0.0), 64.0);
    float diff = max(dot(N, L), 0.0);
    //vec3 lightIntensity = contourlines(k) * vec3(diffuseContribution * diff) + vec3(specularContribution * spec);

    
    /*if (diff < 0.2 && spec < 0.2)
        discard;*/
    
    //Noisy surface
    gl_FragColor = vec4(colormap(k) * diff, 1.0);
    //Plastic look
    //gl_FragColor = vec4(lightIntensity, 1.0);
    //Plain brick texture
    //gl_FragColor = vec4(brickcolor(var_V), 1.0);
    //Relief shaded colormap
    //gl_FragColor = vec4(colormap(k) * kd, 1.0);
}
