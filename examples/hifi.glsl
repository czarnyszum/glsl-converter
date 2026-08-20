//!HOOK MAIN
//!BIND HOOKED
//!DESC Hifi: animated film grain + temporal dithering (kills 8-bit banding)

// A "hifi feel" grain pass:
//   * film-style grain: three octaves of value noise, decorrelated every
//     frame via the `frame` uniform, stronger in the shadows (like real film
//     grain) and zero-mean so it does not shift brightness;
//   * ordered dithering (4x4 Bayer, rotated each frame) applied right before
//     the 8-bit output quantization, which hides gradient banding.
// Both are scaled by the pipeline `strength` (0.0..1.0); dithering stays on
// even at strength 0 so gradients never band.

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);
    vec2 px = HOOKED_pos * HOOKED_texSize; // pixel coordinates

    // --- animated film grain -----------------------------------------------
    // Each octave's noise field is shifted by a different per-frame amount so
    // consecutive frames are decorrelated (true grain motion, no shimmering).
    float g = 0.60 * vnoise(px * 1.7 + vec2(frame * 13.7, frame * 7.1))
            + 0.30 * vnoise(px * 3.4 + vec2(frame * 29.9, frame * 17.3))
            + 0.10 * vnoise(px * 6.8 + vec2(frame * 47.3, frame * 31.7));
    g = g - 0.5; // zero mean

    float lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    float grain_mask = 0.30 + 0.70 * (1.0 - lum); // stronger in shadows
    float grain = g * (0.05 * strength) * grain_mask; // ~2.5% at strength 1

    // --- ordered dithering (4x4 Bayer), rotated per frame -------------------
    // Standard 4x4 Bayer matrix, column-major.
    mat4 bayer = mat4(
        0.0, 12.0, 3.0, 15.0,
        8.0, 4.0, 11.0, 7.0,
        2.0, 14.0, 1.0, 13.0,
        10.0, 6.0, 9.0, 5.0
    ) * 0.0625;
    vec2 dpos = px + vec2(1.7, 3.1) * frame; // rotate the pattern each frame
    int ix = int(mod(dpos.x, 4.0));
    int iy = int(mod(dpos.y, 4.0));
    float d = bayer[ix][iy] - 0.5; // [-0.5, 0.5] LSB
    float dither = d * (1.0 / 255.0) * (0.5 + 0.5 * strength);

    c.rgb = c.rgb + vec3(grain) + vec3(dither);
    return c;
}
