//!HOOK MAIN
//!BIND HOOKED
//!DESC Enhanced local contrast (two scales, fixed 2D kernels, halo-aware)

// Based on st0.glsl / stbw.glsl (local contrast) with these fixes and
// improvements:
//   * the original 5x5 kernel array was indexed 0..48 (out of bounds) and the
//     `HOOKED_texOff((2 * x, 2 * y))` comma expression silently discarded the
//     x offset (sampling only a 1D diagonal); here Gaussian weights are
//     computed analytically and offsets are proper vec2s, so the blur is a
//     real 2D neighborhood.
//   * two scales: a 3x3 fine blur (micro-contrast) and a 13x13 wide blur
//     (local adaptation), both midtone-masked and soft-limited to avoid
//     halos / clipping.
//   * the per-pass `strength` uniform (0.0..1.0) scales the whole effect.

float srgb_to_linear(float c) {
  return (c <= 0.04045f) ? (c / 12.92f) : pow((c + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float x) {
  return (x >= 0.0031308) ? (1.055 * pow(x, 1.0 / 2.4) - 0.055) : (12.92 * x);
}

float luma_of(vec4 c) {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

// Gaussian-weighted local mean of luma over a (2*radius+1)^2 neighborhood.
// `spacing` is the tap distance in texels; `sigma` is in tap units.
float local_mean(int radius, float sigma, vec2 spacing) {
  float s2 = 1.0 / (2.0 * sigma * sigma);
  float sum = 0.0;
  float wsum = 0.0;
  for (int y = -radius; y <= radius; y++) {
    for (int x = -radius; x <= radius; x++) {
      float w = exp(-(float(x * x + y * y)) * s2);
      sum += luma_of(HOOKED_texOff(vec2(float(x), float(y)) * spacing)) * w;
      wsum += w;
    }
  }
  return sum / wsum;
}

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);

    float lum = luma_of(c);

    // Wide local mean (13x13 texels): the "adaptation" level that shadows and
    // highlights should follow; fine local mean (3x3): micro detail.
    float b_wide = local_mean(3, 1.2, vec2(2.0));
    float b_fine = local_mean(1, 0.8, vec2(1.0));

    float midtone = clamp(4.0 * lum * (1.0 - lum), 0.0, 1.0); // 1 in midtones
    float darks   = 1.0 - smoothstep(0.0, 0.65, lum);

    float detail = lum - b_fine; // fine detail (micro contrast)
    float adapt  = lum - b_wide; // deviation from the local average

    // 1) Micro-contrast: subtle sharpening of fine detail, midtone-masked.
    float micro = detail * 0.50 * strength * midtone;

    // 2) Local contrast: pull tones toward/away from their local mean.
    //    `edge` damps the boost near strong local deviations (halo control).
    float edge = smoothstep(0.08, 0.40, abs(adapt));
    float local_amp = (0.55 + 0.45 * midtone) * (1.0 - 0.85 * edge);
    float local = adapt * 0.65 * strength * local_amp;

    // 3) Shadow lift: recover a little detail in dark areas.
    float lift = 0.018 * strength * darks * (1.0 - 0.5 * clamp(lum * 5.0, 0.0, 1.0));

    float corr = micro + local + lift;

    // --- color processing in Oklab (perceptually uniform), as in st0 ---
    float r = srgb_to_linear(c.r);
    float g = srgb_to_linear(c.g);
    float b = srgb_to_linear(c.b);

    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

    float l0 = sign(l) * pow(abs(l), 1.0 / 3.0);
    float m0 = sign(m) * pow(abs(m), 1.0 / 3.0);
    float s0 = sign(s) * pow(abs(s), 1.0 / 3.0);

    float lum_ok = 0.2104542553f * l0 + 0.7936177850f * m0 - 0.0040720468f * s0;
    float coa = 1.9779984951f * l0 - 2.4285922050f * m0 + 0.4505937099f * s0;
    float cob = 0.0259040371f * l0 + 0.7827717662f * m0 - 0.8086757660f * s0;

    float new_lum = clamp(lum_ok + corr, 0.0, 1.0);
    // Very mild chroma boost in the midtones (keeps the look natural).
    float sat = 1.0 + 0.05 * strength * midtone;
    float new_coa = coa * sat;
    float new_cob = cob * sat;

    float l1 = new_lum + 0.3963377774f * new_coa + 0.2158037573f * new_cob;
    float m1 = new_lum - 0.1055613458f * new_coa - 0.0638541728f * new_cob;
    float s1 = new_lum - 0.0894841775f * new_coa - 1.2914855480f * new_cob;

    float l2 = l1 * l1 * l1;
    float m2 = m1 * m1 * m1;
    float s2 = s1 * s1 * s1;

    c.r = linear_to_srgb(clamp(4.0767416621f * l2 - 3.3077115913f * m2 + 0.2309699292f * s2, 0.0, 1.0));
    c.g = linear_to_srgb(clamp(-1.2684380046f * l2 + 2.6097574011f * m2 - 0.3413193965f * s2, 0.0, 1.0));
    c.b = linear_to_srgb(clamp(-0.0041960863f * l2 - 0.7034186147f * m2 + 1.7076147010f * s2, 0.0, 1.0));

    return c;
}
