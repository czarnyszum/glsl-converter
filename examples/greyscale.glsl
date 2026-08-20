//!HOOK MAIN
//!BIND HOOKED
//!DESC Convert to greyscale (simple demo)

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);
    float luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    return vec4(luma, luma, luma, c.a);
}
