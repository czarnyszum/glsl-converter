

//!HOOK MAIN
//!BIND HOOKED
//!DESC Local Constrast

float srgb_to_linear(float c) {
  return (c <= 0.04045f) ? (c / 12.92f) : pow((c + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float x) {
  if (x >= 0.0031308) {
    return 1.055 * pow(x, 1.0/2.4) - 0.055;
  } else {
    return 12.92 * x;
  }
}



vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);

    float ker0[25] = {
    1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0
  };

    float b0 = 0.0;
    float b2 = 0.0;

    int ix = 0;
    for(int y=-3; y <= 3; y++) {
    	    for(int x=-3; x <= 3; x++) {
	    	    vec4 l = HOOKED_texOff((2 * x, 2 * y));
		    float lum0 = 0.2126 * l.r + 0.7152 * l.g + 0.0722 * l.b;
		    float lum2 = lum0 * lum0;
		    
		    b0 += lum0 * ker0[ix];
		    b2 += lum2 * ker0[ix];

		    ix = ix + 1;
    	    }	    
    }

    float luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    float lights = luma * luma * (3.0 - 2.0 * luma);
    float darks = 1.0 - lights;

    float hps = luma - 0.999 * b0;
    float sharpen = 1.0 - exp(-0.66 * hps); 
    float sharpen_amp = 0.1 * lights; // -0.15 * darks;  //0.2 * lights; // highlish // flatten

    float me = -0.5 * (b2 - b0 * b0);
    float mec = exp(me);
    float me_amp = 0.005 * darks; // -0.8 * lights; // 0.25

    float m20 = -0.025 * (b2 - b0 * b0);
    float m20e = exp(m20);
    float m20_amp = 0.05;

    float offset = -0.1; 
    float li = (0.05 * darks * darks - 0.01 * lights * lights);
    float lum_amp = 0.99;

    float lum_change = mec * me_amp + m20e * m20_amp + sharpen_amp * sharpen + offset + li;
 
    float r = srgb_to_linear(c.r);
    float g = srgb_to_linear(c.g);
    float b = srgb_to_linear(c.b);

    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

    float l0 = sign(l) * pow(abs(l), 1.0 / 3.0);
    float m0 = sign(m) * pow(abs(m), 1.0 / 3.0);
    float s0 = sign(s) * pow(abs(s), 1.0 / 3.0);

    float lum = 0.2104542553f * l0 + 0.7936177850f * m0 - 0.0040720468f * s0;
    float coa = 1.9779984951f * l0 - 2.4285922050f * m0 + 0.4505937099f * s0;
    float cob = 0.0259040371f * l0 + 0.7827717662f * m0 - 0.8086757660f * s0;

    float new_lum = lum_amp * lum + lum_change;

    float value = linear_to_srgb(new_lum * new_lum * new_lum);

    c.rgb = vec3(value);
    
    return c; 
}
