//! Preprocessing of mpv-style GLSL user shaders.
//!
//! mpv "user shaders" are GLSL fragments that use `//!` meta tags plus special
//! identifiers and macros. This module rewrites them into self-contained GLSL
//! fragment shaders that run under naga (via wgpu's GLSL frontend).
//!
//! Supported meta tags:
//! * `//!HOOK MAIN` – the shader processes the whole frame (the only hook this
//!   tool supports).
//! * `//!BIND <name>` – a texture the shader samples (all bound to the same
//!   input frame here).
//! * `//!DESC <text>` – human readable description (logged).
//!
//! Rewritten macros (for `MAIN` and every `BIND` name `N`):
//! * `N_texOff(off)` → `texture(N_tex, N_pos + (off) * N_pt)`
//! * `N_tex(p)`     → `texture(N_tex, p)`
//! * `N_pos`, `N_pt`, `N_texSize`, `N_mul`, `N_offset` become globals that are
//!   set up in `main()` before the user's `hook()` is called.
//! * `HOOKED` is replaced by `MAIN`.
//!
//! naga's GLSL frontend does not support the GLSL comma operator `(a, b)`, so
//! it is rewritten to the last operand (semantically identical for the pure
//! expressions used in shaders).

use anyhow::{bail, Context, Result};
use std::collections::HashSet;

/// Meta information extracted from a shader.
pub struct ShaderMeta {
    pub desc: String,
    pub binds: Vec<String>,
}

/// Parse the `//!` meta tags of a shader source.
pub fn parse_meta(source: &str) -> Result<ShaderMeta> {
    let mut desc = String::new();
    let mut hook: Option<String> = None;
    let mut binds: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for line in source.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("//!") {
            continue;
        }
        let rest = trimmed.trim_start_matches("//!").trim_start();
        let (tag, args) = match rest.find(char::is_whitespace) {
            Some(idx) => (&rest[..idx], rest[idx..].trim()),
            None => (rest, ""),
        };
        match tag {
            "HOOK" => {
                if hook.is_some() {
                    bail!("shader declares more than one HOOK");
                }
                hook = Some(args.to_string());
            }
            "BIND" => {
                if args.is_empty() {
                    bail!("empty BIND tag");
                }
                if seen.insert(args.to_string()) {
                    binds.push(args.to_string());
                }
            }
            "DESC" => desc = args.to_string(),
            // Tags we intentionally ignore: this tool renders the full frame at
            // its native resolution, so sizing/scale hints are meaningless.
            "WHEN" | "WIDTH" | "HEIGHT" | "OFFSET" | "COMPONENTS" | "NATIVESIZE"
            | "SCALED" | "MIPMAPS" | "CONV" | "WINDOW" | "RAW" => {
                log::debug!("ignoring meta tag {tag} (native-size processing)");
            }
            // Tags we cannot honour at all.
            "TEXTURE" | "LUT" => {
                bail!(
                    "custom `{tag}` bindings are not supported by video-shader \
                     (shader loads an external texture)"
                );
            }
            other => log::warn!("ignoring unknown meta tag `{other}`"),
        }
    }

    let hook = match hook {
        Some(h) => h,
        None => bail!("shader does not declare a hook target (`//!HOOK MAIN`)"),
    };
    let targets: Vec<&str> = hook.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if targets.is_empty() {
        bail!("shader declares an empty HOOK target");
    }
    if !targets.iter().any(|t| *t == "MAIN") {
        bail!(
            "unsupported hook target(s) `{hook}` – only `MAIN` (the full frame) is supported"
        );
    }
    if targets.len() > 1 {
        log::warn!("shader hooks {targets:?}; only the MAIN pass is processed");
    }

    Ok(ShaderMeta { desc, binds })
}

/// A preprocessed shader ready for GPU compilation.
pub struct Preprocessed {
    /// Self-contained GLSL fragment shader.
    pub glsl: String,
    /// Number of `uniform sampler2D` bindings it declares (MAIN + every BIND).
    pub texture_count: usize,
    /// The shader's `//!DESC` description, if any.
    pub desc: String,
}

/// Preprocess a shader source into a self-contained GLSL fragment shader.
pub fn preprocess(source: &str, file_name: &str, strength: f32) -> Result<Preprocessed> {
    let meta = parse_meta(source).with_context(|| format!("in shader {file_name}"))?;
    if !(0.0..=1.0).contains(&strength) {
        bail!("strength {strength} out of range 0.0..=1.0 for shader {file_name}");
    }

    // Strip the `//!` meta lines from the body.
    let mut body = String::with_capacity(source.len());
    for line in source.lines() {
        if line.trim_start().starts_with("//!") {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }

    // `HOOKED` is a placeholder for the main texture in mpv shaders.
    body = body.replace("HOOKED", "MAIN");

    // Texture names: MAIN plus every BIND, deduplicated.
    let mut names: Vec<String> = vec!["MAIN".to_owned()];
    for bind in &meta.binds {
        let name = if bind == "HOOKED" { "MAIN".to_owned() } else { bind.clone() };
        if !names.contains(&name) {
            names.push(name);
        }
    }

    // Expand the sampling macros. Longest patterns first.
    for name in &names {
        body = expand_macro(&body, name, true).with_context(|| format!("in shader {file_name}"))?;
        body = expand_macro(&body, name, false).with_context(|| format!("in shader {file_name}"))?;
    }

    // Rewrite direct `texture(N_tex, ...)` calls the shader author may have
    // written (the texture alone is not sampleable in the generated code).
    for name in &names {
        body = rewrite_bare_texture(&body, name);
    }

    if !body.contains("hook(") {
        bail!("shader {file_name} does not define a `hook()` function");
    }
    if body.contains("void main(") {
        bail!(
            "shader {file_name} defines `main()` itself; mpv-style shaders must define `hook()`"
        );
    }

    // naga's GLSL frontend cannot parse the comma operator.
    body = rewrite_comma_operators(&body);

    let mut out = String::with_capacity(body.len() + 1024);
    out.push_str("#version 450\n");
    out.push_str(&format!("// preprocessed from {file_name}\n"));
    if !meta.desc.is_empty() {
        out.push_str(&format!("// {}\n", meta.desc));
    }
    // `frame` is the running frame index (float), written by the host every
    // frame so shaders can do temporal effects (film grain, temporal dither).
    out.push_str(
        "layout(set = 0, binding = 0) uniform Params { float strength; float frame; };\n",
    );
    // naga's GLSL frontend requires separate texture and sampler objects
    // (the combined `sampler2D` declaration type is not supported), so each
    // texture gets a `texture2D` at binding 1+2k and a `sampler` at 2+2k.
    for (idx, name) in names.iter().enumerate() {
        out.push_str(&format!(
            "layout(set = 0, binding = {}) uniform texture2D {}_tex;\n",
            1 + 2 * idx,
            name
        ));
        out.push_str(&format!(
            "layout(set = 0, binding = {}) uniform sampler {}_sampler;\n",
            2 + 2 * idx,
            name
        ));
    }
    for name in &names {
        out.push_str(&format!("vec2 {}_pos;\n", name));
        out.push_str(&format!("vec2 {}_pt;\n", name));
        out.push_str(&format!("vec2 {}_texSize;\n", name));
        out.push_str(&format!("vec2 {}_mul;\n", name));
        out.push_str(&format!("vec2 {}_offset;\n", name));
    }
    // No user inputs: the shader position comes from the gl_FragCoord builtin.
    out.push_str("layout(location = 0) out vec4 fragColor;\n\n");
    out.push_str(&body);
    out.push('\n');
    out.push_str("void main() {\n");
    for name in &names {
        out.push_str(&format!("    {}_texSize = vec2(textureSize({}_tex, 0));\n", name, name));
        out.push_str(&format!("    {}_pt = vec2(1.0) / {}_texSize;\n", name, name));
        out.push_str(&format!("    {}_pos = gl_FragCoord.xy * {}_pt;\n", name, name));
        out.push_str(&format!("    {}_mul = vec2(1.0);\n", name));
        out.push_str(&format!("    {}_offset = vec2(0.0);\n", name));
    }
    out.push_str("    fragColor = hook();\n");
    out.push_str("}\n");
    Ok(Preprocessed {
        glsl: out,
        texture_count: names.len(),
        desc: meta.desc,
    })
}

/// Expand every `NAME_texOff(` or `NAME_tex(` call in `body`.
fn expand_macro(body: &str, name: &str, tex_off: bool) -> Result<String> {
    let pattern: Vec<char> = if tex_off {
        format!("{name}_texOff(").chars().collect()
    } else {
        format!("{name}_tex(").chars().collect()
    };
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < chars.len() {
        if i + pattern.len() <= chars.len() && chars[i..i + pattern.len()] == pattern[..] {
            let open = i + pattern.len() - 1; // index of '('
            let mut depth = 0usize;
            let mut j = open;
            while j < chars.len() {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if depth != 0 {
                bail!("unbalanced parentheses in `{name}_texOff(` call");
            }
            let args: String = chars[open + 1..j].iter().collect();
            if tex_off {
                out.push_str(&format!(
                    "texture(sampler2D({name}_tex, {name}_sampler), {name}_pos + ({args}) * {name}_pt)"
                ));
            } else {
                out.push_str(&format!(
                    "texture(sampler2D({name}_tex, {name}_sampler), {args})"
                ));
            }
            i = j + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    Ok(out)
}

/// Rewrite `texture(NAME_tex, ...)` → `texture(sampler2D(NAME_tex, NAME_sampler), ...)`
/// so direct calls written by the shader author keep working.
fn rewrite_bare_texture(body: &str, name: &str) -> String {
    let pattern: Vec<char> = format!("texture({name}_tex,").chars().collect();
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < chars.len() {
        if i + pattern.len() <= chars.len() && chars[i..i + pattern.len()] == pattern[..] {
            out.push_str(&format!("texture(sampler2D({name}_tex, {name}_sampler),"));
            i += pattern.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

struct ParenFrame {
    /// `true` when the parens belong to a function call (`foo(...)`), whose
    /// commas separate arguments and must be kept.
    is_call: bool,
    /// Character offset in `out` where the current comma-separated segment
    /// started (used to discard everything before the last comma in groups).
    segment_start: usize,
}

/// Rewrite GLSL comma operators `(A, B)` → `(B)`.
///
/// Commas inside function calls (`foo(a, b)`) are argument separators and are
/// kept. Only commas at the top level of a parenthesised *grouping* are treated
/// as comma operators and the leading operands are dropped. For the pure
/// expressions used in shaders this is semantically identical to the comma
/// operator.
fn rewrite_comma_operators(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut stack: Vec<ParenFrame> = Vec::new();
    let mut last_significant_is_word = false;
    // Set after a comma operator truncation: skip whitespace that starts the
    // kept segment (it followed the discarded operands).
    let mut skip_leading_ws = false;

    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Strip comments (their content must never affect the rewrite).
        if c == '/' && i + 1 < chars.len() {
            if chars[i + 1] == '/' {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            } else if chars[i + 1] == '*' {
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i = (i + 2).min(chars.len());
                continue;
            }
        }
        if skip_leading_ws {
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            skip_leading_ws = false;
        }
        match c {
            '(' => {
                let is_call = last_significant_is_word;
                out.push(c);
                stack.push(ParenFrame { is_call, segment_start: out.len() });
                last_significant_is_word = false;
            }
            ')' => {
                stack.pop();
                out.push(c);
                last_significant_is_word = false;
            }
            ',' => match stack.last_mut() {
                Some(frame) if !frame.is_call => {
                    out.truncate(frame.segment_start);
                    frame.segment_start = out.len();
                    last_significant_is_word = false;
                    skip_leading_ws = true;
                }
                _ => {
                    out.push(c);
                    last_significant_is_word = false;
                }
            },
            other => {
                out.push(c);
                if !other.is_whitespace() {
                    last_significant_is_word = is_word_char(other);
                }
            }
        }
        i += 1;
    }
    out
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_operator_basic() {
        assert_eq!(rewrite_comma_operators("vec4 x = (1.0, 2.0);"), "vec4 x = (2.0);");
    }

    #[test]
    fn comma_operator_keeps_call_args() {
        let src = "vec4 c = texture(MAIN_tex, (a, b) + vec2(1.0, 2.0));";
        assert_eq!(
            rewrite_comma_operators(src),
            "vec4 c = texture(MAIN_tex, (b) + vec2(1.0, 2.0));"
        );
    }

    #[test]
    fn comma_operator_nested() {
        assert_eq!(
            rewrite_comma_operators("float y = ((a, b), c);"),
            "float y = (c);"
        );
        assert_eq!(rewrite_comma_operators("float y = (a, (b, c), d);"), "float y = (d);");
    }

    #[test]
    fn comma_operator_call_with_space() {
        assert_eq!(
            rewrite_comma_operators("foo (a, b);"),
            "foo (a, b);",
            "parens after a spaced identifier are still a call"
        );
    }

    #[test]
    fn comma_operator_comments() {
        assert_eq!(
            rewrite_comma_operators("// (a, b)\nfloat x = (1.0, 2.0); /* (c, d) */"),
            "\nfloat x = (2.0); "
        );
    }

    #[test]
    fn texoff_expansion() {
        let body = "vec4 c = HOOKED_texOff((2 * x, 2 * y));";
        let body = body.replace("HOOKED", "MAIN");
        let out = expand_macro(&body, "MAIN", true).unwrap();
        assert_eq!(
            out,
            "vec4 c = texture(sampler2D(MAIN_tex, MAIN_sampler), MAIN_pos + ((2 * x, 2 * y)) * MAIN_pt);"
        );
    }

    #[test]
    fn preprocess_stbw_shape() {
        let src = "//!HOOK MAIN\n//!BIND HOOKED\n//!DESC Test\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";
        let out = preprocess(src, "test.glsl", 1.0).unwrap();
        assert!(out.glsl.contains("uniform Params"));
        assert!(out.glsl.contains("uniform texture2D MAIN_tex;"));
        assert!(out.glsl.contains("uniform sampler MAIN_sampler;"));
        assert!(out.glsl.contains("fragColor = hook();"));
        assert!(out.glsl.contains("MAIN_texSize = vec2(textureSize(MAIN_tex, 0));"));
        assert_eq!(out.texture_count, 1);
        assert_eq!(out.desc, "Test");
    }

    #[test]
    fn preprocess_rejects_other_hook() {
        let src = "//!HOOK LUMA\nvec4 hook() { return vec4(0.0); }";
        assert!(preprocess(src, "x.glsl", 1.0).is_err());
    }

    #[test]
    fn preprocess_rejects_missing_hook_fn() {
        let src = "//!HOOK MAIN\nvoid foo() {}";
        assert!(preprocess(src, "x.glsl", 1.0).is_err());
    }
}
