//! Offline validation of the shader preprocessor against naga — the same GLSL
//! frontend wgpu uses. This is the closest thing to a GPU test available in
//! CI: if the preprocessed shader parses and validates here, it will compile
//! on the user's machine.

use naga::front::glsl::Frontend;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use video_shader::glsl;

fn validate(glsl: &str) -> Result<naga::Module, String> {
    let mut frontend = Frontend::default();
    let module = frontend
        .parse(
            &naga::front::glsl::Options::from(naga::ShaderStage::Fragment),
            glsl,
        )
        .map_err(|e| format!("GLSL parse error: {e}"))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .map_err(|e| format!("validation error: {e}"))?;
    Ok(module)
}

fn preprocess_or_skip(path: &str) -> Option<String> {
    let source = std::fs::read_to_string(path).ok()?;
    let pre = glsl::preprocess(&source, path, 1.0).unwrap_or_else(|e| panic!("{path}: {e}"));
    Some(pre.glsl)
}

/// Verify the bind group layout the preprocessor generates matches what the
/// runtime pipeline layout expects: uniform at binding 0, then texture/sampler
/// pairs at 1+2k / 2+2k.
fn check_bindings(module: &naga::Module, texture_count: usize) {
    let mut found: Vec<(u32, String)> = Vec::new();
    for var in module.global_variables.iter() {
        if let Some(binding) = &var.1.binding {
            if binding.group == 0 {
                let kind = match var.1.space {
                    naga::AddressSpace::Uniform => "buffer",
                    naga::AddressSpace::Handle => match module.types[var.1.ty].inner {
                        naga::TypeInner::Sampler { .. } => "sampler",
                        naga::TypeInner::Image { .. } => "texture",
                        _ => "handle",
                    },
                    _ => "other",
                };
                found.push((binding.binding, format!("{kind}:{}", var.1.name.clone().unwrap_or_default())));
            }
        }
    }
    found.sort_by_key(|(b, _)| *b);
    // Anonymous uniform blocks get an empty global name from naga.
    let expected: Vec<(u32, &str)> = vec![(0, "buffer:")]
        .into_iter()
        .chain((0..texture_count).flat_map(|k| {
            vec![
                (1 + 2 * k as u32, "texture:"),
                (2 + 2 * k as u32, "sampler:"),
            ]
        }))
        .collect();
    assert_eq!(found.len(), expected.len(), "bindings: {found:?}");
    for ((fb, fn_), (eb, en)) in found.iter().zip(expected.iter()) {
        assert_eq!(fb, eb, "binding index mismatch: {found:?}");
        assert!(fn_.starts_with(en), "binding {fb}: expected {en}*, got {fn_}");
    }
}

#[test]
fn all_example_shaders_compile() {
    // Every committed example shader must survive preprocessing + naga
    // validation (the closest offline approximation of GPU compilation).
    let mut found_any = false;
    for entry in std::fs::read_dir("examples").expect("examples dir must exist") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("glsl") {
            continue;
        }
        found_any = true;
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let pre = video_shader::glsl::preprocess(&source, &name, 1.0)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let module = validate(&pre.glsl).unwrap_or_else(|e| panic!("{name}: {e}"));
        check_bindings(&module, pre.texture_count);
    }
    assert!(found_any, "no example shaders found");
}

#[test]
fn user_stbw_shader_compiles() {
    // The user's real shader lives in refs/ (gitignored); skip when absent.
    let Some(glsl) = preprocess_or_skip("refs/stbw.glsl") else {
        eprintln!("refs/stbw.glsl not present – skipping");
        return;
    };
    let module = validate(&glsl).unwrap_or_else(|e| panic!("stbw shader: {e}"));
    check_bindings(&module, 1);
}
