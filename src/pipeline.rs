//! Pipeline file parsing.
//!
//! A pipeline file contains one shader per line:
//!
//! ```text
//! path/to/shader.glsl 0.5
//! ```
//!
//! The strength is optional (defaults to 1.0) and must be in 0.0..=1.0.
//! Relative shader paths are resolved against the pipeline file's directory.
//! Blank lines and lines starting with `#` are ignored.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// One shader pass from the pipeline file.
pub struct PassSpec {
    pub shader_path: PathBuf,
    pub strength: f32,
}

pub fn parse_pipeline(path: &Path) -> Result<Vec<PassSpec>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read pipeline file `{}`", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut passes = Vec::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let shader = parts
            .next()
            .with_context(|| format!("pipeline {}:{}: empty line", path.display(), lineno + 1))?;
        let strength = match parts.next() {
            Some(value) => value.parse::<f32>().with_context(|| {
                format!(
                    "pipeline {}:{}: invalid strength `{value}` (expected 0.0..=1.0)",
                    path.display(),
                    lineno + 1
                )
            })?,
            None => 1.0,
        };
        if parts.next().is_some() {
            bail!(
                "pipeline {}:{}: too many columns (expected `shader_path [strength]`)",
                path.display(),
                lineno + 1
            );
        }
        if !(0.0..=1.0).contains(&strength) {
            bail!(
                "pipeline {}:{}: strength {strength} out of range 0.0..=1.0",
                path.display(),
                lineno + 1
            );
        }

        let shader_path = PathBuf::from(shader);
        let shader_path = if shader_path.is_absolute() {
            shader_path
        } else {
            base.join(shader_path)
        };
        if !shader_path.is_file() {
            bail!(
                "pipeline {}:{}: shader file not found: `{}`",
                path.display(),
                lineno + 1,
                shader_path.display()
            );
        }
        passes.push(PassSpec { shader_path, strength });
    }

    if passes.is_empty() {
        bail!("pipeline file `{}` contains no shader entries", path.display());
    }
    Ok(passes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_relative_and_absolute() {
        let dir = std::env::temp_dir().join("video-shader-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shader = dir.join("s.glsl");
        std::fs::write(&shader, "//!HOOK MAIN\nvec4 hook(){return vec4(0.);}\n").unwrap();
        let pipeline = dir.join("p.pipeline");
        std::fs::write(
            &pipeline,
            format!("s.glsl 0.5\n{}\n", shader.display()),
        )
        .unwrap();
        let passes = parse_pipeline(&pipeline).unwrap();
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].strength, 0.5);
        assert!(passes[0].shader_path.is_absolute());
        assert!(passes[1].shader_path.is_absolute());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parse_rejects_bad_strength() {
        let dir = std::env::temp_dir().join("video-shader-test2");
        std::fs::create_dir_all(&dir).unwrap();
        let shader = dir.join("s.glsl");
        std::fs::write(&shader, "x").unwrap();
        let pipeline = dir.join("p.pipeline");
        std::fs::write(&pipeline, "s.glsl 2.0\n").unwrap();
        assert!(parse_pipeline(&pipeline).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
