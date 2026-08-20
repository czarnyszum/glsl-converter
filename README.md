# video-shader

Apply **mpv-style GLSL shaders** to video files on the **GPU**, and re-encode the
result. Supports **MP4** (H.264/H.265 input, H.264 output) and **WebM** (VP8/VP9
input, VP9 output) containers — and any other container ffmpeg can decode.

Shaders use the familiar [mpv user shader format](https://mpv.io/manual/stable/#gpu-shaders):
`//!HOOK MAIN`, `//!BIND HOOKED`, a `hook()` function, and the `HOOKED_tex`,
`HOOKED_texOff`, `HOOKED_pos`, `HOOKED_pt` … identifiers/macros. A pipeline file
lists the shaders to apply in order, each with a strength 0.0–1.0.

```
MP4/WebM ──► ffmpeg decode ──► RGBA frames ──► GPU (wgpu) shader chain ──► RGBA ──► ffmpeg encode ──► MP4/WebM
```

## Requirements

* A GPU with a Vulkan (or OpenGL/GLES, Metal, DX12) driver — the adapter name
  and backend are logged at startup.
* System ffmpeg development libraries, built with the encoders you need:
  * `libx264` for `.mp4` output,
  * `libvpx-vp9` for `.webm` output.

  Debian/Ubuntu example:
  ```bash
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev pkg-config
  ```
  Verify the encoders are present:
  ```bash
  ffmpeg -encoders | grep -E 'libx264|libvpx-vp9'
  ```

## Build

```bash
cargo build --release
# binary: target/release/video-shader
```

## Usage

```
video-shader -i <input> -p <pipeline> [-o <output>] [--dry-run] [-b <bitrate>] [-v]
```

| Option | Meaning |
| --- | --- |
| `-i, --input` | Input video (MP4/WebM; anything ffmpeg decodes works). |
| `-p, --pipeline` | Pipeline file: one `shader_filename strength` per line. |
| `-o, --output` | Output path. **The extension picks the codec**: `.mp4` → H.264 (`libx264`), `.webm` → VP9 (`libvpx-vp9`). Defaults to `out_<basename>.<ext>` beside the input. |
| `--dry-run` | Initialise the GPU, compile all shaders, analyse the input and print codec info — no processing, no output file. |
| `-b, --bitrate` | Encoder bitrate in bits/s. Default is CRF quality mode (`crf=18` for H.264, `crf=32` for VP9). |
| `-v` | More logging (`-vv` for trace). |

### Pipeline file

One shader per line: `path/to/shader.glsl <strength>`. `strength` is optional
(defaults to 1.0) and must be in **0.0–1.0**. Relative paths are resolved
against the pipeline file's directory. `#` starts a comment; blank lines are
ignored. Shaders are applied in order (each pass feeds the next).

### Shader format

mpv-style GLSL user shaders, preprocessed for this tool:

* `//!HOOK MAIN` — required; the shader processes the whole frame.
* `//!BIND <name>` — extra textures, all bound to the input frame.
* `//!DESC <text>` — description, shown in the logs.
* `hook()` — the function that returns the processed `vec4` colour.
* Macros: `N_tex(p)`, `N_texOff(off)`, `N_pos`, `N_pt`, `N_texSize`, `N_mul`,
  `N_offset` (for `N` = `MAIN` and every `BIND`); `HOOKED` is an alias for the
  main texture.
* `gl_FragCoord` is approximated as pixel coordinates (top-left origin).
* `strength` (pipeline value) and `frame` (running frame index, for temporal
  effects like animated film grain or temporal dithering) are available as
  globals in `hook()`.
* Custom `//!TEXTURE`/`//!LUT` bindings are **not** supported.

`examples/` contains:
* `greyscale.glsl` — the simplest possible pass-through demo;
* `stbw.glsl` / `st0-enhanced.glsl` — local-contrast shaders. `st0-enhanced`
  fixes two bugs present in the originals (a kernel array indexed out of
  bounds, and a comma expression that silently dropped the x offset) and adds
  two-scale, halo-aware, `strength`-scaled local contrast in Oklab space;
* `hifi.glsl` — animated film grain (zero-mean, shadow-weighted, ~2.5% at
  strength 1) plus 4×4 Bayer temporal dithering to eliminate 8-bit gradient
  banding. Dithering stays active even at strength 0.

## Examples

```bash
# MP4 → MP4
./video-shader -i videos/sample.mp4 -p examples/stbw.pipeline -o out.mp4

# WebM → WebM
./video-shader -i videos/sample.webm -p examples/greyscale.pipeline -o out.webm

# MP4 → WebM (transcode + shaders)
./video-shader -i videos/sample.mp4 -p examples/stbw.pipeline -o out.webm

# two passes, different strengths
printf 'examples/stbw.glsl 0.5\nexamples/greyscale.glsl 0.8\n' > two.pipeline
./video-shader -i videos/sample.mp4 -p two.pipeline -o out.mp4

# check everything without processing
./video-shader -i videos/sample.mp4 -p examples/stbw.pipeline --dry-run
```

## What gets logged

At startup the tool logs (and `--dry-run` prints):

* the GPU adapter name, backend and device type — **verify your GPU is seen**;
* detected container, codec, resolution, pixel format and frame rate;
* per-shader description, strength and compilation result;
* progress per frame with fps and ETA;
* a final summary (frames processed, wall time, average fps, audio packets copied).

## Limitations & notes

* **Audio**: audio streams are copied through when the target container
  supports the codec (e.g. AAC in `.mp4`, Opus/Vorbis in `.webm`), otherwise
  they are dropped with a warning.
* Output is constant frame rate, `yuv420p`, same resolution as the input.
* Sizing/scale meta tags (`WIDTH`, `OFFSET`, `SCALED`, …) are ignored — every
  pass runs at native resolution.
* If your GPU driver is missing, you will get a clear "no GPU adapter
  available" error from `--dry-run`.
* Shaders that declare `//!TEXTURE`/`//!LUT` bindings or use unsupported GLSL
  constructs are rejected at startup with a readable error — nothing is
  silently skipped.
* **Software renderers**: the tool works under lavapipe/llvmpipe (useful for
  quick checks), but the ancient Mesa versions shipped with older distros can
  crash under sustained load; use a real GPU driver for actual processing.
