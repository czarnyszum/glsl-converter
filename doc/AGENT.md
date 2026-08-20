# AGENT.md — video-shader project guide for coding agents

This file is the entry point for any agent working on this repository. It
explains the architecture, the non-obvious constraints, and the environment
quirks you will hit. **Read this before changing anything.**

## 1. What this project is

`video-shader` is a single-purpose Rust CLI: it takes a video (MP4/WebM; any
container ffmpeg can decode works), applies **mpv-style GLSL user shaders** on
the GPU, and re-encodes the result. It is the deliverable of `PLAN.md` (which
is gitignored, user's file).

The data flow is:

```
input video ──► ffmpeg decode ──► swscale → RGBA8
              ──► wgpu: shader chain (ping-pong textures, fullscreen triangle)
              ──► RGBA8 readback ──► swscale → YUV420P ──► ffmpeg encode ──► output
```

- Output codec is chosen by the **output file extension**: `.mp4` →
  `libx264` (H.264), `.webm` → `libvpx-vp9` (VP9).
- Audio streams are **copied through** when the target container supports the
  codec (AAC/MP3/Opus/ALAC/FLAC → mp4; Opus/Vorbis → webm), otherwise dropped
  with a warning.
- Output is constant frame rate, `yuv420p`, same resolution as the input.

## 2. Build, test, run

```bash
cargo build            # debug
cargo build --release  # release (LTO enabled)
cargo test             # 14 tests, all offline (no GPU needed)
./target/release/video-shader -i in.mp4 -p examples/stbw.pipeline -o out.mp4
./target/release/video-shader -i in.mp4 -p refs/p0.pipeline --dry-run
```

**Environment quirk (this workspace only):** the sandbox makes `~/.cargo`
read-only, so a **workspace-local cargo home** is used:

```bash
export CARGO_HOME=/home/zhxdmu/projects/glsl-converter/.cargo-home
```

`cargo` will not find the registry otherwise (`Read-only file system` errors).
`.cargo-home/` is gitignored. The pre-fetched registry is copied there.

**System requirements** (Debian/Ubuntu): `libavcodec-dev libavformat-dev
libavutil-dev libswscale-dev pkg-config`, plus ffmpeg builds containing
`libx264` and `libvpx-vp9` encoders. Verify with
`ffmpeg -encoders | grep -E 'libx264|libvpx-vp9'`.

## 3. Directory layout

```
Cargo.toml          # deps pinned (see §8 for why)
Cargo.lock          # committed (binary project)
README.md           # user-facing docs
PLAN.md             # original plan; GITIGNORED (user's file)
doc/AGENT.md        # this file
src/
  lib.rs            # re-exports the four modules (lib + bin crate)
  main.rs           # CLI (clap), orchestration, progress bar
  glsl.rs           # mpv-style GLSL preprocessing (pure string rewriting)
  gpu.rs            # wgpu init, pass compilation, per-frame processing
  pipeline.rs       # pipeline-file parsing
  video.rs          # ffmpeg decode/encode, swscale bridges
tests/
  shader_compile.rs # offline naga validation of the preprocessor output
examples/           # committed demo shaders + pipelines
  greyscale.glsl    # simple HOOK MAIN shader
  greyscale.pipeline
  stbw.glsl         # copy of the user's real local-contrast shader
  stbw.pipeline
refs/               # user's REAL shaders/pipelines; GITIGNORED
videos/             # user's REAL sample videos; GITIGNORED
```

`.gitignore` also ignores: `PLAN.md`, `refs/`, `videos/`, `.cargo-home/`,
`.scratch/`, `.tmp-inspect/`. **Never commit anything under those.**

## 4. Module-by-module

### `src/main.rs` — CLI and orchestration

- `Args` (clap derive): `-i/--input`, `-p/--pipeline`, `-o/--output`
  (default `out_<basename>.<ext>`), `--dry-run`, `-b/--bitrate`, `-v/-vv`.
- `main()` order of operations (this order matters):
  1. parse pipeline file + preprocess every shader (CPU-only, fast errors);
  2. `gpu::init()` + compile every pass (`Gpu::compile_pass`) — logs per-shader
     success;
  3. `video::probe()` — analyse the input (also used by `--dry-run`);
  4. if `--dry-run`: print GPU + codec summary, exit;
  5. `video::VideoProcessor::new(probe, output, bitrate)`;
  6. `gpu::FrameProcessor::new(gpu, w, h, passes)`;
  7. `VideoProcessor::run(closure)` where the closure calls
     `FrameProcessor::process_frame` and drives the indicatif progress bar.
- ffmpeg's C logging is set to `Level::Info` (keeps driver/library noise low).

### `src/glsl.rs` — mpv user-shader preprocessor

Turns an mpv shader into a **self-contained GLSL 450 fragment shader** that
naga (wgpu's GLSL frontend) accepts. Pure string rewriting; no GPU involved.

Public API:
- `parse_meta(&str) -> Result<ShaderMeta>` — reads `//!` tags.
- `preprocess(source, file_name, strength) -> Result<Preprocessed>` where
  `Preprocessed { glsl, texture_count, desc }`.

What it does, in order:
1. Parses `//!HOOK MAIN` (other hook targets → error; only whole-frame `MAIN`
   is supported), `//!BIND <name>` (all bound textures are aliases of the
   input frame), `//!DESC`. Sizing tags (`WIDTH`, `SCALED`, …) are ignored with
   a debug log; `//!TEXTURE`/`//!LUT` → error (external textures unsupported).
2. Strips `//!` lines; replaces `HOOKED` → `MAIN`.
3. Expands macros (balanced-paren aware): `N_texOff(off)` →
   `texture(sampler2D(N_tex, N_sampler), N_pos + (off) * N_pt)` and
   `N_tex(p)` → `texture(sampler2D(N_tex, N_sampler), p)`.
   Also rewrites bare `texture(N_tex, ...)` calls.
4. **Comma-operator rewrite**: naga's GLSL frontend cannot parse `(a, b)`
   expressions (your real shader uses `HOOKED_texOff((2*x, 2*y))`). The
   rewriter keeps only the last operand of comma operators at group level while
   preserving function-call argument commas, and strips comments. See
   `rewrite_comma_operators` (unit-tested).
5. Wraps everything: header + `void main()` that sets `N_texSize`/
   `N_pt`/`N_pos` (from the `gl_FragCoord` builtin)/`N_mul`/`N_offset`, then
   calls the user's `hook()` → `fragColor`.

**Critical naga-25 constraints baked into the header generation:**
- The combined `uniform sampler2D` declaration type is **not supported** by
  naga's glsl frontend. The header emits **separate** bindings:
  `layout(set=0, binding=1+2k) uniform texture2D N_tex;` and
  `layout(set=0, binding=2+2k) uniform sampler N_sampler;` and sampling goes
  through the `sampler2D(N_tex, N_sampler)` constructor macro.
- No user vertex inputs: `N_pos` is derived from `gl_FragCoord` (a native naga
  builtin), which avoids stage-interface interpolation mismatches.
- Anonymous uniform block (`uniform Params { float strength; float frame; };`
  without an instance name) is supported by naga. `frame` is the running frame
  index written by the host each frame (`FrameProcessor::process_frame` takes a
  `frame_index` argument and refreshes the params buffer per pass per frame) so
  shaders can do temporal effects; `strength` is the pipeline value.

### `src/gpu.rs` — wgpu

Public API:
- `VERTEX_SHADER` — the WGSL fullscreen-triangle vertex shader
  (outputs `@builtin(position)` + `@location(0) uv`).
- `struct Gpu { device, queue, adapter_info, limits }`,
  `async fn init() -> Result<Gpu>` — adapter/device creation, logs adapter
  name/backend/driver + texture limits.
- `validate_glsl(&str) -> Result<()>` — parses + validates a GLSL fragment
  with **naga directly** (`wgpu::naga`) so users get readable shader errors
  before pipeline creation.
- `Gpu::compile_pass(glsl, texture_count, strength, desc) -> Result<PassPipeline>`
  — shader module (via `ShaderSource::Glsl`), bind-group layout, render
  pipeline, and a 16-byte uniform buffer holding `strength`.
- `struct PassPipeline { desc, strength, texture_count, pipeline, params }`
  — no bind groups yet (textures don't exist until frame size is known).
- `FrameProcessor::new(gpu, w, h, passes) -> Result<Self>` — creates the two
  ping-pong RGBA8 textures, the sampler, the readback staging buffer, and the
  per-pass bind groups (`bind_group_a`/`bind_group_b` bind the input texture,
  either A or B). Validates size against `max_texture_dimension_2d`.
- `FrameProcessor::process_frame(&mut self, rgba: &[u8], stride, frame_index)
  -> Result<Vec<u8>>` — refreshes the per-pass Params uniform
  ({strength, frame}), upload (`queue.write_texture`, arbitrary stride OK) →
  run passes ping-pong
  → `copy_texture_to_buffer` into staging (rows padded to 256 for
  `COPY_BYTES_PER_ROW_ALIGNMENT`) → `map_async` + `poll(PollType::Wait)` →
  copy out tightly packed RGBA8.

**Bind-group layout contract** (must match the GLSL header, verified by
`tests/shader_compile.rs`):
binding 0 = uniform buffer (`Params { float strength; float frame; }`, 16-byte
buffer, `min_binding_size` must cover the block — set to 16), then per texture
k: binding `1+2k` = `texture2D` view, binding `2+2k` = `sampler`.

**Hard-won constraint — one submit per frame:** separate `queue.submit` calls
for the render pass and the readback copy crash the ancient lavapipe driver
(Mesa 20.3, see §7). The render passes and the final copy live in **one
command encoder** with a single submit per frame. Do not split them.

### `src/video.rs` — ffmpeg

Public API:
- `struct VideoInfo` — width/height/fps/container/codec/pixel_format/
  estimated_frames/has_audio/input_path.
- `struct Probe` — an opened input: `ictx`, decoder, RGBA scaler, RGBA frame.
- `fn probe(input) -> Result<Probe>` — opens input, best video stream, decoder
  context, logs container/codec/resolution/fps/frames. Used by dry-run and the
  full run. (Codec name is looked up by id via `decoder::find`, not
  `Context::codec()` — the latter is only set after the context is opened.)
- `struct VideoProcessor`, `fn new(probe, output, bitrate) -> Result<Self>` —
  creates the output muxer (`format::output`), picks encoder settings by
  extension, sets up audio passthrough mappings (with a per-container codec
  allowlist), writes the header, resolves output stream time bases.
- `fn run(self, process: F) -> Result<ProcessStats>` where
  `F: FnMut(&[u8], usize) -> Result<Vec<u8>>` — takes **self by value**,
  destructures into locals (needed to iterate `ictx.packets()` while mutating
  the encoder/muxer). Decodes packets, hands each RGBA frame to the closure,
  converts the processed RGBA back to YUV420P, encodes, flushes decoder,
  encoder, and muxer. `ProcessStats { frames, audio_packets }`.

**Two ffmpeg-4.3 API gotchas you must not regress:**
1. The **encoder context must be created with `avcodec_alloc_context3(codec)`**
   (`encoder_context()` helper), NOT `Context::from_parameters(...)`. The
   latter uses `avcodec_alloc_context3(NULL)` → generic ffmpeg defaults
   (gop=12, qmin=2, qmax=31, …) which libx264's wrapper maps into x264 params
   and then rejects as "broken ffmpeg default settings detected". With the
   codec passed, libx264's own defaults (sentinel -1) apply and the mapping is
   skipped. `encoder.open_as_with(codec, options)` (explicit codec) is also
   required — `open_with` (NULL codec) fails on ffmpeg 4.3 with "No codec
   provided to avcodec_open2()".
2. **Allocate a fresh YUV frame per encode** (`frame::Video::new(...)` inside
   `drain_decoded_frames`). `avcodec_send_frame` hands the encoder references
   to the frame's buffers, and threaded encoders (x264/vp9) may still read them
   asynchronously; reusing one frame races the encoder.

Other details: decoder flush via `send_eof()`, packet timing via
`rescale_ts`, `drain_encoded_packets` rescales from `encoder_time_base` to the
post-header output stream time base, `packet.write_interleaved` for muxing.

### `src/pipeline.rs` — pipeline file parsing

`parse_pipeline(&Path) -> Result<Vec<PassSpec>>`; `PassSpec { shader_path,
strength }`. One `shader_path [strength]` per line; strength optional (1.0),
must be 0.0..=1.0; relative paths resolve against the pipeline file's
directory; `#` comments and blank lines ignored; missing shader files error.

### `tests/shader_compile.rs` — the only automated tests

Offline (no GPU): runs the **real preprocessor** on the examples and on the
user's `refs/stbw.glsl` (skipped if absent), parses/validates the output with
naga 25, and asserts the bind-group layout (buffer@0, texture@1+2k,
sampler@2+2k). This is the guardrail for any change to `glsl.rs`/`gpu.rs`
binding contracts.

## 5. Shader/pipeline semantics

- Every pass runs at **native resolution**; `N_texSize`/`N_pt` etc. are
  derived from `textureSize()`, so no uniforms are needed for them.
- `strength` (0.0–1.0 per pipeline line) is the only per-pass uniform.
- Shader authors get `MAIN` (or `HOOKED`), any `//!BIND` names — all bound to
  the same input texture, and `gl_FragCoord` in pixel coordinates (top-left
  origin).
- Unsupported shader constructs produce **errors at startup** (never silently
  wrong output): non-`MAIN` hooks, `TEXTURE`/`LUT` bindings, missing
  `hook()`, or a user-defined `main()`.

## 6. Constraints and invariants checklist

Before changing anything, make sure you keep all of these:

1. `glsl.rs` header bindings must match `gpu.rs::layout_entries` (the test
   `check_bindings` enforces it).
2. One `queue.submit` per frame in `process_frame` (lavapipe 4.3 bug).
3. Encoder context via `encoder_context()` + `open_as_with` (ffmpeg 4.3).
4. Fresh YUV frame per encode (encoder race).
5. Readback `bytes_per_row` must be a multiple of 256
   (`COPY_BYTES_PER_ROW_ALIGNMENT`); upload rows may be arbitrary (wgpu only
   requires 256-alignment for buffer-side copies).
6. `VideoProcessor::run` consumes `self` (borrow structure).
7. The Params uniform block is `{ float strength; float frame; }`; the bind
   group layout's `min_binding_size` must stay ≥ the block size (16).
7. ffmpeg crates are pinned to 5.1 and wgpu/naga to 25 — see §8.
8. ffmpeg's log level is Info; wgpu's GLSL path needs `wgpu` "glsl" feature.

## 7. Environment / runtime quirks

- **This sandbox has no usable hardware GPU.** wgpu falls back to **lavapipe**
  (Mesa 20.3.5 software Vulkan). Short runs (<~100 frames) work; the driver
  **crashes randomly under sustained load** (verified with a minimal standalone
  wgpu program — a driver bug, not this code). Do not be alarmed by SIGSEGVs
  in `libvulkan_lvp.so` during long test runs here; on the user's real GPU
  (Iris Xe / MX450) this does not occur.
- EGL/GL backend is unusable here (`eglInitialize` fails), so GL fallback
  cannot be tested locally.
- `--dry-run` is fully exercisable locally: GPU init + shader compile + input
  analysis, no rendering.

## 8. Dependency version pins — why these exact versions

- **`ffmpeg-next`/`ffmpeg-sys-next` 5.1.x**: the system ffmpeg is **4.3.8**.
  ffmpeg-sys-next 5.1.x auto-detects and supports FFmpeg 4.3 (newer
  ffmpeg-next 6+/7+ require newer ffmpeg). Uses `default-features = false`
  with only `codec`, `format`, `software-scaling` (avdevice/avfilter dev
  packages are absent and unneeded). `ffmpeg-sys-next` is a direct dependency
  for the `avcodec_alloc_context3(codec)` encoder-context workaround.
- **`wgpu` 25 + `naga` 25** (with `glsl` feature): a stable API baseline the
  code is written against (`PollType::Wait`, `TexelCopyBufferLayout` with
  `Option<u32>` rows, `ShaderSource::Glsl`, `wgpu::naga` re-export). `naga` is
  also a dev-dependency for the offline tests; versions must stay in lockstep
  with wgpu's internal naga.

## 9. Common tasks

- **Add a new supported shader construct** (e.g. `textureLod`, a new meta
  tag): change `glsl.rs`, extend `tests/shader_compile.rs` fixtures, run
  `cargo test`, then a real `--dry-run` + short encode locally.
- **Change the bind-group layout** (e.g. add a uniform): update
  `glsl.rs` header generation, `gpu.rs::layout_entries` +
  `make_bind_group`, and `tests/shader_compile.rs::check_bindings` together.
- **Add an output format**: extend `encoder_settings()` in `video.rs`
  (codec name, options, audio allowlist) — the muxer is auto-guessed from the
  extension.
- **Debug a crash**: check whether it reproduces with the minimal standalone
  wgpu loop pattern (§7) before suspecting this codebase.
