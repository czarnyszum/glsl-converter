//! video-shader: apply mpv-style GLSL shaders to MP4/WebM videos on the GPU.

use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use std::path::{Path, PathBuf};
use std::time::Instant;
use video_shader::{glsl, gpu, pipeline, video};

#[derive(Parser)]
#[command(
    name = "video-shader",
    version,
    about = "Apply mpv-style GLSL shaders to MP4/WebM videos on the GPU"
)]
struct Args {
    /// Input video file (MP4 or WebM; anything ffmpeg can decode works)
    #[arg(short, long)]
    input: PathBuf,

    /// Pipeline file: one `shader_filename strength` per line (strength 0.0–1.0)
    #[arg(short, long)]
    pipeline: PathBuf,

    /// Output video file; the extension selects the codec (.mp4 → H.264, .webm → VP9)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Initialise the GPU, compile all shaders and print codec info, then exit
    #[arg(long)]
    dry_run: bool,

    /// Encoder bitrate in bits/s (default: CRF quality mode)
    #[arg(short = 'b', long)]
    bitrate: Option<u64>,

    /// Increase log verbosity (repeat for debug logging)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_logging(args.verbose);

    ffmpeg::init().context("ffmpeg initialisation failed")?;
    // Keep ffmpeg's own C-level logging quiet; video-shader logs are richer.
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Info);

    // ---- 1. Parse the pipeline file and preprocess every shader (CPU only). ----
    let passes = pipeline::parse_pipeline(&args.pipeline)?;
    log::info!("pipeline `{}`: {} shader pass(es)", args.pipeline.display(), passes.len());

    let mut preprocessed = Vec::new();
    for pass in &passes {
        let source = std::fs::read_to_string(&pass.shader_path)
            .with_context(|| format!("cannot read shader `{}`", pass.shader_path.display()))?;
        let file_name = pass
            .shader_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| pass.shader_path.display().to_string());
        let pre = glsl::preprocess(&source, &file_name, pass.strength)
            .with_context(|| format!("failed to preprocess `{file_name}`"))?;
        log::info!(
            "shader `{file_name}`: strength {} – {}",
            pass.strength,
            if pre.desc.is_empty() { "(no description)" } else { &pre.desc }
        );
        preprocessed.push(pre);
    }

    // ---- 2. Initialise the GPU and compile every shader pass. ----
    log::info!("initialising GPU…");
    let gpu = pollster::block_on(gpu::init())?;

    let mut compiled = Vec::new();
    for (i, pre) in preprocessed.iter().enumerate() {
        let pass = gpu.compile_pass(&pre.glsl, pre.texture_count, passes[i].strength, &pre.desc)?;
        log::info!("shader {} compiled OK (strength {})", i + 1, passes[i].strength);
        compiled.push(pass);
    }

    // ---- 3. Analyse the input video. ----
    let probe = video::probe(&args.input.to_string_lossy())?;

    if args.dry_run {
        println!();
        println!("DRY RUN OK – GPU ready, {} shader(s) compiled, input analysed", compiled.len());
        println!("  GPU:     {}", gpu.adapter_info.name);
        println!("  input:   {}", probe.info.input_path);
        println!("  contain: {}", probe.info.container);
        println!("  codec:   {} ({})", probe.info.codec_name, probe.info.codec_description);
        println!(
            "  size:    {}x{}  pixel format {}",
            probe.info.width, probe.info.height, probe.info.pixel_format
        );
        println!("  fps:     {}", probe.info.fps);
        println!(
            "  frames:  {}",
            if probe.info.estimated_frames > 0 {
                probe.info.estimated_frames.to_string()
            } else {
                "unknown".to_owned()
            }
        );
        println!("  audio:   {}", if probe.info.has_audio { "yes (passthrough if compatible)" } else { "no" });
        return Ok(());
    }

    // ---- 4. Full run: decode → GPU shaders → encode. ----
    let output = match &args.output {
        Some(o) => o.clone(),
        None => default_output(&args.input)?,
    };
    if output.exists() {
        log::warn!("output `{}` already exists – overwriting", output.display());
    }

    let video_proc = video::VideoProcessor::new(probe, &output.to_string_lossy(), args.bitrate)?;
    let mut frame_proc = gpu::FrameProcessor::new(gpu, video_proc.info.width, video_proc.info.height, compiled)?;

    let total = video_proc.info.estimated_frames.max(1);
    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) {msg} eta {eta}",
        )
        .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
        .progress_chars("#>-"),
    );
    pb.set_message("starting");

    let start = Instant::now();
    let stats = video_proc.run(|data, stride| {
        let processed = frame_proc.process_frame(data, stride)?;
        pb.inc(1);
        let elapsed = start.elapsed().as_secs_f64();
        let done = pb.position();
        if done > 0 && elapsed > 0.0 {
            pb.set_message(format!("{:.1} fps", done as f64 / elapsed));
        }
        Ok(processed)
    })?;
    pb.finish_and_clear();

    let elapsed = start.elapsed().as_secs_f64();
    log::info!(
        "processed {} frames in {:.2}s ({:.1} fps)",
        stats.frames,
        elapsed,
        if elapsed > 0.0 { stats.frames as f64 / elapsed } else { 0.0 }
    );
    if stats.audio_packets > 0 {
        log::info!("copied {} audio packets", stats.audio_packets);
    }
    println!("done: {} frames → {}", stats.frames, output.display());
    Ok(())
}

/// Default output path: `out_<basename>.<ext>` next to the input.
fn default_output(input: &Path) -> Result<PathBuf> {
    let file_name = input
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "video".to_owned());
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4")
        .to_ascii_lowercase();
    let stem = file_name
        .strip_suffix(&format!(".{ext}"))
        .unwrap_or(&file_name);
    Ok(PathBuf::from(format!("out_{stem}.{ext}")))
}

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();
}
