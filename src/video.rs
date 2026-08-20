//! FFmpeg-based video decoding and encoding.
//!
//! Input is decoded (any container/codec ffmpeg supports; MP4/WebM are the
//! primary targets) and scaled to RGBA for the GPU. Processed RGBA frames are
//! converted back to YUV420P and encoded with `libx264` (`.mp4`) or
//! `libvpx-vp9` (`.webm`). Audio streams are passed through untouched when the
//! target container supports the codec, otherwise they are dropped with a
//! warning.

use anyhow::{anyhow, bail, Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg::{codec, format, frame, media, packet, software, Dictionary, Rational};
use std::path::Path;

/// Static information about the input video.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub container: String,
    pub codec_name: String,
    pub codec_description: String,
    pub pixel_format: String,
    pub estimated_frames: u64,
    pub has_audio: bool,
    pub input_path: String,
}

/// An opened input: decoder plus everything needed to feed the GPU.
pub struct Probe {
    pub info: VideoInfo,
    pub ictx: format::context::Input,
    pub video_stream_index: usize,
    pub decoder: ffmpeg::decoder::Video,
    pub scaler_to_rgba: software::scaling::Context,
    pub rgba_frame: frame::Video,
}

struct AudioMapping {
    ist_index: usize,
    ost_index: usize,
    ist_time_base: Rational,
    ost_time_base: Rational,
}

/// Encoder settings per output extension.
struct EncoderSettings {
    codec_name: &'static str,
    options: Vec<(&'static str, &'static str)>,
    /// Audio codecs (lowercase names) that may be copied into this container.
    allowed_audio: &'static [&'static str],
}

fn encoder_settings(ext: &str) -> Result<EncoderSettings> {
    match ext {
        "mp4" => Ok(EncoderSettings {
            codec_name: "libx264",
            options: vec![("preset", "medium"), ("crf", "18")],
            allowed_audio: &["aac", "mp3", "opus", "alac", "flac"],
        }),
        "webm" => Ok(EncoderSettings {
            codec_name: "libvpx-vp9",
            options: vec![
                ("crf", "32"),
                ("deadline", "good"),
                ("cpu-used", "4"),
                ("row-mt", "1"),
            ],
            allowed_audio: &["opus", "vorbis"],
        }),
        other => bail!(
            "unsupported output extension `.{other}` – choose `.mp4` or `.webm` \
             (H.264 / VP9 respectively)"
        ),
    }
}

/// Create an encoder codec context from a codec + stream parameters, applying
/// the codec's own defaults.
fn encoder_context(
    codec: ffmpeg::Codec,
    parameters: &codec::Parameters,
) -> Result<codec::context::Context> {
    unsafe {
        let ptr = ffmpeg_sys_next::avcodec_alloc_context3(codec.as_ptr());
        if ptr.is_null() {
            bail!("failed to allocate encoder context");
        }
        let mut ctx = codec::context::Context::wrap(ptr, None);
        let ret = ffmpeg_sys_next::avcodec_parameters_to_context(
            ctx.as_mut_ptr(),
            parameters.as_ptr(),
        );
        if ret < 0 {
            bail!("failed to copy encoder parameters: {}", ffmpeg::Error::from(ret));
        }
        Ok(ctx)
    }
}

/// Estimate the number of video frames of a stream, for the progress bar.
fn estimate_frames(stream: &format::stream::Stream, fps: Rational) -> u64 {
    if stream.frames() > 0 {
        return stream.frames() as u64;
    }
    let duration = stream.duration();
    if duration > 0 {
        let tb = stream.time_base();
        // frames = duration * tb * fps
        let num = duration as i128 * tb.0 as i128 * fps.0 as i128;
        let den = tb.1 as i128 * fps.1 as i128;
        if den > 0 {
            return (num / den).max(0) as u64;
        }
    }
    0
}

/// Open an input video and prepare decoding. Used by both `--dry-run` and the
/// full run.
pub fn probe(input: &str) -> Result<Probe> {
    let ictx = format::input(&input)
        .with_context(|| format!("cannot open input video `{input}`"))?;
    let video_stream = ictx
        .streams()
        .best(media::Type::Video)
        .ok_or_else(|| anyhow!("no video stream found in `{input}`"))?;
    let video_stream_index = video_stream.index();

    let codec_id = video_stream.parameters().id();
    let codec_info = ffmpeg::codec::decoder::find(codec_id);
    let codec_name = codec_info.map(|c| c.name().to_string()).unwrap_or_else(|| format!("{codec_id:?}"));
    let codec_description =
        codec_info.map(|c| c.description().to_string()).unwrap_or_default();
    let context = codec::context::Context::from_parameters(video_stream.parameters())
        .context("failed to create decoder context")?;
    let decoder = context.decoder().video().context("video stream is not decodable")?;

    let width = decoder.width();
    let height = decoder.height();
    if width == 0 || height == 0 {
        bail!("video stream has zero-sized frames");
    }
    let fps = decoder.frame_rate().unwrap_or(Rational(30, 1));
    let pixel_format = decoder
        .format()
        .descriptor()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|| format!("{:?}", decoder.format()));
    let has_audio = ictx.streams().any(|s| s.parameters().medium() == media::Type::Audio);
    let estimated_frames = estimate_frames(&video_stream, fps);

    log::info!(
        "input: {}  container {}  codec {} ({})  {}x{}  pixel format {}  fps {}  ~{} frames",
        input,
        ictx.format().name(),
        codec_name,
        codec_description,
        width,
        height,
        pixel_format,
        fps,
        if estimated_frames > 0 { estimated_frames.to_string() } else { "unknown".to_owned() },
    );

    let scaler_to_rgba = software::scaling::Context::get(
        decoder.format(),
        width,
        height,
        format::Pixel::RGBA,
        width,
        height,
        software::scaling::Flags::BILINEAR,
    )
    .context("failed to create RGBA conversion context")?;
    let rgba_frame = frame::Video::new(format::Pixel::RGBA, width, height);

    Ok(Probe {
        info: VideoInfo {
            width,
            height,
            fps,
            container: ictx.format().name().to_string(),
            codec_name,
            codec_description,
            pixel_format,
            estimated_frames,
            has_audio,
            input_path: input.to_string(),
        },
        ictx,
        video_stream_index,
        decoder,
        scaler_to_rgba,
        rgba_frame,
    })
}

/// Decode + process + encode pipeline.
pub struct VideoProcessor {
    pub info: VideoInfo,
    ictx: format::context::Input,
    video_stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    scaler_to_rgba: software::scaling::Context,
    rgba_frame: frame::Video,
    octx: format::context::Output,
    encoder: ffmpeg::encoder::Video,
    encoder_time_base: Rational,
    output_video_time_base: Rational,
    scaler_to_yuv: software::scaling::Context,
    audio: Vec<AudioMapping>,
    frame_index: u64,
}

/// Number of frames processed by a run.
#[derive(Default, Debug)]
pub struct ProcessStats {
    pub frames: u64,
    pub audio_packets: u64,
}

impl VideoProcessor {
    /// Attach an output muxer/encoder to a probed input.
    pub fn new(probe: Probe, output: &str, bitrate: Option<u64>) -> Result<Self> {
        let info = probe.info.clone();
        let Probe {
            ictx,
            video_stream_index,
            decoder,
            scaler_to_rgba,
            rgba_frame,
            ..
        } = probe;

        let width = info.width;
        let height = info.height;
        let fps = info.fps;

        let extension = Path::new(output)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let settings = encoder_settings(&extension)?;

        let mut octx = format::output(&output)
            .with_context(|| format!("cannot open output `{output}`"))?;

        let codec = ffmpeg::encoder::find_by_name(settings.codec_name).ok_or_else(|| {
            anyhow!(
                "encoder `{}` was not found in this ffmpeg build – the `.{extension}` \
                 output format requires it",
                settings.codec_name
            )
        })?;
        let mut ost = octx.add_stream(codec)?;
        // Create the encoder context with avcodec_alloc_context3(codec) so the
        // codec's own defaults (e.g. x264's "unset" sentinels) are applied.
        // Context::from_parameters would use generic ffmpeg defaults, which
        // libx264 rejects as "broken ffmpeg default settings".
        let mut encoder = encoder_context(codec, &ost.parameters())
            .context("failed to create encoder context")?
            .encoder()
            .video()?;
        encoder.set_width(width);
        encoder.set_height(height);
        encoder.set_format(format::Pixel::YUV420P);
        encoder.set_frame_rate(Some(fps));
        let encoder_time_base = fps.invert();
        encoder.set_time_base(encoder_time_base);
        if let Some(bitrate) = bitrate {
            encoder.set_bit_rate(bitrate as usize);
        }
        let mut options = Dictionary::new();
        for (key, value) in &settings.options {
            if bitrate.is_none() {
                options.set(key, *value);
            }
        }
        let opened = encoder
            .open_as_with(codec, options)
            .map_err(|e| anyhow!("failed to open `{}` encoder: {e}", settings.codec_name))?;
        ost.set_parameters(&opened);
        let encoder = opened;
        log::info!(
            "output: {}  encoder {}  {}x{}  fps {}  yuv420p",
            output,
            settings.codec_name,
            width,
            height,
            fps
        );

        // Audio passthrough (only if the target container supports the codec).
        let mut audio = Vec::new();
        let mut next_ost_index = 1usize;
        for ist in ictx.streams() {
            if ist.parameters().medium() != media::Type::Audio {
                continue;
            }
            let audio_codec = format!("{:?}", ist.parameters().id()).to_ascii_lowercase();
            if !settings.allowed_audio.contains(&audio_codec.as_str()) {
                log::warn!(
                    "audio stream (codec `{audio_codec}`) cannot be muxed into a `.{extension}` \
                     container – dropping audio"
                );
                continue;
            }
            let mut aost = octx.add_stream(ffmpeg::encoder::find(codec::Id::None))?;
            aost.set_parameters(ist.parameters());
            // Clear the codec tag so muxing into another container works.
            let mut params = aost.parameters();
            unsafe {
                (*params.as_mut_ptr()).codec_tag = 0;
            }
            audio.push(AudioMapping {
                ist_index: ist.index(),
                ost_index: next_ost_index,
                ist_time_base: ist.time_base(),
                ost_time_base: Rational(0, 0), // filled in after write_header
            });
            log::info!(
                "audio stream {} (codec `{audio_codec}`) copied to output stream {}",
                ist.index(),
                next_ost_index
            );
            next_ost_index += 1;
        }

        octx.set_metadata(ictx.metadata().to_owned());
        format::context::output::dump(&octx, 0, Some(output));
        octx.write_header().context("failed to write output header")?;

        // Output stream time bases are final after the header is written.
        let output_video_time_base =
            octx.stream(0).map(|s| s.time_base()).unwrap_or(encoder_time_base);
        for mapping in &mut audio {
            mapping.ost_time_base = octx
                .stream(mapping.ost_index)
                .map(|s| s.time_base())
                .unwrap_or(mapping.ist_time_base);
        }

        let scaler_to_yuv = software::scaling::Context::get(
            format::Pixel::RGBA,
            width,
            height,
            format::Pixel::YUV420P,
            width,
            height,
            software::scaling::Flags::BILINEAR,
        )
        .context("failed to create YUV conversion context")?;

        Ok(VideoProcessor {
            info,
            ictx,
            video_stream_index,
            decoder,
            scaler_to_rgba,
            rgba_frame,
            octx,
            encoder,
            encoder_time_base,
            output_video_time_base,
            scaler_to_yuv,
            audio,
            frame_index: 0,
        })
    }

    /// Decode all frames, hand each RGBA frame to `process`, and encode the
    /// result. `process` receives `(rgba, stride)` and must return a tightly
    /// packed RGBA8 frame.
    pub fn run<F>(self, mut process: F) -> Result<ProcessStats>
    where
        F: FnMut(&[u8], usize) -> Result<Vec<u8>>,
    {
        let VideoProcessor {
            info,
            mut ictx,
            video_stream_index,
            mut decoder,
            mut scaler_to_rgba,
            mut rgba_frame,
            mut octx,
            mut encoder,
            encoder_time_base,
            output_video_time_base,
            mut scaler_to_yuv,
            audio,
            mut frame_index,
        } = self;

        let mut stats = ProcessStats::default();

        for (stream, mut packet) in ictx.packets() {
            let ist_index = stream.index();
            if ist_index == video_stream_index {
                packet.rescale_ts(stream.time_base(), decoder.time_base());
                decoder
                    .send_packet(&packet)
                    .map_err(|e| anyhow!("failed to send packet to decoder: {e}"))?;
                drain_decoded_frames(
                    &mut decoder,
                    &mut scaler_to_rgba,
                    &mut rgba_frame,
                    &mut scaler_to_yuv,
                    &mut encoder,
                    encoder_time_base,
                    output_video_time_base,
                    &mut octx,
                    &mut frame_index,
                    &info,
                    &mut process,
                    &mut stats,
                )?;
            } else if let Some(mapping) = audio.iter().find(|m| m.ist_index == ist_index) {
                packet.rescale_ts(mapping.ist_time_base, mapping.ost_time_base);
                packet.set_position(-1);
                packet.set_stream(mapping.ost_index);
                packet
                    .write_interleaved(&mut octx)
                    .map_err(|e| anyhow!("failed to write audio packet: {e}"))?;
                stats.audio_packets += 1;
            } else {
                log::debug!("skipping non-video stream {ist_index}");
            }
        }

        // Flush decoder, then encoder, then the muxer.
        decoder
            .send_eof()
            .map_err(|e| anyhow!("failed to flush decoder: {e}"))?;
        drain_decoded_frames(
            &mut decoder,
            &mut scaler_to_rgba,
            &mut rgba_frame,
            &mut scaler_to_yuv,
            &mut encoder,
            encoder_time_base,
            output_video_time_base,
            &mut octx,
            &mut frame_index,
            &info,
            &mut process,
            &mut stats,
        )?;
        encoder
            .send_eof()
            .map_err(|e| anyhow!("failed to flush encoder: {e}"))?;
        drain_encoded_packets(&mut encoder, encoder_time_base, output_video_time_base, &mut octx)?;
        octx.write_trailer().context("failed to write output trailer")?;

        Ok(stats)
    }
}

/// Pull decoded frames from the decoder, process them and feed the encoder.
#[allow(clippy::too_many_arguments)]
fn drain_decoded_frames<F>(
    decoder: &mut ffmpeg::decoder::Video,
    scaler_to_rgba: &mut software::scaling::Context,
    rgba_frame: &mut frame::Video,
    scaler_to_yuv: &mut software::scaling::Context,
    encoder: &mut ffmpeg::encoder::Video,
    encoder_time_base: Rational,
    output_video_time_base: Rational,
    octx: &mut format::context::Output,
    frame_index: &mut u64,
    info: &VideoInfo,
    process: &mut F,
    stats: &mut ProcessStats,
) -> Result<()>
where
    F: FnMut(&[u8], usize) -> Result<Vec<u8>>,
{
    let w = info.width as usize;
    let h = info.height as usize;
    let tight = w * 4;
    let mut decoded = frame::Video::empty();

    loop {
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => break,
            Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => break,
            Err(e) => return Err(anyhow!("decode error: {e}")),
        }

        scaler_to_rgba
            .run(&decoded, rgba_frame)
            .map_err(|e| anyhow!("RGBA conversion failed: {e}"))?;
        let stride = rgba_frame.stride(0);
        let data = &rgba_frame.data(0)[..stride * h];

        let processed = process(data, stride)?;
        if processed.len() != tight * h {
            bail!(
                "processor returned {} bytes, expected {} ({}x{} RGBA)",
                processed.len(),
                tight * h,
                w,
                h
            );
        }

        let dst_stride = rgba_frame.stride(0);
        {
            let dst = rgba_frame.data_mut(0);
            for y in 0..h {
                dst[y * dst_stride..y * dst_stride + tight]
                    .copy_from_slice(&processed[y * tight..(y + 1) * tight]);
            }
        }
        // Allocate a fresh YUV frame per encode: avcodec_send_frame hands the
        // encoder references to the frame's buffers, and threaded encoders
        // (x264/vp9) may still read them asynchronously, so reusing one frame
        // would race with the encoder.
        let mut fresh_yuv = frame::Video::new(format::Pixel::YUV420P, info.width, info.height);
        fresh_yuv.set_pts(Some(*frame_index as i64));
        fresh_yuv.set_kind(ffmpeg::picture::Type::None);
        scaler_to_yuv
            .run(rgba_frame, &mut fresh_yuv)
            .map_err(|e| anyhow!("YUV conversion failed: {e}"))?;
        encoder
            .send_frame(&fresh_yuv)
            .map_err(|e| anyhow!("encoder error: {e}"))?;

        *frame_index += 1;
        stats.frames += 1;
        drain_encoded_packets(encoder, encoder_time_base, output_video_time_base, octx)?;
    }
    Ok(())
}

/// Pull encoded packets from the encoder into the muxer.
fn drain_encoded_packets(
    encoder: &mut ffmpeg::encoder::Video,
    encoder_time_base: Rational,
    output_video_time_base: Rational,
    octx: &mut format::context::Output,
) -> Result<()> {
    let mut encoded = packet::Packet::empty();
    while encoder.receive_packet(&mut encoded).is_ok() {
        encoded.set_stream(0);
        encoded.rescale_ts(encoder_time_base, output_video_time_base);
        encoded
            .write_interleaved(octx)
            .map_err(|e| anyhow!("failed to write video packet: {e}"))?;
    }
    Ok(())
}
