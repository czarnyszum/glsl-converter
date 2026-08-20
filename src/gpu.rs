//! GPU setup and per-frame shader execution via wgpu.
//!
//! Two RGBA8 textures are ping-ponged through the shader chain; every pass
//! binds the current input texture as `MAIN_tex` (plus every `BIND`), renders
//! a fullscreen triangle, and writes into the other texture. The final result
//! is copied into a staging buffer and mapped back to the CPU.

use anyhow::{anyhow, bail, Context, Result};

/// Fullscreen-triangle vertex shader (WGSL).
pub const VERTEX_SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var out: VsOut;
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    )[idx];
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}
"#;

/// A compiled shader pass: pipeline + per-pass uniform buffer. Bind groups are
/// created once the frame textures exist (see [`FrameProcessor::new`]).
pub struct PassPipeline {
    pub desc: String,
    pub strength: f32,
    pub texture_count: usize,
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
}

/// One compiled shader pass, with bind groups for both ping-pong textures.
struct CompiledPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    _params: wgpu::Buffer,
}

/// GPU handle plus the per-frame processing state.
pub struct FrameProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    tex_a: wgpu::Texture,
    tex_b: wgpu::Texture,
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    staging: wgpu::Buffer,
    staging_bytes_per_row: u32,
    passes: Vec<CompiledPass>,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

/// Initialise wgpu and pick a GPU adapter.
pub async fn init() -> Result<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|e| anyhow!("no GPU adapter available: {e} (is a Vulkan/GL driver installed?)"))?;

    let info = adapter.get_info();
    log::info!(
        "GPU adapter: {}  backend={:?} type={:?}",
        info.name,
        info.backend,
        info.device_type
    );
    log::info!(
        "  driver: {}  version: {}  vendor: {:#x}  device: {:#x}",
        info.driver,
        info.driver_info,
        info.vendor,
        info.device
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("video-shader device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .map_err(|e| anyhow!("failed to create GPU device: {e}"))?;

    let limits = device.limits();
    log::info!(
        "max texture size: {}x{}",
        limits.max_texture_dimension_2d,
        limits.max_texture_dimension_2d
    );

    Ok(Gpu { device, queue, adapter_info: info, limits })
}

/// Validate a GLSL fragment shader with naga (same version wgpu uses) so the
/// user gets a readable error instead of a pipeline-creation failure.
pub fn validate_glsl(source: &str) -> Result<()> {
    let mut frontend = wgpu::naga::front::glsl::Frontend::default();
    let module = frontend
        .parse(&wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Fragment), source)
        .map_err(|e| anyhow!("shader rejected by the GLSL frontend: {e}"))?;
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| anyhow!("shader failed validation: {e}"))?;
    Ok(())
}

fn layout_entries(texture_count: usize) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(4),
        },
        count: None,
    }];
    for k in 0..texture_count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (1 + 2 * k) as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (2 + 2 * k) as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    entries
}

impl Gpu {
    /// Compile one pass. `glsl_source` must already be preprocessed and
    /// `texture_count` must match the number of `uniform sampler2D` bindings it
    /// declares (MAIN plus every BIND).
    pub fn compile_pass(
        &self,
        glsl_source: &str,
        texture_count: usize,
        strength: f32,
        desc: &str,
    ) -> Result<PassPipeline> {
        validate_glsl(glsl_source).with_context(|| format!("shader `{desc}` is invalid"))?;

        let device = &self.device;
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vs"),
            source: wgpu::ShaderSource::Wgsl(VERTEX_SHADER.into()),
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(desc),
            source: wgpu::ShaderSource::Glsl {
                shader: glsl_source.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: &[],
            },
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pass bind group layout"),
            entries: &layout_entries(texture_count),
        });
        let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pass pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc),
            layout: Some(&ppl),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // Uniform block: { float strength; } padded to 16 bytes.
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params, 0, &strength.to_le_bytes());

        Ok(PassPipeline {
            desc: desc.to_string(),
            strength,
            texture_count,
            pipeline,
            params,
        })
    }
}

impl FrameProcessor {
    /// Create the frame textures and bind groups for all passes.
    pub fn new(gpu: Gpu, width: u32, height: u32, passes: Vec<PassPipeline>) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("cannot process a zero-sized frame ({width}x{height})");
        }
        let limit = gpu.limits.max_texture_dimension_2d;
        if width > limit || height > limit {
            bail!(
                "frame size {width}x{height} exceeds the GPU texture limit {limit} \
                 (choose a smaller resolution or a GPU with larger limits)"
            );
        }

        let device = &gpu.device;
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("frame"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let tex_a = device.create_texture(&texture_desc);
        let tex_b = device.create_texture(&texture_desc);
        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frame sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let staging_bytes_per_row = ((width as usize * 4 + 255) / 256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: staging_bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Build the ping-pong bind groups for every pass now that the frame
        // textures exist.
        let passes = passes
            .into_iter()
            .map(|p| {
                let layout = p
                    .pipeline
                    .get_bind_group_layout(0);
                let bind_group_a = make_bind_group(
                    device,
                    &layout,
                    &p.params,
                    &view_a,
                    &sampler,
                    p.texture_count,
                );
                let bind_group_b = make_bind_group(
                    device,
                    &layout,
                    &p.params,
                    &view_b,
                    &sampler,
                    p.texture_count,
                );
                CompiledPass {
                    pipeline: p.pipeline,
                    bind_group_a,
                    bind_group_b,
                    _params: p.params,
                }
            })
            .collect::<Vec<_>>();

        Ok(FrameProcessor {
            device: gpu.device,
            queue: gpu.queue,
            width,
            height,
            tex_a,
            tex_b,
            view_a,
            view_b,
            staging,
            staging_bytes_per_row: staging_bytes_per_row as u32,
            passes,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Run the whole shader chain on one RGBA8 frame.
    ///
    /// `rgba` is `height` rows of `stride` bytes each (rows are tight iff
    /// `stride == width * 4`).
    pub fn process_frame(&mut self, rgba: &[u8], stride: usize) -> Result<Vec<u8>> {
        let w = self.width;
        let h = self.height;
        let tight = w as usize * 4;
        if stride < tight {
            bail!("frame stride {stride} < width*4 {tight}");
        }
        if rgba.len() < stride * h as usize {
            bail!("frame data too small: {} bytes for {stride}*{h}", rgba.len());
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let mut src_is_a = true;
        let mut result: &wgpu::Texture = &self.tex_a;
        // One command encoder and one submit per frame: separate submits per
        // frame crash some software Vulkan drivers (lavapipe).
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        for pass in &self.passes {
            let (input_view, output_texture, output_view) = if src_is_a {
                (&self.view_a, &self.tex_b, &self.view_b)
            } else {
                (&self.view_b, &self.tex_a, &self.view_a)
            };
            let bind_group = if src_is_a { &pass.bind_group_a } else { &pass.bind_group_b };

            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shader pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(&pass.pipeline);
                rp.set_bind_group(0, bind_group, &[]);
                rp.draw(0..3, 0..1);
                // `input_view` is bound via the bind group; keep the borrow alive.
                let _ = input_view;
            }
            result = output_texture;
            src_is_a = !src_is_a;
        }

        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: result,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.staging_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| anyhow!("GPU poll failed: {e}"))?;
        rx.recv().map_err(|_| anyhow!("GPU map callback was never invoked"))??;

        let mapped = slice.get_mapped_range();
        let mut out = vec![0u8; tight * h as usize];
        let row_bytes = self.staging_bytes_per_row as usize;
        for y in 0..h as usize {
            let src = &mapped[y * row_bytes..y * row_bytes + tight];
            out[y * tight..(y + 1) * tight].copy_from_slice(src);
        }
        drop(mapped);
        self.staging.unmap();
        Ok(out)
    }
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    texture_count: usize,
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(1 + 2 * texture_count);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: params,
            offset: 0,
            size: wgpu::BufferSize::new(16),
        }),
    });
    for k in 0..texture_count {
        entries.push(wgpu::BindGroupEntry {
            binding: (1 + 2 * k) as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: (2 + 2 * k) as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass bind group"),
        layout,
        entries: &entries,
    })
}
