use std::path::Path;

use ash::vk::{self, Handle};
use ash::{Device, Instance};

use anyhow::{Result, anyhow};
use image::ImageReader;

use crate::buffer::{Buffer, BufferType};
use crate::{begin_single_time_commands, end_single_time_commands, get_memory_type_index, set_object_name, RendererData};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MipLevels {
    #[default]
    One,
    Value(u32),
    Maximum,
}

impl MipLevels {
    pub fn mips_1d(&self, width: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width as f32).log2().floor() as u32 + 1,
        }
    }

    pub fn mips_2d(&self, width: u32, height: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width.max(height) as f32).log2().floor() as u32 + 1,
        }
    }

    pub fn mips_3d(&self, width: u32, height: u32, depth: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width.max(height).max(depth) as f32).log2().floor() as u32 + 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    mip_level: u32,
    sampler: Option<vk::Sampler>,
    extent: vk::Extent3D,
}

impl Image {
    pub fn new_1d(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        width: u32,
        mip_levels: MipLevels,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        view_type: vk::ImageViewType,
        layer_count: u32,
        properties: vk::MemoryPropertyFlags,
        aspects: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &str
    ) -> Self {
        let mips = mip_levels.mips_1d(width);

        let (image, memory) = unsafe {
            create_image_1d(
                instance,
                device,
                data,
                width,
                mips,
                layer_count,
                samples,
                format,
                tiling,
                usage,
                properties,
            )
        }
        .unwrap();

        set_object_name(instance, device, name, image).unwrap();
        set_object_name(instance, device, &format!("{name}_memory"), memory).unwrap();

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        set_object_name(instance, device, &format!("{name}_view"), view).unwrap();

        let sampler = match sampler {
            Some((filter, address_mode)) => {
                let sampler = unsafe { create_texture_sampler(
                    instance,
                    device,
                    &mips,
                    &filter,
                    &address_mode,
                    ""
                ) }.unwrap();
                set_object_name(instance, device, &format!("{name}_sampler"), sampler).unwrap();
                Some(sampler)
            },
            None => None,
        };

        Image {
            image,
            memory,
            view,
            mip_level: mips,
            sampler,
            extent: vk::Extent3D { width, height: 1, depth: 1 }
        }
    }

    pub fn new_2d(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        width: u32,
        height: u32,
        mip_levels: MipLevels,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        view_type: vk::ImageViewType,
        layer_count: u32,
        properties: vk::MemoryPropertyFlags,
        aspects: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &str,
    ) -> Self {
        let mips = mip_levels.mips_2d(width, height);

        let (image, memory) = unsafe {
            create_image_2d(
                instance,
                device,
                data,
                width,
                height,
                mips,
                layer_count,
                samples,
                format,
                tiling,
                usage,
                properties,
            )
        }
        .unwrap();

        set_object_name(instance, device, name, image).unwrap();
        set_object_name(instance, device, &format!("{name}_memory"), memory).unwrap();

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        set_object_name(instance, device, &format!("{name}_view"), view).unwrap();

        let sampler = match sampler {
            Some((filter, address_mode)) => {
                let sampler = unsafe { create_texture_sampler(
                    instance,
                    device,
                    &mips,
                    &filter,
                    &address_mode,
                    ""
                ) }.unwrap();
                set_object_name(instance, device, &format!("{name}_sampler"), sampler).unwrap();
                Some(sampler)
            },
            None => None,
        };

        Image {
            image,
            memory,
            view,
            mip_level: mips,
            sampler,
            extent: vk::Extent3D { width, height, depth: 1 }
        }
    }

    pub fn new_3d(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        width: u32,
        height: u32,
        depth: u32,
        mip_levels: MipLevels,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        view_type: vk::ImageViewType,
        layer_count: u32,
        properties: vk::MemoryPropertyFlags,
        aspects: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &str,
    ) -> Self {
        let mips = mip_levels.mips_3d(width, height, depth);

        let (image, memory) = unsafe {
            create_image_3d(
                instance,
                device,
                data,
                width,
                height,
                depth,
                mips,
                layer_count,
                samples,
                format,
                tiling,
                usage,
                properties,
            )
        }
        .unwrap();

        set_object_name(instance, device, name, image).unwrap();
        set_object_name(instance, device, &format!("{name}_memory"), memory).unwrap();

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        set_object_name(instance, device, &format!("{name}_view"), view).unwrap();

        let sampler = match sampler {
            Some((filter, address_mode)) => {
                let sampler = unsafe { create_texture_sampler(
                    instance,
                    device,
                    &mips,
                    &filter,
                    &address_mode,
                    ""
                ) }.unwrap();
                set_object_name(instance, device, &format!("{name}_sampler"), sampler).unwrap();
                Some(sampler)
            },
            None => None,
        };

        Image {
            image,
            memory,
            view,
            mip_level: mips,
            sampler,
            extent: vk::Extent3D { width, height, depth }
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        path: P,
        mip_levels: MipLevels,
        sampler: Option<(Filter, AddressMode)>,
    ) -> Image {
        let img = ImageReader::open(path).unwrap().decode().unwrap();
        let bytes = img.as_bytes();

        let width = img.width();
        let height = img.height();

        let (bpc, format) = match (img.color(), img.color_space().primaries) {
            (image::ColorType::L8, image::metadata::CicpColorPrimaries::SRgb) => (1, vk::Format::R8_SRGB),
            (image::ColorType::L8, _) => (1, vk::Format::R8_UNORM),
            (image::ColorType::La8, image::metadata::CicpColorPrimaries::SRgb) => (2, vk::Format::R8G8_SRGB),
            (image::ColorType::La8, _) => (2, vk::Format::R8G8_UNORM),
            (image::ColorType::Rgb8, image::metadata::CicpColorPrimaries::SRgb) => (3, vk::Format::R8G8B8_SRGB),
            (image::ColorType::Rgb8, _) => (3, vk::Format::R8G8B8_UNORM),
            (image::ColorType::Rgba8, image::metadata::CicpColorPrimaries::SRgb) => (4, vk::Format::R8G8B8A8_SRGB),
            (image::ColorType::Rgba8, _) => (4, vk::Format::R8G8B8A8_UNORM),
            (image::ColorType::L16, _) => (2, vk::Format::R16_UNORM),
            (image::ColorType::La16, _) => (4, vk::Format::R16G16_UNORM),
            (image::ColorType::Rgb16, _) => (6, vk::Format::R16G16B16_UNORM),
            (image::ColorType::Rgba16, _) => (8, vk::Format::R16G16B16A16_UNORM),
            (image::ColorType::Rgb32F, _) => (12, vk::Format::R32G32B32_SFLOAT),
            (image::ColorType::Rgba32F, _) => (16, vk::Format::R32G32B32A32_SFLOAT),
            (color, space) => panic!("unknown ColorType-ColorSpace combination: {:?}, {:?}", color, space),
        };

        let size = (width * height * bpc) as u64;

        let mips = mip_levels.mips_2d(width, height);

        let (image, image_memory) = unsafe {
            create_texture_image(
                instance,
                device,
                data,
                size,
                bytes,
                width,
                height,
                mips,
                format
            )
        }.unwrap();

        let view = unsafe {
            create_image_view(
                device,
                image,
                format,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                mips,
                1,
            )
        }.unwrap();

        let sampler = sampler.map(|(filter, mode)| unsafe {
            create_texture_sampler(
                instance,
                device,
                &mips,
                &filter,
                &mode,
                "name"
            )
        }.unwrap());

        Image {
            image,
            memory: image_memory,
            view,
            mip_level: mips,
            sampler: sampler,
            extent: vk::Extent3D { width, height, depth: 1 },
        }
    }

    /// # Warning !!!
    /// This function produces an invalid image where all pointers are null. \
    /// this should only be used as a default and then replaced with a properly created image.
    pub fn null() -> Self {
        Image {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            mip_level: 0,
            sampler: None,
            extent: vk::Extent3D::default(),
        }
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn mip_level(&self) -> u32 {
        self.mip_level
    }

    pub fn sampler(&self) -> Option<vk::Sampler> {
        self.sampler
    }

    pub fn width(&self) -> u32 {
        self.extent.width
    }

    pub fn height(&self) -> u32 {
        self.extent.height
    }

    pub fn depth(&self) -> u32 {
        self.extent.depth
    }

    pub fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D { width: self.extent.width, height: self.extent.height }
    }

    pub fn extent_3d(&self) -> vk::Extent3D {
        self.extent
    }

    pub fn destroy(&mut self, device: &Device) {
        if let Some(sampler) = self.sampler {
            if !sampler.is_null() {
                unsafe { device.destroy_sampler(sampler, None) };
            }
        }
        if !self.view.is_null() {
            unsafe { device.destroy_image_view(self.view, None) };
        }
        if !self.image.is_null() {
            unsafe { device.destroy_image(self.image, None) };
        }
        if !self.memory.is_null() {
            unsafe { device.free_memory(self.memory, None) };
        }
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
        self.view = vk::ImageView::null();
        self.mip_level = 0;
        self.sampler = None;
        self.extent = vk::Extent3D::default();
    }

    pub fn descriptor_info(&self, layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(self.view())
            .sampler(if let Some(sampler) = self.sampler() {sampler} else {vk::Sampler::null()})
    }
}

pub unsafe fn create_image_1d(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    width: u32,
    mip_levels: u32,
    layers: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_1D)
        .extent(vk::Extent3D {
            width,
            height: 1,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(layers)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

pub unsafe fn create_image_2d(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    width: u32,
    height: u32,
    mip_levels: u32,
    layers: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(layers)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

pub unsafe fn create_image_3d(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
    layers: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .extent(vk::Extent3D {
            width,
            height,
            depth,
        })
        .mip_levels(mip_levels)
        .array_layers(layers)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

pub unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    size: vk::DeviceSize,
    pixels: &[u8],
    width: u32,
    height: u32,
    mip_levels: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let mut staging_buffer = Buffer::new(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        BufferType::HostLocal,
        "Image staging buffer",
    )?;

    // Copy (staging)

    staging_buffer.copy_array_into_buffer(device, pixels);

    // Create (image)

    let (texture_image, texture_image_memory) = create_image_2d(
        instance,
        device,
        data,
        width,
        height,
        mip_levels,
        1,
        vk::SampleCountFlags::TYPE_1,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Transition + Copy (image)

    transition_image_layout(
        instance,
        device,
        data,
        texture_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;

    copy_buffer_to_image(
        instance,
        device,
        data,
        staging_buffer.buffer(),
        texture_image,
        width,
        height,
    )?;

    // Cleanup

    staging_buffer.destroy(device);

    // Mipmaps

    generate_mipmaps(
        instance,
        device,
        data,
        texture_image,
        format,
        width,
        height,
        mip_levels,
    )?;

    Ok((texture_image, texture_image_memory))
}

/// Transitions image layout  \
/// Creates its own one time use command buffers so do not use when drawing
unsafe fn transition_image_layout(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition!")),
        };

    let command_buffer = begin_single_time_commands(
        instance,
        device,
        data,
        "Transition Image Layout",
    )?;

    let subresource = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );

    end_single_time_commands(instance, device, data, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer =
        begin_single_time_commands(instance, device, data, "Copy Buffer To Image")?;

    let subresource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(instance, device, data, command_buffer)?;

    Ok(())
}

/// Copies one image to another
/// - `src` image must be in [TRANSFER_SRC_OPTIMAL](vk::ImageLayout::TRANSFER_SRC_OPTIMAL) image layout
/// - `dst` image must be in [TRANSFER_DST_OPTIMAL](vk::ImageLayout::TRANSFER_DST_OPTIMAL) image layout
pub fn copy_image_to_image(device: &Device, command_buffer: vk::CommandBuffer, src: vk::Image, dst: vk::Image, src_size: vk::Extent2D, dst_size: vk::Extent2D, filter: vk::Filter) {
    let regions = [
        vk::ImageBlit2::default()
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::default()
                    .x(src_size.width as i32)
                    .y(src_size.height as i32)
                    .z(1)
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D::default()
                    .x(dst_size.width as i32)
                    .y(dst_size.height as i32)
                    .z(1)
            ])
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0)
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0)
            )
    ];
    
    let blit_image_info = vk::BlitImageInfo2::default()
        .src_image(src)
        .dst_image(dst)
        
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        
        .regions(&regions)
        .filter(filter);
    unsafe { device.cmd_blit_image2(command_buffer, &blit_image_info) };
}

pub unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    view_type: vk::ImageViewType,
    mip_levels: u32,
    layer_count: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(layer_count);

    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

pub unsafe fn create_texture_sampler(
    instance: &Instance,
    device: &Device,
    mip_level: &u32,
    filter: &Filter,
    address_mode: &AddressMode,
    name: &str,
) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(filter.mag)
        .min_filter(filter.min)
        .address_mode_u(address_mode.u)
        .address_mode_v(address_mode.v)
        .address_mode_w(address_mode.w)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(*mip_level as f32)
        .mip_lod_bias(0.0);

    let texture_sampler = device.create_sampler(&info, None)?;
    set_object_name(
        instance,
        device,
        &format!("{} Texture Sampler", name),
        texture_sampler,
    )
    .unwrap();

    Ok(texture_sampler)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    pub min: vk::Filter,
    pub mag: vk::Filter,
}

impl Filter {
    pub const LINEAR: Self = {
        Filter {
            min: vk::Filter::LINEAR,
            mag: vk::Filter::LINEAR,
        }
    };

    pub const NEAREST: Self = {
        Filter {
            min: vk::Filter::NEAREST,
            mag: vk::Filter::NEAREST,
        }
    };

    pub const MIN_LINEAR: Self = {
        Filter {
            min: vk::Filter::LINEAR,
            mag: vk::Filter::NEAREST,
        }
    };

    pub const MAG_LINEAR: Self = {
        Filter {
            min: vk::Filter::NEAREST,
            mag: vk::Filter::LINEAR,
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressMode {
    pub u: vk::SamplerAddressMode,
    pub v: vk::SamplerAddressMode,
    pub w: vk::SamplerAddressMode,
}

impl AddressMode {
    pub const REPEAT: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::REPEAT,
            v: vk::SamplerAddressMode::REPEAT,
            w: vk::SamplerAddressMode::REPEAT,
        }
    };

    pub const MIRRORED_REPEAT: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            w: vk::SamplerAddressMode::MIRRORED_REPEAT,
        }
    };

    pub const MIRROR_CLAMP_TO_EDGE: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            v: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            w: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        }
    };
    
    pub const CLAMP_TO_BORDER: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    };
    
    pub const CLAMP_TO_EDGE: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    };
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // Support

    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!(
            "Texture image format does not support linear blitting for mipmaps"
        ));
    }

    // Mipmaps

    let command_buffer =
        begin_single_time_commands(instance, device, data, "Generate MipMaps")?;

    let subresource = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );

    end_single_time_commands(instance, device, data, command_buffer)?;

    Ok(())
}
