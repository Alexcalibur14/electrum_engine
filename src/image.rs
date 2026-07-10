//! This Module contains all structs and functions related to creating [Images](Image)
//! 
//! # Safety
//! All images and related handles must be destroyed before the [device](ash::Device) is.\
//! If the image is added to the RendererData struct or structs it contains, this will be managed automatically.\
//! 
//! Although the [Image] struct impls [Clone] and [Copy] it is neither
//! and although a double free is not possible on the same struct via the [destroy](Image::destroy()) function, that does hold true for [Clone]d versions
use std::path::Path;

use ash::vk::{self, Handle};

use anyhow::{Result, anyhow};
use image::ImageReader;

use crate::buffer::{Buffer, BufferType};
use crate::{RendererData, RenderingDevice, begin_single_time_commands, end_single_time_commands, get_memory_type_index, set_object_name};


/// An enum for choosing mip levels
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MipLevels {
    /// One mip level
    #[default]
    One,
    /// A specific number of mip levels
    Value(u32),
    /// The maximum number of mip levels until the image becomes 1x1x1 in dimentions
    Maximum,
}

impl MipLevels {
    /// Calculates the number of mip levels for a one dimentional image
    pub fn mips_1d(&self, width: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width as f32).log2().floor() as u32 + 1,
        }
    }

    /// Calculates the number of mip levels for a two dimentional image
    pub fn mips_2d(&self, width: u32, height: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width.max(height) as f32).log2().floor() as u32 + 1,
        }
    }

    /// Calculates the number of mip levels for a three dimentional image
    pub fn mips_3d(&self, width: u32, height: u32, depth: u32) -> u32 {
        match self {
            MipLevels::One => 1,
            MipLevels::Value(v) => *v,
            MipLevels::Maximum => (width.max(height).max(depth) as f32).log2().floor() as u32 + 1,
        }
    }
}

/// A wrapper around the [vk::Image], [vk::DeviceMemory], [vk::ImageView], and [vk::Sampler] types
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
    /// Creates an Image
    pub fn new(
        device: &RenderingDevice,
        data: &RendererData,
        extent: vk::Extent3D,
        mip_levels: MipLevels,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        view_type: vk::ImageViewType,
        usage: vk::ImageUsageFlags,
        layer_count: u32,
        properties: vk::MemoryPropertyFlags,
        aspects: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &str
    ) -> Result<Self> {
        let mips = mip_levels.mips_3d(extent.width, extent.height, extent.depth);

        let image_type = match extent {
            vk::Extent3D { height: _, width: 1, depth: 1 } => vk::ImageType::TYPE_1D,
            vk::Extent3D { height: _, width: _, depth: 1 } => vk::ImageType::TYPE_2D,
            _ => vk::ImageType::TYPE_3D,
        };

        let (image, memory) = create_image(
            device,
            data,
            extent,
            image_type,
            mips,
            layer_count,
            samples,
            format,
            tiling,
            usage,
            properties,
        ).unwrap();

        set_object_name(device, name, image).unwrap();
        set_object_name(device, &format!("{name}_memory"), memory).unwrap();

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        set_object_name(device, &format!("{name}_view"), view).unwrap();

        let sampler = match sampler {
            Some((filter, address_mode)) => {
                let sampler = unsafe { create_texture_sampler(
                    device,
                    &mips,
                    &filter,
                    &address_mode,
                    ""
                ) }.unwrap();
                set_object_name(device, &format!("{name}_sampler"), sampler).unwrap();
                Some(sampler)
            },
            None => None,
        };

        Ok(Image {
            image,
            memory,
            view,
            mip_level: mips,
            sampler,
            extent,
        })
    }

    /// Creats an Image with a [View Type](vk::ImageViewType) of one of
    /// * [TYPE_1D](vk::ImageViewType::TYPE_1D)
    /// * [TYPE_2D](vk::ImageViewType::TYPE_2D)
    /// * [TYPE_3D](vk::ImageViewType::TYPE_3D)
    pub fn new_regular(
        device: &RenderingDevice,
        data: &RendererData,
        extent: vk::Extent3D,
        mip_levels: MipLevels,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        layer_count: u32,
        properties: vk::MemoryPropertyFlags,
        aspects: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &str
    ) -> Self {
        let mips = mip_levels.mips_3d(extent.width, extent.height, extent.depth);

        let view_type = match extent {
            vk::Extent3D { height: _, width: 1, depth: 1 } => vk::ImageViewType::TYPE_1D,
            vk::Extent3D { height: _, width: _, depth: 1 } => vk::ImageViewType::TYPE_2D,
            _ => vk::ImageViewType::TYPE_3D,
        };

        let (image, memory) = create_regular_image(
            device,
            data,
            extent,
            mips,
            layer_count,
            samples,
            format,
            tiling,
            usage,
            properties,
        ).unwrap();

        set_object_name(device, name, image).unwrap();
        set_object_name(device, &format!("{name}_memory"), memory).unwrap();

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        set_object_name(device, &format!("{name}_view"), view).unwrap();

        let sampler = match sampler {
            Some((filter, address_mode)) => {
                let sampler = unsafe { create_texture_sampler(
                    device,
                    &mips,
                    &filter,
                    &address_mode,
                    ""
                ) }.unwrap();
                set_object_name(device, &format!("{name}_sampler"), sampler).unwrap();
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
            extent,
        }
    }

    /// Creates an Image and loads an image file into it\
    /// Supports the file types that the Image crate supports
    pub fn load_from_file<P: AsRef<Path>>(
        device: &RenderingDevice,
        data: &RendererData,
        path: P,
        mip_levels: MipLevels,
        sampler: Option<(Filter, AddressMode)>,
    ) -> Image {
        let img = ImageReader::open(&path).unwrap().decode().unwrap();
        let bytes = img.as_bytes();

        let name = path.as_ref().file_prefix().unwrap().to_str().unwrap();

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

        let (image, memory) = unsafe {
            create_texture_image(
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

        set_object_name(device, name, image).unwrap();
        set_object_name(device, &format!("{name}_memory"), memory).unwrap();

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

        set_object_name(device, &format!("{name}_view"), view).unwrap();

        let sampler = sampler.map(|(filter, mode)| unsafe {
            create_texture_sampler(
                device,
                &mips,
                &filter,
                &mode,
                name,
            )
        }.unwrap());

        Image {
            image,
            memory,
            view,
            mip_level: mips,
            sampler: sampler,
            extent: vk::Extent3D { width, height, depth: 1 },
        }
    }

    /// Creates an Image struct
    /// 
    /// # Safety
    /// All of the arguments *must* be properly created
    pub unsafe fn from_raw(
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        mip_level: u32,
        sampler: Option<vk::Sampler>,
        extent: vk::Extent3D,
    ) -> Self {
        Image {
            image,
            memory,
            view,
            mip_level,
            sampler,
            extent,
        }
    }

    /// # Warning !!!
    /// This function produces an invalid image where all pointers are null. \
    /// this should only be used as a default and then replaced with a properly created image.
    pub unsafe fn null() -> Self {
        Image {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            mip_level: 0,
            sampler: None,
            extent: vk::Extent3D::default(),
        }
    }

    /// The Image's vk::Image
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// The Image's device memory
    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// The Image's view
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// The Image's number of mip levels
    pub fn mip_level(&self) -> u32 {
        self.mip_level
    }

    /// The Image's sampler
    pub fn sampler(&self) -> Option<vk::Sampler> {
        self.sampler
    }

    /// The Image's width
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    /// The Image's height
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    /// The Image's depth
    pub fn depth(&self) -> u32 {
        self.extent.depth
    }

    /// The Image's 2D extent
    pub fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D { width: self.extent.width, height: self.extent.height }
    }

    /// The Image's extent
    pub fn extent_3d(&self) -> vk::Extent3D {
        self.extent
    }

    /// Destroys and deallocates driver memory associated with the image
    pub fn destroy(&mut self, device: &RenderingDevice) {
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

    /// Creates a vk::DescriptorImageInfo for allocating Descriptor Sets
    pub fn descriptor_info(&self, layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(self.view())
            .sampler(if let Some(sampler) = self.sampler() {sampler} else {vk::Sampler::null()})
    }
}

/// Creats a [vk::Image] with an [Image Type](vk::ImageType) based on the image extent
pub fn create_regular_image(
    device: &RenderingDevice,
    data: &RendererData,
    extent: vk::Extent3D,
    mip_levels: u32,
    layers: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let image_type = match extent {
        vk::Extent3D { height: _, width: 1, depth: 1 } => vk::ImageType::TYPE_1D,
        vk::Extent3D { height: _, width: _, depth: 1 } => vk::ImageType::TYPE_2D,
        _ => vk::ImageType::TYPE_3D,
    };

    create_image(device, data, extent, image_type, mip_levels, layers, samples, format, tiling, usage, properties)
}

/// Creates a [vk::Image]
/// 
/// # Safety
/// The returned [vk::Image] must be destroyed using [destroy_image()](ash::Device::destroy_image())\
/// The returned [vk::DeviceMemory] must be freed using [free_memory()](ash::Device::free_memory())
pub fn create_image(
    device: &RenderingDevice,
    data: &RendererData,
    extent: vk::Extent3D,
    image_type: vk::ImageType,
    mip_levels: u32,
    layers: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(image_type)
        .extent(extent)
        .mip_levels(mip_levels)
        .array_layers(layers)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe { device.create_image(&info, None) }?;

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let info = unsafe { vk::MemoryAllocateInfo::default()
    .allocation_size(requirements.size)
    .memory_type_index(get_memory_type_index(
        &device.instance,
        data,
        properties,
        requirements,
    )?) };

    let image_memory = unsafe { device.allocate_memory(&info, None) }?;

    unsafe { device.bind_image_memory(image, image_memory, 0) }?;

    Ok((image, image_memory))
}

/// Creates a 2D [vk::Image] that is filled with the bytes.\
/// The [image usage](vk::ImageUsageFlags) will be [SAMPLED](vk::ImageUsageFlags::SAMPLED), [TRANSFER_DST](vk::ImageUsageFlags::TRANSFER_DST), and [TRANSFER_SRC](vk::ImageUsageFlags::TRANSFER_SRC)
/// 
/// # Safety
/// The returned [vk::Image] must be destroyed using [destroy_image()](ash::Device::destroy_image())\
/// The returned [vk::DeviceMemory] must be freed using [free_memory()](ash::Device::free_memory())
pub unsafe fn create_texture_image(
    device: &RenderingDevice,
    data: &RendererData,
    size: vk::DeviceSize,
    bytes: &[u8],
    width: u32,
    height: u32,
    mip_levels: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let mut staging_buffer = Buffer::new(
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        BufferType::HostLocal,
        "Image staging buffer",
    )?;

    // Copy (staging)

    staging_buffer.copy_array_into_buffer(device, data, bytes)?;

    // Create (image)

    let (texture_image, texture_image_memory) = create_image(
        device,
        data,
        vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        vk::ImageType::TYPE_2D,
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
        device,
        data,
        texture_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;

    copy_buffer_to_image(
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
    device: &RenderingDevice,
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

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &RenderingDevice,
    data: &RendererData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer =
        begin_single_time_commands(device, data, "Copy Buffer To Image")?;

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

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

/// Copies one image to another
/// - `src` image must be in [TRANSFER_SRC_OPTIMAL](vk::ImageLayout::TRANSFER_SRC_OPTIMAL) image layout
/// - `dst` image must be in [TRANSFER_DST_OPTIMAL](vk::ImageLayout::TRANSFER_DST_OPTIMAL) image layout
pub fn copy_image_to_image(device: &RenderingDevice, command_buffer: vk::CommandBuffer, src: vk::Image, dst: vk::Image, src_size: vk::Extent2D, dst_size: vk::Extent2D, filter: vk::Filter) {
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

/// Creates a [vk::ImageView] that covers all layers and layers
pub unsafe fn create_image_view(
    device: &RenderingDevice,
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

/// Creates a [vk::Sampler]
pub unsafe fn create_texture_sampler(
    device: &RenderingDevice,
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
        .border_color(address_mode.border_color)
        .unnormalized_coordinates(false)
        .compare_enable(true)
        .compare_op(vk::CompareOp::GREATER_OR_EQUAL)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(*mip_level as f32)
        .mip_lod_bias(0.0);

    let texture_sampler = device.create_sampler(&info, None)?;
    set_object_name(
        device,
        &format!("{} Texture Sampler", name),
        texture_sampler,
    )?;

    Ok(texture_sampler)
}

/// Determins a [sampler](vk::Sampler)'s min and mag [filter](vk::Filter)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    /// The sampler min filter
    pub min: vk::Filter,
    /// The sampler mag filter
    pub mag: vk::Filter,
}

impl Filter {
    /// min and mag Filters will be set to [LINEAR](vk::Filter::LINEAR)
    pub const LINEAR: Self = {
        Filter {
            min: vk::Filter::LINEAR,
            mag: vk::Filter::LINEAR,
        }
    };

    /// min and mag Filters will be set to [NEAREST](vk::Filter::NEAREST)
    pub const NEAREST: Self = {
        Filter {
            min: vk::Filter::NEAREST,
            mag: vk::Filter::NEAREST,
        }
    };

    /// min Filter will be [LINEAR](vk::Filter::LINEAR) and mag will be [NEAREST](vk::Filter::NEAREST)
    pub const MIN_LINEAR: Self = {
        Filter {
            min: vk::Filter::LINEAR,
            mag: vk::Filter::NEAREST,
        }
    };

    /// min Filter will be [NEAREST](vk::Filter::NEAREST) and mag will be [LINEAR](vk::Filter::LINEAR)
    pub const MAG_LINEAR: Self = {
        Filter {
            min: vk::Filter::NEAREST,
            mag: vk::Filter::LINEAR,
        }
    };
}

/// Determins a [sampler](vk::Sampler)'s u, v, and w [address mode](vk::SamplerAddressMode)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressMode {
    /// The sampler u sampler address mode
    pub u: vk::SamplerAddressMode,
    /// The sampler v sampler address mode
    pub v: vk::SamplerAddressMode,
    /// The sampler w sampler address mode
    pub w: vk::SamplerAddressMode,
    /// The sampler border color used in case one of the u, v, or w address modes is [CLAMP_TO_BORDER](vk::SamplerAddressMode::CLAMP_TO_BORDER)
    pub border_color: vk::BorderColor,
}

impl AddressMode {
    /// All directions will be [REPEAT](vk::SamplerAddressMode::REPEAT)
    pub const REPEAT: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::REPEAT,
            v: vk::SamplerAddressMode::REPEAT,
            w: vk::SamplerAddressMode::REPEAT,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        }
    };

    /// All directions will be [MIRRORED_REPEAT](vk::SamplerAddressMode::MIRRORED_REPEAT)
    pub const MIRRORED_REPEAT: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            w: vk::SamplerAddressMode::MIRRORED_REPEAT,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        }
    };

    /// All directions will be [MIRROR_CLAMP_TO_EDGE](vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE)
    pub const MIRROR_CLAMP_TO_EDGE: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            v: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            w: vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        }
    };
    
    /// All directions will be [CLAMP_TO_BORDER](vk::SamplerAddressMode::CLAMP_TO_BORDER) with the specified border color
    pub const fn clamp_to_border(border_color: vk::BorderColor) -> Self {
        AddressMode {
            u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
            border_color,
        }
    }
    
    /// All directions will be [CLAMP_TO_EDGE](vk::SamplerAddressMode::CLAMP_TO_EDGE)
    pub const CLAMP_TO_EDGE: Self = {
        AddressMode {
            u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        }
    };
}

unsafe fn generate_mipmaps(
    device: &RenderingDevice,
    data: &RendererData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // Support

    if !device.instance
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
        begin_single_time_commands(device, data, "Generate MipMaps")?;

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

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}
