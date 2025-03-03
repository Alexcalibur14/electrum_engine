#![allow(clippy::too_many_arguments)]

use std::io::Read;

use ash::vk;
use ash::{Device, Instance};

use anyhow::{anyhow, Result};
use image::ImageReader;

use crate::{
    buffer::{
        begin_single_time_commands, create_buffer, end_single_time_commands, get_memory_type_index,
    }, set_object_name, Loadable, RendererData
};

#[allow(dead_code)]
pub enum MipLevels {
    One,
    Value(u32),
    Maximum,
}

#[derive(Debug, Clone, Default)]
pub struct Image {
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub mip_level: u32,
    pub sampler: Option<vk::Sampler>,
    loaded: bool,
}

impl Image {
    pub fn new(
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
        generate_sampler: bool,
    ) -> Self {
        let mips = match mip_levels {
            MipLevels::One => 1,
            MipLevels::Value(v) => v,
            MipLevels::Maximum => (width.max(height) as f32).log2().floor() as u32 + 1,
        };

        let (image, image_memory) = unsafe {
            create_image(
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

        let view = unsafe { create_image_view(
            device,
            image,
            format,
            aspects,
            view_type,
            mips,
            layer_count,
        ) }.unwrap();

        let sampler = if generate_sampler {
            Some(
                unsafe { create_texture_sampler(instance, device, &mips, "") }.unwrap()
            )
        } else {
            None
        };

        Image {
            image,
            image_memory,
            view,
            mip_level: mips,
            sampler,
            loaded: true,
        }
    }

    pub fn from_path(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        path: &str,
        mip_levels: MipLevels,
        format: vk::Format,
        generate_sampler: bool,
    ) -> Self {
        let img = ImageReader::open(path).unwrap().decode().unwrap();
        let bytes = img
            .to_rgba8()
            .bytes()
            .filter(|b| b.is_ok())
            .flatten()
            .collect::<Vec<u8>>();

        let width = img.width();
        let height = img.height();
        let size = (width * height * 8 * 4) as u64;

        let mips = match mip_levels {
            MipLevels::One => 1,
            MipLevels::Value(v) => v,
            MipLevels::Maximum => (width.max(height) as f32).log2().floor() as u32 + 1,
        };

        let (image, image_memory) = unsafe {
            create_texture_image(
                instance, device, data, size, &bytes, width, height, mips, format,
            )
        }
        .unwrap();

        let view =
            unsafe { create_image_view(
                device,
                image,
                format,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                mips,
                1,
            ) }.unwrap();

        let sampler = if generate_sampler {
            Some(
                unsafe { create_texture_sampler(instance, device, &mips, "") }.unwrap()
            )
        } else {
            None
        };

        Image {
            image,
            image_memory,
            view,
            mip_level: mips,
            sampler,
            loaded: true,
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            if let Some(sampler) = self.sampler {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.image_memory, None);
        }
    }
}

impl Loadable for Image {
    fn is_loaded(&self) -> bool {
        self.loaded
    }
}

pub unsafe fn create_image(
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
    let staging_buffer = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        "Image staging buffer",
    )?;

    // Copy (staging)

    staging_buffer.copy_vec_into_buffer(device, pixels);

    // Create (image)

    let (texture_image, texture_image_memory) = create_image(
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
        staging_buffer.buffer,
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
    name: &str,
) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
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
            "Texture image format does not support linear blitting!"
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
