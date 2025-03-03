#![allow(dead_code)]

use ash::khr::swapchain;
use ash::{vk, Entry};
use ash::{Device, Instance};

use anyhow::{Ok, Result};

use crate::{
    texture::{Image, MipLevels},
    QueueFamilyIndices, RendererData, SwapchainSupport,
};

use crate::{texture::*, Attachment, AttachmentSize};

pub fn generate_render_pass_images(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    attachments: &[(
        Attachment,
        vk::ImageUsageFlags,
        vk::ImageAspectFlags,
    )],
) -> Vec<Image> {
    let mut images = vec![];

    for attachment in attachments {
        let width = match attachment.0.x {
            AttachmentSize::Absolute(w) => w,
            AttachmentSize::Relative(w) => (data.swapchain_extent.width as f32 * w) as u32,
        };

        let height = match attachment.0.y {
            AttachmentSize::Absolute(h) => h,
            AttachmentSize::Relative(h) => (data.swapchain_extent.height as f32 * h) as u32,
        };

        let image = Image::new(
            instance,
            device,
            data,
            width,
            height,
            MipLevels::One,
            attachment.0.attachment_desc.samples,
            attachment.0.attachment_desc.format,
            vk::ImageTiling::OPTIMAL,
            attachment.1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            attachment.2,
            false,
        );
        
        images.push(image)
    }

    images
}

pub(crate) unsafe fn create_swapchain(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut RendererData,
    width: u32,
    height: u32,
) -> Result<()> {
    // Image

    let indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(entry, instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(width, height, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    // Create

    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(false)
        .old_swapchain(data.swapchain);

    let swapchain_loader = swapchain::Device::new(&instance, &device);

    data.swapchain = swapchain_loader.create_swapchain(&info, None)?;

    // Images

    data.swapchain_images = swapchain_loader.get_swapchain_images(data.swapchain)?;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(width: u32, height: u32, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::default()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                height,
            ))
    }
}

pub(crate) unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut RendererData,
) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                vk::ImageViewType::TYPE_2D,
                1,
                1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}
