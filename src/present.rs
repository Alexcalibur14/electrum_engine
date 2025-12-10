use ash::khr::swapchain;
use ash::vk;
use ash::{Entry, Instance, Device};

use anyhow::Result;

use crate::{create_image_view, QueueFamilyIndices, RendererData, SwapchainSupport};


pub(crate) unsafe fn create_swapchain(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut RendererData,
    width: u32,
    height: u32,
) -> Result<u32> {
    // Image

    let indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(entry, instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = vk::Extent2D {
        width,
        height,
    };

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;
    data.swapchain_rect = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };

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
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
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

    Ok(image_count)
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

pub fn transition_image(device: &Device, command_buffer: vk::CommandBuffer, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) {
    let (src_stage_mask, dst_stage_mask) = if new_layout == vk::ImageLayout::PRESENT_SRC_KHR {
        (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
    } else {
        (vk::PipelineStageFlags2::ALL_COMMANDS, vk::PipelineStageFlags2::ALL_COMMANDS)
    };

    let image_barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .dst_stage_mask(dst_stage_mask)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_access_mask(
            if new_layout == vk::ImageLayout::PRESENT_SRC_KHR {
                vk::AccessFlags2::MEMORY_READ
            } else {
                vk::AccessFlags2::MEMORY_WRITE
            }
        )

        .old_layout(old_layout)
        .new_layout(new_layout)

        .subresource_range(
            image_subresource_range(if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {vk::ImageAspectFlags::DEPTH} else {vk::ImageAspectFlags::COLOR})
        )
        .image(image)];

    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(&image_barrier);

    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
}

pub fn image_subresource_range(image_aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(image_aspect)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn semaphore_submit_info<'a>(stage_mask: vk::PipelineStageFlags2, semaphore: &'a vk::Semaphore) -> vk::SemaphoreSubmitInfo<'a> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(*semaphore)
        .stage_mask(stage_mask)
        .device_index(0)
        .value(1)
}

pub fn command_buffer_submit_info<'a>(command_buffer: &'a vk::CommandBuffer) -> vk::CommandBufferSubmitInfo<'a> {
    vk::CommandBufferSubmitInfo::default()
        .command_buffer(*command_buffer)
        .device_mask(0)
}

pub fn submit_info<'a>(
    command_info: &'a[vk::CommandBufferSubmitInfo],
    signal_semaphore_info: &'a[vk::SemaphoreSubmitInfo],
    wait_semaphore_info: &'a[vk::SemaphoreSubmitInfo]
) -> vk::SubmitInfo2<'a> {
    vk::SubmitInfo2::default()
        .wait_semaphore_infos(wait_semaphore_info)
        .signal_semaphore_infos(signal_semaphore_info)
        .command_buffer_infos(command_info)
}
