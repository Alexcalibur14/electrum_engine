use vulkanalia::{prelude::v1_2::*, vk::KhrSwapchainExtension};

use anyhow::{Ok, Result};
use winit::window::Window;

use crate::{texture::{Image, MipLevels}, AppData, QueueFamilyIndices, SwapchainSupport};

use crate::texture::*;

#[derive(Debug, Clone)]
pub struct SubpassData {
    pub bind_point: vk::PipelineBindPoint,
    pub attachments: Vec<(u32, vk::ImageLayout, AttachmentType)>,
    pub dependencies: Vec<vk::SubpassDependency>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentType {
    InputAttachment,
    ColorAttachment,
    ResolveAttachment,
    DepthStencilAttachment,
    PreserveAttachment,
}

pub fn generate_render_pass(subpass_datas: &mut Vec<SubpassData>, attachments: &Vec<vk::AttachmentDescription>, device: &Device) -> Result<vk::RenderPass> {
    let mut dependencies = vec![];
    let mut subpasses = vec![];

    for subpass_data in subpass_datas {
        dependencies.append(&mut subpass_data.dependencies);

        let mut input_attachments = vec![];
        let mut color_attachments = vec![];
        let mut resolve_attachments = vec![];
        let mut depth_stencil_attachment = vk::AttachmentReference::default();
        let mut preserve_attachments = vec![];

        for attachment in &subpass_data.attachments {
            if attachment.2 == AttachmentType::PreserveAttachment {
                preserve_attachments.push(attachment.0);
                continue;
            }

            let attachment_ref = vk::AttachmentReference::builder()
                .attachment(attachment.0)
                .layout(attachment.1)
                .build();

            match attachment.2 {
                AttachmentType::InputAttachment => input_attachments.push(attachment_ref),
                AttachmentType::ColorAttachment => color_attachments.push(attachment_ref),
                AttachmentType::ResolveAttachment => resolve_attachments.push(attachment_ref),
                AttachmentType::DepthStencilAttachment => depth_stencil_attachment = attachment_ref,
                AttachmentType::PreserveAttachment => panic!("This should not happen"),
            }
        }

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(subpass_data.bind_point)
            .input_attachments(&input_attachments)
            .color_attachments(&color_attachments)
            .resolve_attachments(&resolve_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment)
            .preserve_attachments(&preserve_attachments)
            .build();

        subpasses.push(subpass);
    }

    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    Ok(unsafe { device.create_render_pass(&info, None) }?)
}

pub fn generate_render_pass_images(attachments: Vec<(vk::AttachmentDescription, vk::ImageUsageFlags, vk::ImageAspectFlags)>, data: &AppData, instance: &Instance, device: &Device) -> Vec<Image> {
    let mut images = vec![];
    
    let mut atta = attachments.clone();
    atta.pop();

    for attachment in atta {

        images.push(Image::new(
            data,
            instance,
            device,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            MipLevels::One,
            attachment.0.samples,
            attachment.0.format,
            vk::ImageTiling::OPTIMAL,
            attachment.1,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            attachment.2,
        ))
    }

    images
}

pub unsafe fn create_framebuffers(data: &AppData, device: &Device) -> Result<Vec<vk::Framebuffer>> {
    let framebuffers = data.swapchain_image_views.iter()
        .map(|i| {
            let mut views = data.images.iter().map(|i| i.view).collect::<Vec<_>>();
            views.push(*i);

            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(&views)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1)
                .build();

            device.create_framebuffer(&info, None)
        })
        .collect::<Result<Vec<_>, _>>().unwrap();

    Ok(framebuffers)
}


pub unsafe fn create_swapchain(window: &Window, instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Image

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count {
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

    let info = vk::SwapchainCreateInfoKHR::builder()
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
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;

    // Images

    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height,
            ))
            .build()
    }
}

pub unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| create_image_view(device, *i, data.swapchain_format, vk::ImageAspectFlags::COLOR, 1,))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

