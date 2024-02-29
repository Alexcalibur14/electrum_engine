use vulkanalia::prelude::v1_2::*;

use anyhow::{Ok, Result};

use crate::{texture::{Image, MipLevels}, AppData};

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
                AttachmentType::PreserveAttachment => panic!(),
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
