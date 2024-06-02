use vulkanalia::prelude::v1_2::*;
use anyhow::{Ok, Result};

#[derive(Debug, Clone, Copy, Default)]
pub struct Attachment {
    pub flags: vk::AttachmentDescriptionFlags,
    pub format: vk::Format,
    pub sample_count: vk::SampleCountFlags,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,

    pub attachment_desc: vk::AttachmentDescription,
}

impl Attachment {
    pub fn template_colour() -> Self {
        Attachment {
            format: vk::Format::R32G32B32A32_SFLOAT,
            sample_count: vk::SampleCountFlags::_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        }
    }

    pub fn template_depth() -> Self {
        Attachment {
            format: vk::Format::D32_SFLOAT,
            sample_count: vk::SampleCountFlags::_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        }
    }

    pub fn template_present() -> Self {
        Attachment {
            format: vk::Format::R8G8B8A8_SRGB,
            sample_count: vk::SampleCountFlags::_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        }
    }

    pub fn generate(&mut self) {
        self.attachment_desc = vk::AttachmentDescription::builder()
            .flags(self.flags)
            .format(self.format)
            .samples(self.sample_count)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .stencil_load_op(self.stencil_load_op)
            .stencil_store_op(self.stencil_store_op)
            .initial_layout(self.initial_layout)
            .final_layout(self.final_layout)
            .build();
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentType {
    Input,
    Color,
    Resolve,
    DepthStencil,
    Preserve,
}

pub struct SubpassData {
    pub bind_point: vk::PipelineBindPoint,
    pub attachments: Vec<(u32, vk::ImageLayout, AttachmentType)>,
}

pub fn generate_render_pass(
    device: &Device,
    subpass_datas: &mut Vec<SubpassData>,
    attachments: &[vk::AttachmentDescription],
    dependencies: Vec<vk::SubpassDependency>,
) -> Result<vk::RenderPass> {
    let mut subpasses = vec![];

    for subpass_data in subpass_datas {

        let mut input_attachments = vec![];
        let mut color_attachments = vec![];
        let mut resolve_attachments = vec![];
        let mut depth_stencil_attachment = vk::AttachmentReference::default();
        let mut preserve_attachments = vec![];

        for attachment in &subpass_data.attachments {
            if attachment.2 == AttachmentType::Preserve {
                preserve_attachments.push(attachment.0);
                continue;
            }

            let attachment_ref = vk::AttachmentReference::builder()
                .attachment(attachment.0)
                .layout(attachment.1)
                .build();

            match attachment.2 {
                AttachmentType::Input => input_attachments.push(attachment_ref),
                AttachmentType::Color => color_attachments.push(attachment_ref),
                AttachmentType::Resolve => resolve_attachments.push(attachment_ref),
                AttachmentType::DepthStencil => depth_stencil_attachment = attachment_ref,
                AttachmentType::Preserve => panic!("This should not happen"),
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
        .attachments(attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    Ok(unsafe { device.create_render_pass(&info, None) }?)
}
