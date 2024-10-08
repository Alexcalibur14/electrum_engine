use vulkanalia::prelude::v1_2::*;
use anyhow::{Ok, Result};

use crate::RendererData;

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

#[derive(Debug, Clone)]
pub struct SubpassData {
    pub bind_point: vk::PipelineBindPoint,
    pub attachments: Vec<(u32, vk::ImageLayout, AttachmentType)>,
}

pub fn generate_render_pass(
    device: &Device,
    subpass_datas: &[SubpassData],
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
            let attachment_ref = vk::AttachmentReference::builder()
                .attachment(attachment.0)
                .layout(attachment.1)
                .build();

            match attachment.2 {
                AttachmentType::Input => input_attachments.push(attachment_ref),
                AttachmentType::Color => color_attachments.push(attachment_ref),
                AttachmentType::Resolve => resolve_attachments.push(attachment_ref),
                AttachmentType::DepthStencil => depth_stencil_attachment = attachment_ref,
                AttachmentType::Preserve => preserve_attachments.push(attachment.0),
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


#[derive(Debug, Clone)]
pub struct SubPassRenderData {
    pub subpass_id: usize,
    pub objects: Vec<u64>,
    pub light_group: u64,

    pub command_buffers: Vec<vk::CommandBuffer>,
}

impl SubPassRenderData {
    pub fn new(id: usize, objects: Vec<u64>, light_group: u64) -> Self {
        SubPassRenderData {
            subpass_id: id,
            objects,
            light_group,
            command_buffers: vec![],
        }
    }

    pub fn setup_command_buffers(&mut self, device: &Device, data: &RendererData) {
        for i in 0..data.swapchain_images.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(data.command_pools[i])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            self.command_buffers.push(unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0])
        }
    }

    pub fn record_command_buffers(&mut self, instance: &Instance, device: &Device, data: &RendererData, image_index: usize) -> Result<()> {
        let command_buffer = self.command_buffers[image_index];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(data.render_pass)
            .subpass(self.subpass_id as u32)
            .framebuffer(data.framebuffers[image_index])
            .build();


        let begin_info = vk::CommandBufferBeginInfo::builder()
            .inheritance_info(&inheritance_info)
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .build();

        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }?;

        let light_group = data.light_groups.get(&self.light_group).unwrap();
        let other_descriptors = vec![(1, data.camera.get_descriptor_sets()[image_index]), (2, light_group.get_descriptor_sets()[image_index])];

        self.objects.iter().map(|k| data.objects.get(k).unwrap()).for_each(|o| unsafe { o.draw(instance, device, command_buffer, image_index, other_descriptors.clone()) });

        unsafe { device.end_command_buffer(command_buffer) }?;

        Ok(())
    }
}
