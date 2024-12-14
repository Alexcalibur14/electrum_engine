use std::ptr;

use anyhow::{Ok, Result};
use vulkanalia::prelude::v1_2::*;

use crate::{generate_render_pass_images, get_c_ptr, get_c_ptr_slice, Image, RenderStats, Renderable, RendererData};

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
    bind_point: vk::PipelineBindPoint,
    input_attachments: Vec<vk::AttachmentReference>,
    color_attachments: Vec<vk::AttachmentReference>,
    resolve_attachments: Vec<vk::AttachmentReference>,
    depth_stencil_attachment: vk::AttachmentReference,
    preserve_attachments: Vec<u32>,
}

impl SubpassData {
    pub fn new(
        bind_point: vk::PipelineBindPoint,
        attachments: Vec<(u32, vk::ImageLayout, AttachmentType)>,
    ) -> Self {
        let mut input_attachments = vec![];
        let mut color_attachments = vec![];
        let mut resolve_attachments = vec![];
        let mut depth_stencil_attachment = vk::AttachmentReference::default();
        let mut preserve_attachments = vec![];

        for attachment in attachments {
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

        SubpassData {
            bind_point,
            input_attachments,
            color_attachments,
            resolve_attachments,
            depth_stencil_attachment,
            preserve_attachments,
        }
    }

    pub fn description(&self) -> vk::SubpassDescription {
        vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: self.bind_point,
            input_attachment_count: self.input_attachments.len() as u32,
            input_attachments: get_c_ptr_slice(&self.input_attachments),
            color_attachment_count: self.color_attachments.len() as u32,
            color_attachments: get_c_ptr_slice(&self.color_attachments),
            resolve_attachments: get_c_ptr_slice(&self.resolve_attachments),
            depth_stencil_attachment: get_c_ptr(&self.depth_stencil_attachment),
            preserve_attachment_count: self.preserve_attachments.len() as u32,
            preserve_attachments: get_c_ptr_slice(&self.preserve_attachments),
        }
    }
}

pub fn generate_render_pass(
    device: &Device,
    subpass_datas: &[SubpassData],
    attachments: &[vk::AttachmentDescription],
    dependencies: &[vk::SubpassDependency],
) -> Result<vk::RenderPass> {
    let mut subpasses = vec![];

    for subpass_data in subpass_datas {
        subpasses.push(subpass_data.description());
    }

    let info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        next: ptr::null(),
        flags: vk::RenderPassCreateFlags::empty(),
        attachment_count: attachments.len() as u32,
        attachments: get_c_ptr_slice(attachments),
        subpass_count: subpasses.len() as u32,
        subpasses: get_c_ptr_slice(&subpasses),
        dependency_count: dependencies.len() as u32,
        dependencies: get_c_ptr_slice(dependencies),
    };

    Ok(unsafe { device.create_render_pass(&info, None) }?)
}

#[derive(Debug, Clone)]
pub struct SubPassRenderData {
    pub subpass_id: usize,
    pub objects: Vec<(usize, usize)>,
    pub light_group: usize,
    pub camera: usize,

    pub command_buffers: Vec<vk::CommandBuffer>,
}

impl SubPassRenderData {
    pub fn new(id: usize, objects: Vec<(usize, usize)>, camera: usize, light_group: usize) -> Self {
        SubPassRenderData {
            subpass_id: id,
            objects,
            light_group,
            camera,
            command_buffers: vec![],
        }
    }

    pub fn setup_command_buffers(&mut self, device: &Device, data: &RendererData) {
        for i in 0..data.swapchain_images.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(data.command_pools[i])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            self.command_buffers
                .push(unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0])
        }
    }

    pub fn record_command_buffers(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        image_index: usize,
    ) -> Result<()> {
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

        let light_group = data.light_groups.get(self.light_group).unwrap();
        let camera = data.cameras.get(self.camera).unwrap();
        let other_descriptors = vec![
            (1, camera.get_descriptor_sets()[image_index]),
            (2, light_group.get_descriptor_sets()[image_index]),
        ];

        self.objects
            .iter()
            .map(|(k_o, k_m)| (data.objects.get(*k_o).unwrap(), data.materials.get(*k_m).unwrap()))
            .for_each(|(o, m)| {
                m.draw(
                    instance,
                    device,
                    command_buffer,
                    o.descriptor_set(image_index),
                    other_descriptors.clone(),
                    o.mesh_data(),
                    &o.name(),
                )
            });

        unsafe { device.end_command_buffer(command_buffer) }?;

        Ok(())
    }

    pub fn destroy_swapchain(&self, device: &Device, data: &RendererData) {
        data.materials.get(self.objects[0].1).unwrap().destroy_swapchain(device);
    }

    pub fn recreate_swapchain(&self, instance: &Instance, device: &Device, data: &mut RendererData) {
        let mut mat = data.materials.get_mut(self.objects[0].1).unwrap().clone();
        mat.recreate_swapchain(instance, device, data);
        data.materials.replace(self.objects[0].1, mat);
    }

    pub fn update(
        &self,
        device: &Device,
        data: &mut RendererData,
        stats: &RenderStats,
        image_index: usize,
    ) {
        let camera = data.cameras.get_mut(self.camera).unwrap();
        camera.calculate_view(device, image_index);

        let vp = camera.proj() * camera.view();

        let objects = self
            .objects
            .iter()
            .map(|(k, _)| (*k, data.objects.get(*k).unwrap().clone()))
            .collect::<Vec<(usize, Box<dyn Renderable>)>>();

        objects
            .into_iter()
            .for_each(|(id, mut o)| {
                o.update(device, data, stats, image_index, vp);
                data.objects.replace(id, o);
            });
    }
}


#[derive(Debug, Clone)]
pub struct SubpassBuilder {
    bind_point: vk::PipelineBindPoint,
    attachments: Vec<(AttachmentType, vk::ImageLayout)>,
}

impl SubpassBuilder {
    pub fn new(bind_point: vk::PipelineBindPoint) -> Self {
        SubpassBuilder {
            bind_point,
            attachments: vec![],
        }
    }

    pub fn add_attachment(&mut self, attachment_type: AttachmentType, layout: vk::ImageLayout) -> &mut Self {
        self.attachments.push((attachment_type, layout));
        self
    }

    pub fn add_attachments(&mut self, attachments: &[(AttachmentType, vk::ImageLayout)]) -> &mut Self {
        self.attachments.append(&mut attachments.into());
        self
    }

    pub fn add_input_attachment(&mut self, layout: vk::ImageLayout) -> &mut Self {
        self.attachments.push((AttachmentType::Input, layout));
        self
    }

    pub fn add_color_attachment(&mut self, layout: vk::ImageLayout) -> &mut Self {
        self.attachments.push((AttachmentType::Color, layout));
        self
    }

    pub fn add_resolve_attachment(&mut self, layout: vk::ImageLayout) -> &mut Self {
        self.attachments.push((AttachmentType::Resolve, layout));
        self
    }

    pub fn add_depth_stencil_attachment(&mut self, layout: vk::ImageLayout) -> &mut Self {
        self.attachments.push((AttachmentType::DepthStencil, layout));
        self
    }
    
    pub fn add_preserve_attachment(&mut self) -> &mut Self {
        self.attachments.push((AttachmentType::DepthStencil, vk::ImageLayout::UNDEFINED));
        self
    }

    pub fn build(&mut self) -> Self {
        self.clone()
    }
}

#[derive(Debug, Clone)]
struct SubpassDescriptionData {
    bind_point: vk::PipelineBindPoint,
    input_attachments: Vec<vk::AttachmentReference>,
    color_attachments: Vec<vk::AttachmentReference>,
    resolve_attachments: Vec<vk::AttachmentReference>,
    depth_stencil_attachment: vk::AttachmentReference,
    preserve_attachments: Vec<u32>,
}

impl SubpassDescriptionData {
    fn description(&self) -> vk::SubpassDescription {
        vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: self.bind_point,
            input_attachment_count: self.input_attachments.len() as u32,
            input_attachments: get_c_ptr_slice(&self.input_attachments),
            color_attachment_count: self.color_attachments.len() as u32,
            color_attachments: get_c_ptr_slice(&self.color_attachments),
            resolve_attachments: get_c_ptr_slice(&self.resolve_attachments),
            depth_stencil_attachment: get_c_ptr(&self.depth_stencil_attachment),
            preserve_attachment_count: self.preserve_attachments.len() as u32,
            preserve_attachments: get_c_ptr_slice(&self.preserve_attachments),
        }
    }
}

#[derive(Clone, Default)]
pub struct RenderPassBuilder {
    attachments: Vec<(vk::AttachmentDescription, AttachmentUse, Option<usize>)>,
    subpasses: Vec<SubpassDescriptionData>,
    dependencies: Vec<vk::SubpassDependency>,

    attachment_images: Vec<Image>,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        RenderPassBuilder {
            attachments: vec![],
            subpasses: vec![],
            dependencies: vec![],

            attachment_images: vec![],
        }
    }

    pub fn add_attachment(&mut self, attachment: vk::AttachmentDescription, attachment_use: AttachmentUse) -> &mut Self {
        self.attachments.push((attachment, attachment_use, None));
        self
    }

    pub fn add_subpass(&mut self, subpass: &SubpassBuilder, attachment_indices: &[(u32, Option<vk::SubpassDependency>)]) -> &mut Self {
        let mut description_data = SubpassDescriptionData {
            bind_point: subpass.bind_point,
            input_attachments: vec![],
            color_attachments: vec![],
            resolve_attachments: vec![],
            depth_stencil_attachment: vk::AttachmentReference::default(),
            preserve_attachments: vec![],
        };

        for ((attachment_type, layout), (index, dependency)) in subpass.attachments.iter().zip(attachment_indices) {
            let attachment_ref = vk::AttachmentReference::builder()
                .attachment(*index)
                .layout(*layout)
                .build();

            match attachment_type {
                AttachmentType::Input => {
                    description_data.input_attachments.push(attachment_ref);
                },
                AttachmentType::Color => {
                    description_data.color_attachments.push(attachment_ref);
                },
                AttachmentType::Resolve => {
                    description_data.resolve_attachments.push(attachment_ref);
                },
                AttachmentType::DepthStencil => {
                    description_data.depth_stencil_attachment = attachment_ref
                },
                AttachmentType::Preserve => {
                    description_data.preserve_attachments.push(*index);
                },
            }

            match dependency {
                Some(dependency) => {
                    self.dependencies.push(*dependency);
                    self.attachments[*index as usize].2 = Some(self.subpasses.len());
                },
                None => {},
            }
        }

        self.subpasses.push(description_data);

        self
    }

    pub fn build(&mut self) -> Self {
        self.clone()
    }

    pub fn create_render_pass(&mut self, instance: &Instance, device: &Device, data: &RendererData) -> Result<(vk::RenderPass, Vec<vk::Framebuffer>)> {
        let subpass_descriptors = self.subpasses.iter().map(|s| s.description()).collect::<Vec<vk::SubpassDescription>>();

        let attachment_descriptions = self.attachments.iter().map(|(description, _, _)| *description).collect::<Vec<vk::AttachmentDescription>>();

        let create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachment_descriptions.len() as u32,
            attachments: get_c_ptr_slice(&attachment_descriptions),
            subpass_count: subpass_descriptors.len() as u32,
            subpasses: get_c_ptr_slice(&subpass_descriptors),
            dependency_count: self.dependencies.len() as u32,
            dependencies: get_c_ptr_slice(&self.dependencies),
        };

        let render_pass = unsafe { device.create_render_pass(&create_info, None) }?;

        let image_attachments = self.attachments.iter().map(|(description, a_use, _)| (*description, a_use.get_usage_flags(), a_use.get_aspect_flags())).collect::<Vec<_>>();
        let attachment_images = generate_render_pass_images(instance, device, data, &image_attachments[0..(image_attachments.len() - 1)]);

        let framebuffers = data
            .swapchain_image_views
            .iter()
            .map(|i| {
                let mut views = attachment_images.iter().map(|i| i.view).collect::<Vec<_>>();
                views.push(*i);

                let info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&views)
                    .width(data.swapchain_extent.width)
                    .height(data.swapchain_extent.height)
                    .layers(1)
                    .build();

                unsafe { device.create_framebuffer(&info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.attachment_images = attachment_images;
        
        Ok((render_pass, framebuffers))
    }

    pub fn destroy_swapchain(&self, device: &Device) {
        self.attachment_images.iter().for_each(|a| a.destroy(device));
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AttachmentUse {
    Color,
    Depth,
    DepthStencil,
}

impl AttachmentUse {
    pub fn get_usage_flags(&self) -> vk::ImageUsageFlags {
        match self {
            AttachmentUse::Color => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AttachmentUse::Depth => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            AttachmentUse::DepthStencil => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        }
    }

    pub fn get_aspect_flags(&self) -> vk::ImageAspectFlags {
        match self {
            AttachmentUse::Color => vk::ImageAspectFlags::COLOR,
            AttachmentUse::Depth => vk::ImageAspectFlags::DEPTH,
            AttachmentUse::DepthStencil => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        }
    }
}
