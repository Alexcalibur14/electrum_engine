use std::ptr;

use anyhow::{Ok, Result};

use ash::vk;
use ash::{Device, Instance};

use crate::{begin_command_label, end_command_label, generate_render_pass_images, get_c_ptr_slice, set_object_name, Image, Mesh, RenderStats, RendererData};

#[derive(Debug, Clone, Copy, Default)]
pub struct Attachment {
    pub x: AttachmentSize,
    pub y: AttachmentSize,

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
            sample_count: vk::SampleCountFlags::TYPE_1,
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
            sample_count: vk::SampleCountFlags::TYPE_1,
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
            sample_count: vk::SampleCountFlags::TYPE_1,
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
        self.attachment_desc = vk::AttachmentDescription::default()
            .flags(self.flags)
            .format(self.format)
            .samples(self.sample_count)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .stencil_load_op(self.stencil_load_op)
            .stencil_store_op(self.stencil_store_op)
            .initial_layout(self.initial_layout)
            .final_layout(self.final_layout)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttachmentSize {
    Relative(f32),
    Absolute(u32),
}

impl Default for AttachmentSize {
    fn default() -> Self {
        AttachmentSize::Relative(1.0)
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
pub struct SubpassLayoutBuilder {
    bind_point: vk::PipelineBindPoint,
    attachments: Vec<(AttachmentType, vk::ImageLayout)>,
}

impl SubpassLayoutBuilder {
    pub fn new(bind_point: vk::PipelineBindPoint) -> Self {
        SubpassLayoutBuilder {
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
pub struct SubpassDescriptionData {
    pub bind_point: vk::PipelineBindPoint,
    pub input_attachments: Vec<vk::AttachmentReference>,
    pub color_attachments: Vec<vk::AttachmentReference>,
    pub resolve_attachments: Vec<vk::AttachmentReference>,
    pub depth_stencil_attachment: vk::AttachmentReference,
    pub preserve_attachments: Vec<u32>,
}

impl SubpassDescriptionData {
    fn description(&self) -> vk::SubpassDescription {

        let depth_ptr = if self.depth_stencil_attachment.attachment == 0 && self.depth_stencil_attachment.layout == vk::ImageLayout::UNDEFINED {
            ptr::null()
        } else {
            &self.depth_stencil_attachment
        };

        vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: self.bind_point,
            input_attachment_count: self.input_attachments.len() as u32,
            p_input_attachments: get_c_ptr_slice(&self.input_attachments),
            color_attachment_count: self.color_attachments.len() as u32,
            p_color_attachments: get_c_ptr_slice(&self.color_attachments),
            p_resolve_attachments: get_c_ptr_slice(&self.resolve_attachments),
            p_depth_stencil_attachment: depth_ptr,
            preserve_attachment_count: self.preserve_attachments.len() as u32,
            p_preserve_attachments: get_c_ptr_slice(&self.preserve_attachments),
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Default)]
pub struct RenderPassBuilder {
    attachments: Vec<(Attachment, AttachmentUse, Option<usize>)>,
    subpasses: Vec<SubpassDescriptionData>,
    dependencies: Vec<vk::SubpassDependency>,

    attachment_images: Vec<Image>,
    pub clear_values: Vec<vk::ClearValue>,
    swapchain_output: bool,
    pub name: String,
}

impl RenderPassBuilder {
    pub fn new(swapchain_output: bool, name: String) -> Self {
        RenderPassBuilder {
            attachments: vec![],
            subpasses: vec![],
            dependencies: vec![],

            attachment_images: vec![],
            clear_values: vec![],

            swapchain_output,
            name,
        }
    }

    pub fn add_attachment(&mut self, attachment: Attachment, attachment_use: AttachmentUse, clear_value: vk::ClearValue) -> &mut Self {
        self.attachments.push((attachment, attachment_use, None));
        self.clear_values.push(clear_value);
        self
    }

    pub fn add_subpass(&mut self, subpass: &SubpassLayoutBuilder, attachment_indices: &[(u32, Option<vk::SubpassDependency>)]) -> &mut Self {
        let mut description_data = SubpassDescriptionData {
            bind_point: subpass.bind_point,
            input_attachments: vec![],
            color_attachments: vec![],
            resolve_attachments: vec![],
            depth_stencil_attachment: vk::AttachmentReference::default(),
            preserve_attachments: vec![],
        };

        for ((attachment_type, layout), (index, dependency)) in subpass.attachments.iter().zip(attachment_indices) {
            let attachment_ref = vk::AttachmentReference::default()
                .attachment(*index)
                .layout(*layout);

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

        let attachment_descriptions = self.attachments.iter().map(|(attachment, _, _)| attachment.attachment_desc).collect::<Vec<vk::AttachmentDescription>>();

        let create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachment_descriptions.len() as u32,
            p_attachments: get_c_ptr_slice(&attachment_descriptions),
            subpass_count: subpass_descriptors.len() as u32,
            p_subpasses: get_c_ptr_slice(&subpass_descriptors),
            dependency_count: self.dependencies.len() as u32,
            p_dependencies: get_c_ptr_slice(&self.dependencies),
            _marker: std::marker::PhantomData,
        };

        let render_pass = unsafe { device.create_render_pass(&create_info, None) }?;
        set_object_name(instance, device, &format!("{} Render Pass", self.name), render_pass).unwrap();

        let image_attachments = self.attachments.iter().map(|(attachment, a_use, _)| (*attachment, a_use.get_usage_flags(), a_use.get_aspect_flags())).collect::<Vec<_>>();
        // the last attachment must be the swapchain image
        let attachment_images = if self.swapchain_output{
            generate_render_pass_images(instance, device, data, &image_attachments[..(image_attachments.len() - 1)])
        } else {
            generate_render_pass_images(instance, device, data, &image_attachments)
        };

        let framebuffers = data
            .swapchain_image_views
            .iter()
            .enumerate()
            .map(|(index, i)| {
                let mut views = attachment_images.iter().map(|i| i.view).collect::<Vec<_>>();
                
                if self.swapchain_output{
                    views.push(*i);
                }

                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&views)
                    .width(data.swapchain_extent.width)
                    .height(data.swapchain_extent.height)
                    .layers(1);

                let framebuffer = unsafe { device.create_framebuffer(&info, None) };
                set_object_name(instance, device, &format!("{} Framebuffer({})", self.name.clone(), index), framebuffer.unwrap()).unwrap();
                framebuffer
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.attachment_images = attachment_images;
        
        Ok((render_pass, framebuffers))
    }

    pub fn destroy_swapchain(&self, device: &Device) {
        self.attachment_images.iter().for_each(|a| a.destroy(device));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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


#[derive(Debug, Clone)]
pub struct SubpassRenderData {
    pub render_pass_id: u32,
    pub subpass_id: u32,
    pub objects: Vec<(usize, usize)>,
    pub camera: usize,

    pub command_buffers: Vec<vk::CommandBuffer>,
    pub name: String,
}

impl SubpassRenderData {
    pub fn new(render_pass_id: u32, id: u32, objects: Vec<(usize, usize)>, camera: usize, name: String) -> Self {
        SubpassRenderData {
            render_pass_id,
            subpass_id: id,
            objects,
            camera,
            command_buffers: vec![],
            name,
        }
    }

    pub fn setup_command_buffers(&mut self, device: &Device, data: &RendererData) {
        for i in 0..data.swapchain_images.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
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
        render_pass: vk::RenderPass,
        framebuffers: vk::Framebuffer,
        image_index: usize,
    ) -> Result<()> {
        let command_buffer = self.command_buffers[image_index];

        let inheritance_info = vk::CommandBufferInheritanceInfo::default()
            .render_pass(render_pass)
            .subpass(self.subpass_id as u32)
            .framebuffer(framebuffers);

        let begin_info = vk::CommandBufferBeginInfo::default()
            .inheritance_info(&inheritance_info)
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE);

        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }?;

        let camera = data.cameras.get_loaded(self.camera).unwrap();
        let mut other_descriptors = vec![
            camera.get_descriptor_sets()[image_index],
        ];

        other_descriptors.append(&mut data.other_descriptors.iter().map(|d| d.descriptor_sets()[image_index]).collect::<Vec<_>>());

        begin_command_label(instance, device, command_buffer, &self.name, [1.0, 1.0, 0.0, 1.0]);

        self.objects
            .iter()
            .map(|(k_o, k_m)| (data.objects.get_loaded(*k_o).unwrap(), data.materials.get_loaded(*k_m).unwrap()))
            .for_each(|(o, m)| {
                let scene_descriptors = m.get_scene_descriptor_ids().iter().map(|id| other_descriptors[*id]).collect::<Vec<_>>();

                m.draw(
                    instance,
                    device,
                    command_buffer,
                    o.descriptor_set(self.render_pass_id, self.subpass_id, image_index),
                    scene_descriptors,
                    &o.mesh_data,
                    &o.name,
                )
            });
        
        end_command_label(instance, device, command_buffer);

        unsafe { device.end_command_buffer(command_buffer) }?;

        Ok(())
    }

    pub fn destroy_swapchain(&self, device: &Device, data: &RendererData) {
        data.materials.get_loaded(self.objects[0].1).unwrap().destroy_swapchain(device);
    }

    pub fn recreate_swapchain(&self, instance: &Instance, device: &Device, data: &mut RendererData, render_pass: vk::RenderPass) {
        let mut mat = data.materials.get_mut_loaded(self.objects[0].1).unwrap().clone();
        mat.recreate_swapchain(instance, device, data, render_pass);
        data.materials[self.objects[0].1] = mat;
    }

    pub fn update(
        &self,
        device: &Device,
        data: &mut RendererData,
        stats: &RenderStats,
        image_index: usize,
    ) {
        let camera = data.cameras.get_mut_loaded(self.camera).unwrap();
        camera.calculate_view(device, image_index);

        let objects = self
            .objects
            .iter()
            .map(|(k, _)| (*k, data.objects.get_loaded(*k).unwrap().clone()))
            .collect::<Vec<(usize, Mesh)>>();

        objects
            .into_iter()
            .for_each(|(id, mut o)| {
                o.update(device, data, stats, image_index);
                data.objects[id] = o;
            });
    }
}


#[derive(Clone)]
pub struct RenderPass {
    pub name: String,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub subpasses: Vec<SubpassRenderData>,
    pub clear_values: Vec<vk::ClearValue>,
    pub dependencies: Vec<vk::SubpassDependency>,
    pub subpass_data: Vec<SubpassDescriptionData>,
    pub swapchain_output: bool,
    
    pub width: u32,
    pub height: u32,
    attachments: Vec<(Attachment, AttachmentUse)>,
    attachment_images: Vec<Image>,
}

impl RenderPass {
    pub fn new(name: &str, subpasses: Vec<SubpassRenderData>, subpass_data: Vec<SubpassDescriptionData>, clear_values: Vec<vk::ClearValue>, dependencies: Vec<vk::SubpassDependency>, attachments: Vec<(Attachment, AttachmentUse)>, swapchain_output: bool) -> Self {
        RenderPass {
            name: name.into(),
            render_pass: vk::RenderPass::null(),
            framebuffers: vec![],
            subpasses,
            clear_values,
            dependencies,
            subpass_data,
            swapchain_output,
            attachments,
            attachment_images: vec![],
            width: 0,
            height: 0,
        }
    }

    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &mut RendererData) -> Result<()> {
        let subpass_descriptors = self.subpass_data.iter().map(|s| s.description()).collect::<Vec<vk::SubpassDescription>>();

        let attachment_descriptions = self.attachments.iter().map(|(attachment, _)| attachment.attachment_desc).collect::<Vec<vk::AttachmentDescription>>();

        self.width = self.attachments.iter().map(|(a, _)| match a.x {
            AttachmentSize::Relative(r) => (r * data.swapchain_extent.width as f32) as u32,
            AttachmentSize::Absolute(a) => a,
        }).max().unwrap();

        self.height = self.attachments.iter().map(|(a, _)| match a.y {
            AttachmentSize::Relative(r) => (r * data.swapchain_extent.height as f32) as u32,
            AttachmentSize::Absolute(a) => a,
        }).max().unwrap();

        let create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachment_descriptions.len() as u32,
            p_attachments: get_c_ptr_slice(&attachment_descriptions),
            subpass_count: subpass_descriptors.len() as u32,
            p_subpasses: get_c_ptr_slice(&subpass_descriptors),
            dependency_count: self.dependencies.len() as u32,
            p_dependencies: get_c_ptr_slice(&self.dependencies),
            _marker: std::marker::PhantomData,
        };

        let render_pass = unsafe { device.create_render_pass(&create_info, None) }?;
        set_object_name(instance, device, &self.name, render_pass).unwrap();

        let image_attachments = self.attachments.iter().map(|(attachment, a_use)| (*attachment, a_use.get_usage_flags(), a_use.get_aspect_flags())).collect::<Vec<_>>();
        // the last attachment must be the swapchain image
        let attachment_images = if self.swapchain_output{
            generate_render_pass_images(instance, device, data, &image_attachments[..(image_attachments.len() - 1)])
        } else {
            generate_render_pass_images(instance, device, data, &image_attachments)
        };

        let framebuffers = data
            .swapchain_image_views
            .iter()
            .map(|i| {
                let mut views = attachment_images.iter().map(|i| i.view).collect::<Vec<_>>();

                if self.swapchain_output {
                    views.push(*i);
                }

                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&views)
                    .width(self.width)
                    .height(self.height)
                    .layers(1);

                let framebuffer = unsafe { device.create_framebuffer(&info, None) };
                set_object_name(instance, device, &(self.name.clone() + "Framebuffer"), framebuffer.unwrap()).unwrap();
                framebuffer
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.framebuffers = framebuffers;
        self.render_pass = render_pass;
        self.attachment_images = attachment_images;

        self.subpasses.iter_mut().for_each(|s| s.recreate_swapchain(instance, device, data, self.render_pass));

        Ok(())
    }

    pub fn destroy_swapchain(&self, device: &Device, data: &mut RendererData) {
        self.attachment_images.iter().for_each(|a| a.destroy(device));
        self.framebuffers.iter().for_each(|f| unsafe { device.destroy_framebuffer(*f, None) });
        unsafe { device.destroy_render_pass(self.render_pass, None) };
        self.subpasses.iter().for_each(|s| s.destroy_swapchain(device, data));
    }
}
