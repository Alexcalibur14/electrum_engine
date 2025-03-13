use std::collections::HashMap;
use std::ptr;

use anyhow::Result;

use ash::vk;
use ash::{Device, Instance};
use thiserror::Error;

use crate::{begin_command_label, end_command_label, get_c_ptr_slice, set_object_name, Image, Mesh, MipLevels, RenderStats, RendererData};

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
    pub fn new(render_pass_id: u32, id: u32, objects: Vec<(usize, usize)>, camera: usize, name: &str) -> Self {
        SubpassRenderData {
            render_pass_id,
            subpass_id: id,
            objects,
            camera,
            command_buffers: vec![],
            name: name.into(),
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
pub struct FrameGraph {
    pub name: String,
    pub passes: Vec<Pass>,
    pub resources: Vec<Resource>,
}

impl FrameGraph {
    pub fn new(name: &str) -> Self {
        FrameGraph {
            name: name.into(),
            passes: vec![],
            resources: vec![],
        }
    }

    pub fn add_pass(&mut self, pass: Pass) -> &mut Self {
        self.passes.push(pass);
        self
    }

    pub fn get_attachment(&self, name: &str) -> Resource {
        let mut name = name.to_string();
        loop {
            let resource = self.resources.iter().find(|r| r.name == name).unwrap();
            match resource.resource.clone() {
                ResourceType::Attachment { .. } => return resource.clone(),
                ResourceType::Transitioned { attachment, .. } => name = attachment.clone(),
                ResourceType::Texture { attachment, .. } => name = attachment.clone(),
            }
        }
    }

    pub fn validate(&mut self) -> Result<(), FrameGraphError> {
        for pass in &self.passes {
            for resource in &pass.created_resources {
                if self.resources.iter().find(|r| r.name == resource.name).is_none() {
                    self.resources.push(resource.clone());
                }
            }

            for resource_ref in &pass.resource_refs {
                if self.resources.iter().find(|r| r.name == resource_ref.name).is_none() {
                    return Err(FrameGraphError::ResourceNotCreated(resource_ref.name.clone()));
                }
            }
        }

        Ok(())
    }

    pub fn generate_render_pass(&mut self, instance: &Instance, device: &Device, data: &mut RendererData) -> Vec<RenderPass> {
        self.validate().unwrap();

        let mut passes = self.passes.clone();

        let mut previous_uses = vec![None; self.resources.len()];
        let mut preserves = vec![vec![]; passes.len()];

        for (pass_id, pass) in passes.iter_mut().enumerate() {
            for resource_ref in pass.resource_refs.iter() {
                let usage = resource_ref.usage;
                let resource = self.resources.iter().find(|r| r.name == resource_ref.name).unwrap();

                let resource_id = self.resources.iter().position(|r| r.name == resource_ref.name).unwrap();
                match previous_uses[resource_id] {
                    Some(previous_use) => {
                        let difference = pass_id - previous_use;
                        if difference > 1 {
                            for i in previous_use..pass_id {
                                preserves[i].push(resource_id as u32);
                            }
                        }
                    },
                    None => {
                        previous_uses[resource_id] = Some(pass_id);
                    },
                }

                match resource.resource.clone() {
                    ResourceType::Attachment { samples, .. } => {
                        if usage == ResourceUsage::Read {
                            continue;
                        }
                        if samples > pass.samples {
                            pass.samples = samples;
                        }
                    },
                    ResourceType::Transitioned { attachment, .. } => {
                        let root = self.get_attachment(&attachment);
                        if usage == ResourceUsage::Read {
                            continue;
                        }
                        if let ResourceType::Attachment { samples, .. } = root.resource {
                            if samples > pass.samples {
                                pass.samples = samples;
                            }
                        }
                    },
                    ResourceType::Texture { attachment, .. } => {
                        let root = self.get_attachment(&attachment);
                        if usage == ResourceUsage::Read {
                            continue;
                        }
                        if let ResourceType::Attachment { samples, .. } = root.resource {
                            if samples > pass.samples {
                                pass.samples = samples;
                            }
                        }
                    },
                }
            }
        }

        self.passes = passes;

        let name = "Render Pass";
        let mut subpasses = vec![];
        let mut subpass_data = vec![];
        let dependencies = vec![];
        let attachments = self.resources.clone();
        let swapchain_output = true;

        for (subpass_id, pass) in self.passes.iter().enumerate() {
            let bind_point = match pass.bind_point {
                BindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
            };

            let mut subpass = SubpassDescriptionData {
                bind_point,
                input_attachments: vec![],
                color_attachments: vec![],
                resolve_attachments: vec![],
                depth_stencil_attachment: vk::AttachmentReference::default(),
                preserve_attachments: vec![],
            };

            for resource_ref in &pass.resource_refs {
                let resource = self.resources.iter().find(|r| r.name == resource_ref.name).unwrap();

                let mut layout = match self.resources.iter().find(|r| r.name == resource_ref.name).unwrap().resource {
                    ResourceType::Attachment { layout, .. } => layout,
                    ResourceType::Transitioned { layout, .. } => layout,
                    ResourceType::Texture { layout, .. } => layout,
                };

                let samples = if let ResourceType::Attachment { samples, .. } = self.get_attachment(&resource_ref.name).resource {
                    samples
                } else {
                    vk::SampleCountFlags::TYPE_1
                };

                if layout == vk::ImageLayout::PRESENT_SRC_KHR {
                    layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                }

                let attachment_ref = vk::AttachmentReference::default()
                    .attachment(attachments.iter().position(|a| a.name == resource.name).unwrap() as u32)
                    .layout(layout);

                if is_depth_layout(layout) {
                    subpass.depth_stencil_attachment = attachment_ref;
                    continue;
                }

                match resource_ref.usage {
                    ResourceUsage::Read => {
                        subpass.input_attachments.push(attachment_ref);
                    },
                    ResourceUsage::Write => {
                        if samples == pass.samples {
                            subpass.color_attachments.push(attachment_ref);
                        } else {
                            subpass.resolve_attachments.push(attachment_ref);
                        }
                    },
                }
            }

            preserves[subpass_id].iter().for_each(|p| subpass.preserve_attachments.push(*p));

            subpass_data.push(subpass);
            let mut render_data = SubpassRenderData::new(
                0,
                subpass_id as u32,
                vec![],
                0,
                &pass.name,
            );
            render_data.setup_command_buffers(device, data);
            subpasses.push(
                render_data
            );
        }

        let mut render_passes = vec![RenderPass::new(
            name,
            vec![],
            subpass_data,
            dependencies,
            attachments,
            swapchain_output,
        )];

        render_passes.iter_mut().for_each(|rp| rp.recreate_swapchain(instance, device, data).unwrap());

        render_passes[0].subpasses = subpasses;

        render_passes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameGraphError {
    #[error("Resource {0} not created")]
    ResourceNotCreated(String),
}

#[derive(Clone)]
pub struct Pass {
    pub name: String,
    pub bind_point: BindPoint,
    pub resource_refs: Vec<ResourceRef>,
    pub created_resources: Vec<Resource>,

    pub samples: vk::SampleCountFlags,
}

impl Pass {
    pub fn new(name: &str, bind_point: BindPoint) -> Self {
        Pass {
            name: name.into(),
            bind_point,
            resource_refs: vec![],
            created_resources: vec![],
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    pub fn add_resource(&mut self, name: &str, usage: ResourceUsage) -> &mut Self {
        self.resource_refs.push(ResourceRef {
            name: name.into(),
            usage,
        });

        self
    }

    pub fn create_resource(&mut self, resource: Resource, usage: ResourceUsage) -> &mut Self {
        let name = resource.name.clone();
        self.created_resources.push(resource);
        self.resource_refs.push(ResourceRef {
            name,
            usage,
        });

        self
    }

    pub fn build(&mut self) -> Self {
        self.clone()
    }
}

#[derive(Clone)]
pub struct Resource {
    pub name: String,
    pub clear_value: Option<vk::ClearValue>,
    pub resource: ResourceType,

    pub width: u32,
    pub height: u32,
    pub description: Option<vk::AttachmentDescription>,
    pub image: Option<Image>,
    pub view: Option<vk::ImageView>,
}

impl Resource {
    pub fn new(name: &str, clear_value: Option<vk::ClearValue>, resource: ResourceType) -> Self {
        Resource {
            name: name.into(),
            clear_value,
            resource,
            width: 0,
            height: 0,
            description: None,
            image: None,
            view: None,
        }
    }

    pub fn generate_description(&mut self, resources: HashMap<String, Resource>) {
        match &self.resource {
            ResourceType::Attachment {
                width: _,
                height: _,
                layout,
                samples,
                format,
                tiling: _,
                usage: _,
                view_type: _,
                layer_count: _,
                aspects: _,
            } => {
                self.description = Some(vk::AttachmentDescription::default()
                    .format(*format)
                    .samples(*samples)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(*layout));
            },
            ResourceType::Transitioned {
                attachment,
                layout,
                format,
                aspects: _,
                view_type: _,
            } => {
                let attachment = resources.get(attachment).unwrap();

                self.description = Some(vk::AttachmentDescription::default()
                    .format(*format)
                    .samples(attachment.description.unwrap().samples)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(attachment.description.unwrap().final_layout)
                    .final_layout(*layout));
            },
            a => panic!("Resource type {:?} not supported", a),
        }
    }

    pub fn create_image(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        match &self.resource {
            ResourceType::Attachment {
                width,
                height,
                layout: _,
                samples,
                format,
                tiling,
                usage,
                view_type,
                layer_count,
                aspects,
            } => {
                let width = match width {
                    AttachmentSize::Absolute(w) => *w,
                    AttachmentSize::Relative(w) => (data.swapchain_extent.width as f32 * w) as u32,
                };

                self.width = width;

                let height = match height {
                    AttachmentSize::Absolute(h) => *h,
                    AttachmentSize::Relative(h) => (data.swapchain_extent.height as f32 * h) as u32,
                };

                self.height = height;

                let image = Image::new(
                    instance,
                    device,
                    data,
                    width,
                    height,
                    MipLevels::One,
                    *samples,
                    *format,
                    *tiling,
                    *usage,
                    *view_type,
                    *layer_count,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    *aspects,
                    false,
                );

                self.view = Some(image.view);
                self.image = Some(image);
            },
            a => panic!("Resource type {:?} not supported", a),
        }
    }

    pub fn destroy(&self, device: &Device) {
        match &self.resource {
            ResourceType::Attachment {
                width: _,
                height: _,
                layout: _,
                samples: _,
                format: _,
                tiling: _,
                usage: _,
                view_type: _,
                layer_count: _,
                aspects: _,
            } => {
                self.image.as_ref().unwrap().destroy(device);
            },
            a => panic!("Resource type {:?} not supported", a),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResourceType {
    Attachment {
        width: AttachmentSize,
        height: AttachmentSize,

        layout: vk::ImageLayout,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        view_type: vk::ImageViewType,
        layer_count: u32,
        aspects: vk::ImageAspectFlags,
    },
    Transitioned {
        attachment: String,
        
        layout: vk::ImageLayout,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    },
    Texture {
        attachment: String,

        layout: vk::ImageLayout,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    },
}

impl ResourceType {
    pub fn color() -> Self {
        ResourceType::Attachment {
            width: AttachmentSize::default(),
            height: AttachmentSize::default(),
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            view_type: vk::ImageViewType::TYPE_2D,
            layer_count: 1,
            aspects: vk::ImageAspectFlags::COLOR,
        }
    }

    pub fn depth() -> Self {
        ResourceType::Attachment {
            width: AttachmentSize::default(),
            height: AttachmentSize::default(),
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::D32_SFLOAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            view_type: vk::ImageViewType::TYPE_2D,
            layer_count: 1,
            aspects: vk::ImageAspectFlags::DEPTH,
        }
    }

    pub fn present() -> Self {
        ResourceType::Attachment {
            width: AttachmentSize::default(),
            height: AttachmentSize::default(),
            layout: vk::ImageLayout::PRESENT_SRC_KHR,
            samples: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::R8G8B8A8_SNORM,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            view_type: vk::ImageViewType::TYPE_2D,
            layer_count: 1,
            aspects: vk::ImageAspectFlags::COLOR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceRef {
    pub name: String,
    pub usage: ResourceUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceUsage {
    Read,
    Write,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindPoint {
    Graphics,
}

pub fn is_depth_layout(layout: vk::ImageLayout) -> bool {
    match layout {
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => true,
        vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL => true,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => true,
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => true,
        _ => false,
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
    attachments: Vec<Resource>,
}

impl RenderPass {
    pub fn new(
        name: &str,
        subpasses: Vec<SubpassRenderData>,
        subpass_data: Vec<SubpassDescriptionData>,
        dependencies: Vec<vk::SubpassDependency>,
        attachments: Vec<Resource>,
        swapchain_output: bool
    ) -> Self {
        let clear_values = attachments.iter().map(|r| match r.clear_value {
            Some(c) => c,
            None => vk::ClearValue::default(),
        }).collect::<Vec<_>>();

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
            width: 0,
            height: 0,
        }
    }

    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &mut RendererData) -> Result<()> {
        let subpass_descriptors = self.subpass_data.iter().map(|s| s.description()).collect::<Vec<vk::SubpassDescription>>();

        let hashmap = self.attachments.iter().map(|a| (a.name.clone(), a.clone())).collect::<HashMap<_, _>>();

        self.attachments.iter_mut().for_each(|attachment| attachment.generate_description(hashmap.clone()));
        self.attachments.iter_mut().for_each(|attachment| attachment.create_image(instance, device, &data));

        let attachment_descriptions = self.attachments.iter().map(|attachment| attachment.description.unwrap()).collect::<Vec<vk::AttachmentDescription>>();

        self.width = self.attachments.iter().map(|a| a.width ).max().unwrap();

        self.height = self.attachments.iter().map(|a| a.height ).max().unwrap();

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

        // the last attachment must be the swapchain image
        let attachment_images = if self.swapchain_output{
            self.attachments.iter().map(|a| a.view.unwrap()).take(self.attachments.len() -1).collect::<Vec<_>>()
        } else {
            self.attachments.iter().map(|a| a.view.unwrap()).collect::<Vec<_>>()
        };

        let framebuffers = data
            .swapchain_image_views
            .iter()
            .enumerate()
            .map(|(index, i)| {
                let mut views = attachment_images.iter().map(|view| *view).collect::<Vec<_>>();

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
                set_object_name(instance, device, &format!("{} Framebuffer({})", self.name, index), framebuffer.unwrap()).unwrap();
                framebuffer
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.framebuffers = framebuffers;
        self.render_pass = render_pass;

        self.subpasses.iter_mut().for_each(|s| s.recreate_swapchain(instance, device, data, self.render_pass));

        Ok(())
    }

    pub fn destroy_swapchain(&self, device: &Device, data: &mut RendererData) {
        self.attachments.iter().for_each(|r| r.destroy(device));
        self.framebuffers.iter().for_each(|f| unsafe { device.destroy_framebuffer(*f, None) });
        unsafe { device.destroy_render_pass(self.render_pass, None) };
        self.subpasses.iter().for_each(|s| s.destroy_swapchain(device, data));
    }
}
