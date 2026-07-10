use std::ops::BitOr;

use ash::vk;

use crate::{RenderStats, RendererData, RenderingDevice, buffer::Buffer, image::{AddressMode, Filter, Image, MipLevels}, present::image_subresource_range};

#[derive(Clone)]
pub struct TaskGraph<'a> {
    nodes: Vec<Node<'a>>,
    images: Vec<(ImageData<'a>, AccessType)>,
    buffers: Vec<(&'a str, Buffer, AccessType)>,
    swapchain_extent: vk::Extent2D,
}

impl<'a> TaskGraph<'a> {
    pub fn new() -> Self {
        TaskGraph {
            nodes: vec![],
            images: vec![],
            buffers: vec![],
            swapchain_extent: vk::Extent2D::default(),
        }
    }

    pub fn add_node(&mut self, node: Node<'a>) {
        self.nodes.push(node);
    }

    pub fn add_image(&mut self, image: ImageData<'a>) {
        self.images.push((image, AccessType::UNDEFINED));
    }

    pub fn add_buffer(&mut self, name: &'a str, buffer: Buffer) {
        self.buffers.push((name, buffer, AccessType::UNDEFINED));
    }

    pub fn create_images(&mut self, device: &RenderingDevice, data: &RendererData) {
        self.images.iter_mut().for_each(|(image_data, _)| image_data.create(device, data));
        self.swapchain_extent = data.swapchain_extent;
    }

    pub fn destroy(&mut self, device: &RenderingDevice) {
        self.images.iter_mut().for_each(|(image_data, access_type)| {
            image_data.destroy(device);
            *access_type = AccessType::UNDEFINED;
        });
        self.buffers.iter_mut().for_each(|(_, buffer, access_type)| {
            buffer.destroy(device);
            *access_type = AccessType::UNDEFINED;
        });
        self.nodes.iter_mut().for_each(|node| node.destroy(device));
        
        self.swapchain_extent = vk::Extent2D::default();
    }

    pub fn recreate_swapchain(&mut self, device: &RenderingDevice, data: &mut RendererData) {
        if self.swapchain_extent == data.swapchain_extent {
            return;
        }
        
        self.images.iter_mut().for_each(|(image_data, access_type)| {
            let resized = image_data.recreate_swapchain(device, data);
            
            if resized {
                *access_type = AccessType::UNDEFINED
            }
        });

        self.swapchain_extent = data.swapchain_extent;
        self.nodes.iter_mut().for_each(|node| {

            let color_attachments = node.color_attachments.iter().map(|(name, dst_access)| {
                let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == *name).expect(&format!("could not find color attachment with name: {:?}", name));
                (image_data, src_access, dst_access)
            }).collect::<Vec<_>>();

            let depth_attachment = {
                match node.depth_attachment {
                    Some((name, dst_access)) => {
                        let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == name).expect(&format!("could not find depth attachment with name: {:?}", name));
                        Some((image_data, src_access, dst_access))
                    },
                    None => None,
                }
            };

            let stencil_attachment = {
                match node.stencil_attachment {
                    Some((name, dst_access)) => {
                        let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == name).expect(&format!("could not find stencil attachment with name: {:?}", name));
                        Some((image_data, src_access, dst_access))
                    },
                    None => None,
                }
            };

            let internal_images = node.internal_images.iter().map(|(name, dst_access, aspect)| {
                let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == *name).expect(&format!("could not find internal image with name: {:?}", name));
                (image_data, src_access, dst_access, aspect)
            }).collect::<Vec<_>>();

            let internal_buffers = node.internal_buffers.iter().map(|(name, dst_access)| {
                let (_, buffer, src_access) = self.buffers.iter().find(|(buffer_name, _, _)| buffer_name == name).expect(&format!("could not find internal buffer with name: {:?}", name));
                (buffer, src_access, dst_access)
            }).collect::<Vec<_>>();

            let draw_data = DrawData {
                color_attachments: color_attachments.iter().map(|(image_data, _, _)| image_data.image).collect(),
                depth_attachment: if let Some((image_data, _, _)) = depth_attachment { Some(image_data.image) } else { None },
                stencil_attachment: if let Some((image_data, _, _)) = stencil_attachment { Some(image_data.image) } else { None },
                internal_images: internal_images.iter().map(|(image_data, _, _, _)| image_data.image).collect(),
                internal_buffers: internal_buffers.iter().map(|(buffer, _, _)| **buffer).collect(),
                external_images: vec![],
                external_buffers: vec![],
            };
        
            node.recreate_swapchain(device, data, &draw_data);
        });
    }

    pub fn execute(&mut self, device: &RenderingDevice, command_buffer: vk::CommandBuffer, data: &mut RendererData, stats: &mut RenderStats) {
        for node in self.nodes.iter() {
            let color_attachments = node.color_attachments.iter().map(|(name, dst_access)| {
                let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == *name).expect(&format!("could not find color attachment with name: {:?}", name));
                (image_data, src_access, dst_access)
            }).collect::<Vec<_>>();

            let depth_attachment = {
                match node.depth_attachment {
                    Some((name, dst_access)) => {
                        let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == name).expect(&format!("could not find depth attachment with name: {:?}", name));
                        Some((image_data, src_access, dst_access))
                    },
                    None => None,
                }
            };

            let stencil_attachment = {
                match node.stencil_attachment {
                    Some((name, dst_access)) => {
                        let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == name).expect(&format!("could not find stencil attachment with name: {:?}", name));
                        Some((image_data, src_access, dst_access))
                    },
                    None => None,
                }
            };

            let internal_images = node.internal_images.iter().map(|(name, dst_access, aspect)| {
                let (image_data, src_access) = self.images.iter().find(|(image_data, _)| image_data.name == *name).expect(&format!("could not find internal image with name: {:?}", name));
                (image_data, src_access, dst_access, aspect)
            }).collect::<Vec<_>>();

            let internal_buffers = node.internal_buffers.iter().map(|(name, dst_access)| {
                let (_, buffer, src_access) = self.buffers.iter().find(|(buffer_name, _, _)| buffer_name == name).expect(&format!("could not find internal buffer with name: {:?}", name));
                (buffer, src_access, dst_access)
            }).collect::<Vec<_>>();

            let mut image_barriers = vec![];

            color_attachments.iter().for_each(|(image_data, src_access, dst_access)| {
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(image_data.image.image())
                        .old_layout(src_access.image_layout)
                        .new_layout(dst_access.image_layout)
                        .src_access_mask(src_access.access_mask)
                        .dst_access_mask(dst_access.access_mask)
                        .src_stage_mask(src_access.pipeline_stage)
                        .dst_stage_mask(dst_access.pipeline_stage)
                        .subresource_range(image_subresource_range(vk::ImageAspectFlags::COLOR))
                );
            });

            if let Some((image_data, src_access, dst_access)) = depth_attachment {
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(image_data.image.image())
                        .old_layout(src_access.image_layout)
                        .new_layout(dst_access.image_layout)
                        .src_access_mask(src_access.access_mask)
                        .dst_access_mask(dst_access.access_mask)
                        .src_stage_mask(src_access.pipeline_stage)
                        .dst_stage_mask(dst_access.pipeline_stage)
                        .subresource_range(image_subresource_range(vk::ImageAspectFlags::DEPTH))
                );
            }

            if let Some((image_data, src_access, dst_access)) = stencil_attachment {
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(image_data.image.image())
                        .old_layout(src_access.image_layout)
                        .new_layout(dst_access.image_layout)
                        .src_access_mask(src_access.access_mask)
                        .dst_access_mask(dst_access.access_mask)
                        .src_stage_mask(src_access.pipeline_stage)
                        .dst_stage_mask(dst_access.pipeline_stage)
                        .subresource_range(image_subresource_range(vk::ImageAspectFlags::STENCIL))
                );
            }

            internal_images.iter().for_each(|(image_data, src_access, dst_access, aspect)| {
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(image_data.image.image())
                        .old_layout(src_access.image_layout)
                        .new_layout(dst_access.image_layout)
                        .src_access_mask(src_access.access_mask)
                        .dst_access_mask(dst_access.access_mask)
                        .src_stage_mask(src_access.pipeline_stage)
                        .dst_stage_mask(dst_access.pipeline_stage)
                        .subresource_range(image_subresource_range(**aspect))
                );
            });

            let mut buffer_barriers = Vec::with_capacity(internal_buffers.len());

            internal_buffers.iter().for_each(|(buffer, src_access, dst_access)| {
                buffer_barriers.push(
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer.buffer())
                        .src_access_mask(src_access.access_mask)
                        .dst_access_mask(dst_access.access_mask)
                        .src_stage_mask(src_access.pipeline_stage)
                        .dst_stage_mask(dst_access.pipeline_stage)
                        .size(buffer.size())
                );
            });

            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(&buffer_barriers)
                .image_memory_barriers(&image_barriers);

            unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };

            let draw_data = DrawData {
                color_attachments: color_attachments.iter().map(|(image_data, _, _)| image_data.image).collect(),
                depth_attachment: if let Some((image_data, _, _)) = depth_attachment { Some(image_data.image) } else { None },
                stencil_attachment: if let Some((image_data, _, _)) = stencil_attachment { Some(image_data.image) } else { None },
                internal_images: internal_images.iter().map(|(image_data, _, _, _)| image_data.image).collect(),
                internal_buffers: internal_buffers.iter().map(|(buffer, _, _)| **buffer).collect(),
                external_images: vec![],
                external_buffers: vec![],
            };

            node.execute(device, command_buffer, &draw_data, data, stats);

            let mut images_clone = self.images.clone();
            
            node.color_attachments.iter().for_each(|(name, dst_access)| {
                let (_, latest_access) = images_clone.iter_mut().find(|(image_data, _)| image_data.name == *name).expect("msg");
                *latest_access = *dst_access;
            });

            if let Some((name, dst_access)) = node.depth_attachment {
                let (_, latest_access) = images_clone.iter_mut().find(|(image_data, _)| image_data.name == name).expect("msg");
                *latest_access = dst_access;
            }

            if let Some((name, dst_access)) = node.stencil_attachment {
                let (_, latest_access) = images_clone.iter_mut().find(|(image_data, _)| image_data.name == name).expect("msg");
                *latest_access = dst_access;
            }

            node.internal_images.iter().for_each(|(name, dst_access, _)| {
                let (_, latest_access) = images_clone.iter_mut().find(|(image_data, _)| image_data.name == *name).expect("msg");
                *latest_access = *dst_access;
            });
            
            self.images = images_clone;

            let mut buffers_clone = self.buffers.clone();
            
            node.internal_buffers.iter().for_each(|(name, dst_access)| {
                let (_, _, latest_access) = buffers_clone.iter_mut().find(|(buffer_name, _, _)| buffer_name == name).expect("msg");
                *latest_access = *dst_access;
            });
            
            self.buffers = buffers_clone;
        }
    }

    pub fn images(&self) -> &[(ImageData<'a>, AccessType)] {
        &self.images
    }

    pub fn buffers(&self) -> &[(&'a str, Buffer, AccessType)] {
        &self.buffers
    }
}

#[derive(Clone)]
pub struct Node<'a> {
    color_attachments: Vec<(&'a str, AccessType)>,
    depth_attachment: Option<(&'a str, AccessType)>,
    stencil_attachment: Option<(&'a str, AccessType)>,
    internal_images: Vec<(&'a str, AccessType, vk::ImageAspectFlags)>,
    internal_buffers: Vec<(&'a str, AccessType)>,
    external_images: Vec<(&'a str, AccessType, vk::ImageAspectFlags)>,
    external_buffers: Vec<(&'a str, AccessType)>,
    task: Box<dyn Task>,
}

impl<'a> Node<'a> {
    pub fn new() -> Self {
        Node {
            color_attachments: vec![],
            depth_attachment: None,
            stencil_attachment: None,
            internal_images: vec![],
            internal_buffers: vec![],
            external_images: vec![],
            external_buffers: vec![],
            task: Box::new(DummyTask),
        }
    }

    pub fn add_attachment(&mut self, name: &'a str, access_type: AccessType) {
        self.color_attachments.push((name, access_type));
    }

    pub fn set_depth(&mut self, name: &'a str, access_type: AccessType) {
        self.depth_attachment = Some((name, access_type));
    }

    pub fn set_stencil(&mut self, name: &'a str, access_type: AccessType) {
        self.stencil_attachment = Some((name, access_type));
    }

    pub fn add_internal_image(&mut self, name: &'a str, aspect: vk::ImageAspectFlags, access_type: AccessType) {
        self.internal_images.push((name, access_type, aspect));
    }

    pub fn add_internal_buffer(&mut self, name: &'a str, access_type: AccessType) {
        self.internal_buffers.push((name, access_type));
    }

    /// `access_type` currently does nothing
    pub fn add_external_image(&mut self, name: &'a str, aspect: vk::ImageAspectFlags, access_type: AccessType) {
        self.external_images.push((name, access_type, aspect));
    }

    /// `access_type` currently does nothing
    pub fn add_external_buffer(&mut self, name: &'a str, access_type: AccessType) {
        self.external_buffers.push((name, access_type));
    }

    pub fn set_task(&mut self, task: Box<dyn Task>) {
        self.task = task
    }

    pub fn execute(&self, device: &RenderingDevice, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats) {
        self.task.execute(device, command_buffer, draw_data, data, stats);
    }

    pub fn recreate_swapchain(&mut self, device: &RenderingDevice, data: &mut RendererData, draw_data: &DrawData) {
        self.task.recreate_swapchain(device, data, draw_data);
    }

    pub fn destroy(&mut self, device: &RenderingDevice) {
        self.task.destroy(device);
    }
}

pub trait Task: TaskClone {
    fn execute<'a>(&self, device: &RenderingDevice, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats);
    fn recreate_swapchain<'a>(&mut self, device: &RenderingDevice, data: &mut RendererData, draw_data: &DrawData);
    fn destroy(&mut self, device: &RenderingDevice);
}

pub trait TaskClone {
    fn clone_box(&self) -> Box<dyn Task>;
}

impl<T> TaskClone for T
where 
    T: 'static + Task + Clone,
{
    fn clone_box(&self) -> Box<dyn Task> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Task> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub struct DrawData {
    pub color_attachments: Vec<Image>,
    pub depth_attachment: Option<Image>,
    pub stencil_attachment: Option<Image>,
    pub internal_images: Vec<Image>,
    pub internal_buffers: Vec<Buffer>,
    pub external_images: Vec<Image>,
    pub external_buffers: Vec<Buffer>,
}

#[derive(Debug, Clone, Copy)]
pub struct DummyTask;

impl Task for DummyTask {
    fn execute<'a>(&self, _: &RenderingDevice, _: vk::CommandBuffer, _: &DrawData, _: &mut RendererData, _: &mut RenderStats) {}
    fn recreate_swapchain<'a>(&mut self, _: &RenderingDevice, _: &mut RendererData, _: &DrawData) {}
    fn destroy(&mut self, _: &RenderingDevice) {}
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AccessType {
    pub pipeline_stage: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
    pub image_layout: vk::ImageLayout,
}

impl AccessType {
    pub fn index_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::INDEX_INPUT,
            access_mask: vk::AccessFlags2::INDEX_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        }
    }

    pub fn vertex_attribute_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        }
    }
    
    // ----- Vertex Shader ----- //
    pub fn vertex_shader_uniform_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        }
    }
    
    pub fn vertex_shader_sampled_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
    
    pub fn vertex_shader_storage_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    
    pub fn vertex_shader_storage_write() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    

    // ----- Fragment Shader ----- //
    pub fn fragment_shader_uniform_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        }
    }
    
    pub fn fragment_shader_sampled_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
    
    pub fn fragment_shader_storage_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    
    pub fn fragment_shader_storage_write() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    
    pub fn fragment_shader_color_input_attachment_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }
    
    pub fn fragment_shader_depth_stencil_input_attachment_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }
    
    // ----- Depth Stencil Attachment ----- //
    pub fn depth_stencil_attachment_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }
    
    pub fn depth_stencil_attachment_write() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }
    
    pub fn depth_attachment_stencil_read_only() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        }
    }
    
    pub fn depth_read_only_stencil_attachment() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }
    
    // ----- Color Attachment ----- //
    pub fn color_attachment_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    pub fn color_attachment_write() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    // ----- Transfer ----- //
    pub fn transfer_src() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::TRANSFER,
            access_mask: vk::AccessFlags2::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    pub fn transfer_dst() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::TRANSFER,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        }
    }

    pub fn copy_src() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COPY,
            access_mask: vk::AccessFlags2::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    pub fn copy_dst() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COPY,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        }
    }

    pub fn blit_src() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::BLIT,
            access_mask: vk::AccessFlags2::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    pub fn blit_dst() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::BLIT,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        }
    }

    pub fn resolve_src() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::RESOLVE,
            access_mask: vk::AccessFlags2::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    pub fn resolve_dst() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::RESOLVE,
            access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        }
    }

    // ----- Compute ----- //
    pub fn compute_shader_uniform_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        }
    }
    
    pub fn compute_shader_sampled_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
    
    pub fn compute_shader_storage_read() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_READ,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    
    pub fn compute_shader_storage_write() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }

    /// Swapchain final layout
    pub fn present_src() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            access_mask: vk::AccessFlags2::MEMORY_READ,
            image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
    
    /// Not recomended for prduction \
    /// please only use during prototyping
    pub fn general() -> Self {
        AccessType {
            pipeline_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            access_mask: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        }
    }
    
    /// Used internally for the initial state of resources
    pub const UNDEFINED: AccessType = AccessType {
        pipeline_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
        access_mask: vk::AccessFlags2::NONE,
        image_layout: vk::ImageLayout::UNDEFINED,
    };
}

impl BitOr for AccessType {
    type Output = AccessType;

    fn bitor(self, rhs: Self) -> Self::Output {
        AccessType {
            pipeline_stage: self.pipeline_stage | rhs.pipeline_stage,
            access_mask: self.access_mask | rhs.access_mask,
            image_layout: if self.image_layout == rhs.image_layout { self.image_layout } else { vk::ImageLayout::UNDEFINED },
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ImageData<'a> {
    image: Image,
    size: ImageSize,
    format: vk::Format,
    mip_levels: MipLevels,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    layers: u32,
    aspect: vk::ImageAspectFlags,
    sampler: Option<(Filter, AddressMode)>,
    name: &'a str
}

impl<'a> ImageData<'a> {
    pub fn new(
        size: ImageSize,
        format: vk::Format,
        mips: MipLevels,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
        layers: u32,
        aspect: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &'a str,
    ) -> Self {
        ImageData {
            image: unsafe { Image::null() },
            size,
            format,
            mip_levels: mips,
            usage,
            samples,
            layers,
            aspect,
            sampler,
            name,
        }
    }

    /// Generates Image based on struct data
    pub fn create(&mut self, device: &RenderingDevice, data: &RendererData) {
        let (width, height) = self.size.size(data);
        
        self.image = Image::new_regular(
            device,
            data,
            vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            self.mip_levels,
            self.samples,
            self.format,
            vk::ImageTiling::OPTIMAL,
            self.usage,
            self.layers,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            self.aspect,
            self.sampler,
            self.name
        )
    }

    /// Recreates image only if size is not fixed \
    /// Returns true if image was recreated
    pub fn recreate_swapchain(&mut self, device: &RenderingDevice, data: &RendererData) -> bool {
        match self.size {
            ImageSize::Swapchain => {
                self.destroy(device);
                self.create(device, data);
                true
            },
            ImageSize::SwapchainRelative {..} => {
                self.destroy(device);
                self.create(device, data);
                true
            },
            ImageSize::Fixed {..} => false,
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn image(&self) -> &Image {
        &self.image
    }

    /// Destroys image
    pub fn destroy(&mut self, device: &RenderingDevice) {
        self.image.destroy(device);
    }
}


#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum ImageSize {
    #[default]
    Swapchain,
    SwapchainRelative {
        multiplier: f32
    },
    Fixed {
        width: u32,
        height: u32
    },
}

impl ImageSize {
    pub fn size(&self, data: &RendererData) -> (u32, u32) {
        match self {
            ImageSize::Swapchain => {
                let swapchain_extent = data.swapchain_extent;
                (swapchain_extent.width, swapchain_extent.height)
            },
            ImageSize::SwapchainRelative { multiplier } => {
                let swapchain_extent = data.swapchain_extent;
                ((swapchain_extent.width as f32 * multiplier) as u32, (swapchain_extent.height as f32 * multiplier) as u32)  
            },
            ImageSize::Fixed { width, height } => (*width, *height),
        }
    }
}

pub fn transition_image_access(device: &RenderingDevice, command_buffer: vk::CommandBuffer, image: vk::Image, src_access: AccessType, dst_access: AccessType, aspect: vk::ImageAspectFlags) {
    let image_barrier = [vk::ImageMemoryBarrier2::default()
        .src_access_mask(src_access.access_mask)
        .src_stage_mask(src_access.pipeline_stage)
        .dst_access_mask(dst_access.access_mask)
        .dst_stage_mask(dst_access.pipeline_stage)
        .old_layout(src_access.image_layout)
        .new_layout(dst_access.image_layout)
        .subresource_range(image_subresource_range(aspect))
        .image(image)];

    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(&image_barrier);
    
    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
}
