use ash::{Device, Instance, vk};

use crate::{RendererData, buffer::BufferWrapper, image::{AddressMode, Filter, Image, MipLevels}};


pub struct TaskGraph<'a> {
    nodes: Vec<Node<'a>>,
    images: Vec<(ImageData<'a>, AccessType)>,
    buffers: Vec<(&'a str, BufferWrapper, AccessType)>,
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

    pub fn add_buffer(&mut self, name: &'a str, buffer: BufferWrapper) {
        self.buffers.push((name, buffer, AccessType::UNDEFINED));
    }

    pub fn destroy(&mut self, device: &Device) {
        self.images.iter_mut().for_each(|(image_data, access_type)| {
            image_data.destroy(device);
            *access_type = AccessType::UNDEFINED;
        });
        self.buffers.iter_mut().for_each(|(_, buffer, access_type)| {
            buffer.destroy(device);
            *access_type = AccessType::UNDEFINED;
        });
        self.nodes.iter_mut().for_each(|node| node.destroy(device));
    }

    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        self.nodes.iter_mut().for_each(|node| node.recreate_swapchain(instance, device, data));
        
        if self.swapchain_extent == data.swapchain_extent {
            return;
        }

        self.images.iter_mut().for_each(|(image_data, access_type)| {
            if image_data.recreate_swapchain(instance, device, data) {
                *access_type = AccessType::UNDEFINED
            }
        });
    }
}

pub struct Node<'a> {
    color_attachments: Vec<(&'a str, AccessType)>,
    depth_attachment: Option<(&'a str, AccessType)>,
    stencil_attachment: Option<(&'a str, AccessType)>,
    internal_images: Vec<(&'a str, AccessType)>,
    internal_buffers: Vec<(&'a str, AccessType)>,
    external_images: Vec<(&'a str, AccessType)>,
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

    pub fn add_internal_image(&mut self, name: &'a str, access_type: AccessType) {
        self.internal_images.push((name, access_type));
    }

    pub fn add_internal_buffer(&mut self, name: &'a str, access_type: AccessType) {
        self.internal_buffers.push((name, access_type));
    }

    /// `access_type` currently does nothing
    pub fn add_external_image(&mut self, name: &'a str, access_type: AccessType) {
        self.external_images.push((name, access_type));
    }

    /// `access_type` currently does nothing
    pub fn add_external_buffer(&mut self, name: &'a str, access_type: AccessType) {
        self.external_buffers.push((name, access_type));
    }

    pub fn set_task(&mut self, task: Box<dyn Task>) {
        self.task = task
    }

    pub fn execute(&mut self, device: &Device, command_buffer: vk::CommandBuffer, draw_data: DrawData<'a>, data: &RendererData) {
        self.task.execute(device, command_buffer, draw_data, data);
    }

    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        self.task.recreate_swapchain(instance, device, data);
    }

    pub fn destroy(&mut self, device: &Device) {
        self.task.destroy(device);
    }
}

pub trait Task {
    fn execute<'a>(&mut self, device: &Device, command_buffer: vk::CommandBuffer, draw_data: DrawData<'a>, data: &RendererData);
    fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData);
    fn destroy(&mut self, device: &Device);
}

pub struct DrawData<'a> {
    pub color_attachments: Vec<&'a Image>,
    pub depth_attachment: Option<&'a Image>,
    pub stencil_attachment: Option<&'a Image>,
    pub internal_images: Vec<&'a Image>,
    pub internal_buffers: Vec<&'a BufferWrapper>,
    pub external_images: Vec<&'a Image>,
    pub external_buffers: Vec<&'a BufferWrapper>,
}

pub struct DummyTask;

impl Task for DummyTask {
    fn execute<'a>(&mut self, _: &Device, _: vk::CommandBuffer, _: DrawData<'a>, _: &RendererData) {}
    fn recreate_swapchain(&mut self, _: &Instance, _: &Device, _: &RendererData) {}
    fn destroy(&mut self, _: &Device) {}
}

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

pub struct ImageData<'a> {
    image: Image,
    size: ImageSize,
    format: vk::Format,
    mip_levels: MipLevels,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    view_type: vk::ImageViewType,
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
        view_type: vk::ImageViewType,
        layers: u32,
        aspect: vk::ImageAspectFlags,
        sampler: Option<(Filter, AddressMode)>,
        name: &'a str,
    ) -> Self {
        ImageData {
            image: Image {
                image: vk::Image::null(),
                image_memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                mip_level: 0,
                sampler: None,
            },
            size,
            format,
            mip_levels: mips,
            usage,
            samples,
            view_type,
            layers,
            aspect,
            sampler,
            name,
        }
    }

    /// Generates Image based on struct data
    pub fn create(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        let (width, height) = self.size.size(data);
        
        self.image = Image::new(
            instance,
            device,
            data,
            width,
            height,
            self.mip_levels,
            self.samples,
            self.format,
            vk::ImageTiling::OPTIMAL,
            self.usage,
            self.view_type,
            self.layers,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            self.aspect,
            self.sampler,
            self.name
        )
    }

    /// Recreates image only if size is not fixed \
    /// Returns true if image was recreated
    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData) -> bool {
        match self.size {
            ImageSize::Swapchain => {
                self.destroy(device);
                self.create(instance, device, data);
                true
            },
            ImageSize::SwapchainRelative {..} => {
                self.destroy(device);
                self.create(instance, device, data);
                true
            },
            ImageSize::Fixed {..} => false,
        }
    }

    /// Destroys image
    pub fn destroy(&self, device: &Device) {
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
