use ash::vk;

use crate::{RendererData, image::{Image, MipLevels}};


pub struct AttchmentInfo {
    pub size: AttachmentSize,
    pub format: vk::Format,
    pub name: String,
    pub samples: u32,
    pub layers: u32,
    pub levels: MipLevels,
}

pub enum AttachmentSize {
    Swapchain,
    SwapchainRelative {x: f32, y: f32},
    Fixed {width: u32, height: u32},
}

impl AttachmentSize {
    fn size(&self, data: &RendererData) -> (u32, u32) {
        match self {
            AttachmentSize::Swapchain => {
                let extent = data.swapchain_extent;
                (extent.width, extent.height)
            },
            AttachmentSize::SwapchainRelative { x, y } => {
                let extent = data.swapchain_extent;
                ((extent.width as f32 * x) as u32, (extent.height as f32 * y) as u32)
            },
            AttachmentSize::Fixed {width, height } => (*width, *height),
        }
    }
}

pub struct Attachment {
    image: Image,
    width: u32,
    height: u32,
}



pub struct RenderGraph {
    nodes: Vec<Node>
}

pub struct Node {
    
}

impl Node {
    pub fn add_attachment() {
        
    }
}
