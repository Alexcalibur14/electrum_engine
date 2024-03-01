use vulkanalia::{prelude::v1_2::*, vk::SampleLocationEXTBuilder};

pub trait Renderable {
    fn draw();
    fn destroy_swapchain();
    fn recreate_swapchain();
    fn destroy();
}

pub trait Vertex {
    fn binding_description(&self) -> vk::VertexInputBindingDescription;
    fn attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription>;
}