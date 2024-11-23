#![allow(clippy::too_many_arguments)]

use glam::Mat4;
use vulkanalia::prelude::v1_2::*;

use crate::{BufferWrapper, Loadable, RenderStats, RendererData};

pub mod mesh;
pub mod vertices;

pub trait Renderable {
    fn update(
        &mut self,
        device: &Device,
        data: &RendererData,
        stats: &RenderStats,
        index: usize,
        view_proj: Mat4,
    );
    fn destroy_swapchain(&self, device: &Device);
    fn recreate_swapchain(&mut self, device: &Device, data: &mut RendererData);
    fn destroy(&mut self, device: &Device);
    fn clone_dyn(&self) -> Box<dyn Renderable>;

    fn descriptor_set(&self, image_index: usize) -> vk::DescriptorSet;
    fn vertex_buffer(&self) -> BufferWrapper;
    fn index_buffer(&self) -> BufferWrapper;
    fn index_len(&self) -> u32;
    fn name(&self) -> String;

    fn loaded(&self) -> bool;
}

impl Clone for Box<dyn Renderable> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl Loadable for Box<dyn Renderable> {
    fn is_loaded(&self) -> bool {
        self.loaded()
    }
}

pub trait Vertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelMVP {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl ModelMVP {
    pub fn get_data(&self, view_proj: &Mat4) -> ModelData {
        ModelData {
            model: self.model,
            mvp: *view_proj * self.model,
            normal: self.model.inverse().transpose(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelData {
    model: Mat4,
    mvp: Mat4,
    normal: Mat4,
}
