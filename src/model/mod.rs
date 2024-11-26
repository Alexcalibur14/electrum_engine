#![allow(clippy::too_many_arguments)]

use glam::Mat4;
use vulkanalia::prelude::v1_2::*;

use crate::{create_and_stage_buffer, BufferWrapper, Loadable, RenderStats, RendererData};

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
    
    fn descriptor_set(&self, image_index: usize) -> vk::DescriptorSet;
    
    fn mesh_data(&self) -> &MeshData;
    
    /// Used for debugging
    /// Not used in release builds
    /// Can be empty
    fn name(&self) -> String;
    
    fn clone_dyn(&self) -> Box<dyn Renderable>;
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

#[derive(Debug, Clone, Copy)]
pub struct MeshData {
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_buffer: BufferWrapper,
    pub index_buffer: BufferWrapper,
}

impl MeshData {
    pub fn new<T>(instance: &Instance, device: &Device, data: &RendererData, vertices: &[T], indices: &[u32], name: &str) -> Self {
        let vertex_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            (size_of::<T>() * vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some(format!("{} Vertex Buffer", name)),
            vertices,
        ).unwrap();

        let vertex_count = vertices.len() as u32;

        let index_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            (size_of::<u32>() * indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            Some(format!("{} Index Buffer", name)),
            indices,
        ).unwrap();

        let index_count = indices.len() as u32;
        
        MeshData {
            vertex_count,
            index_count,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn destroy(&self, device: &Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            index_count: 0,
            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),
        }
    }
}
